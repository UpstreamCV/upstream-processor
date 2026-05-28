"""P6-D — End-to-End Geometry Lineage Verification.

Repeatable CLI proof that operator-saved geometry flowed all the way from the
dashboard editor → api-server → processor → nvdsanalytics runtime → emitted
observations. Reads three witnesses from the api-server (no shell access to
the processor host required):

1. ``GET /v1/cameras/{id}/processor-status/latest`` — confirms
   ``payload.applied_geometry.source = "operator_saved"`` and the matching
   revision/line_count/roi_count.
2. ``GET /v1/cameras/{id}/observations/recent?type=<...>`` for each of
   ``processor.line_crossing`` / ``processor.roi_occupancy`` /
   ``processor.zone_entered`` / ``processor.dwell_started`` — confirms each
   emitted row carries ``payload.source.geometry_source = "operator_saved"``,
   ``payload.source.geometry_revision`` matches expected, and
   ``payload.line_id`` / ``payload.roi_id`` is in the expected-slugs set.
3. ``GET /v1/cameras/{id}/perception/latest`` — independent witness from a
   different table; the snapshot embeds the applied geometry so we can
   verify ``snapshot.geometry.lines[].id`` and ``snapshot.geometry.rois[].id``.

Optional fourth witness: ``--analytics-config-path`` parses the nvdsanalytics
INI on the processor host so we can pin that the saved slugs are in the
runtime config and that the probe-default ROI is NOT present alongside them.

Exit codes (machine-checkable):

- 0  PASS      — every witness checked passes
- 1  PARTIAL   — status + ROI/zone/dwell + snapshot pass, but no
                 line_crossing rows in window (most common: operator walked
                 parallel to a vertical line — not a P6-C bug)
- 2  FAIL      — lineage mismatch (geometry_source = default, revision
                 mismatch, slug not in expected set, probe-default leaking
                 into runtime INI)
- 3  BLOCKED   — processor-status 404 (no run yet), HTTP 4xx/5xx, missing
                 ``UPSTREAM_API_TOKEN``

Secrets / safety:

- ``UPSTREAM_API_TOKEN`` is sourced from env only; never printed.
- ``--debug`` dumps JSON payloads with ``rtsp://…`` substrings scrubbed via
  ``app.redact.scrub_rtsp_in_text``.

Usage::

    UPSTREAM_API_TOKEN=... uv run python scripts/verify_geometry_lineage.py \\
        --camera-id 7b8313e5-155c-498f-9cc5-7af613d06074 \\
        --expected-revision 9 \\
        --expected-lines entry-line,line-1 \\
        --expected-rois roi-1 \\
        --since 2026-05-27T15:30:00Z

The accompanying live checklist lives at
``docs/p6-d-geometry-lineage-verification.md``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from enum import IntEnum
from pathlib import Path
from typing import Any

import httpx

# The verify script is intentionally self-contained — it does NOT import from
# ``app.api_client`` (that's the webhook-authed processor client; this script
# uses caller/Bearer auth and reads). The one app.* import is for RTSP
# scrubbing on --debug, where leaking session keys would be unsafe.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.redact import scrub_rtsp_in_text

_log = logging.getLogger("p6d.verify_geometry_lineage")

# Observation types we expect from the analytics worker. line_crossing is
# treated specially (absence → PARTIAL, not FAIL).
_OBS_TYPE_LINE = "processor.line_crossing"
_OBS_TYPE_ROI = "processor.roi_occupancy"
_OBS_TYPE_ZONE = "processor.zone_entered"
_OBS_TYPE_DWELL = "processor.dwell_started"
_ROI_OBS_TYPES = (_OBS_TYPE_ROI, _OBS_TYPE_ZONE, _OBS_TYPE_DWELL)
_ALL_OBS_TYPES = (_OBS_TYPE_LINE, *_ROI_OBS_TYPES)

# The probe-default ROI key the processor emits when no operator geometry
# exists. If we see this in the runtime INI alongside operator slugs, the
# resolver leaked the fallback — a hard FAIL.
_PROBE_DEFAULT_ROI_LINE = "roi-entry-roi=64;36;1216;36;1216;684;64;684"


class Verdict(IntEnum):
    PASS = 0
    PARTIAL = 1
    FAIL = 2
    BLOCKED = 3


@dataclass
class CheckResult:
    """One witness's outcome plus the bullet points to print under it.

    ``verdict`` defaults to PASS so check functions can build the result up
    front and only flip it when they hit a failure condition — the typical
    happy-path-by-default shape. Aggregation never reads a default-initialized
    result; every code path assigns ``verdict`` before the result reaches
    ``aggregate_verdict``.
    """

    name: str
    verdict: Verdict = Verdict.PASS
    details: list[str] = field(default_factory=list)

    def add(self, line: str) -> None:
        self.details.append(line)


# ---------------------------------------------------------------------------
# HTTP fetchers — thin async wrappers around the three (optionally four)
# read endpoints. Return ``None`` when the resource is missing (404 with the
# expected ``code``), raise on auth/connectivity errors.
# ---------------------------------------------------------------------------


class FetchError(Exception):
    """Wraps an underlying HTTP/transport failure with a human-readable hint."""

    def __init__(self, message: str, *, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


async def fetch_processor_status(
    client: httpx.AsyncClient,
    *,
    camera_id: str,
    stream_session_id: str | None,
) -> dict[str, Any] | None:
    params: dict[str, str] = {}
    if stream_session_id is not None:
        params["stream_session_id"] = stream_session_id
    r = await client.get(f"/v1/cameras/{camera_id}/processor-status/latest", params=params)
    if r.status_code == 404:
        code = r.json().get("error", {}).get("code")
        if code == "processor_status_not_found":
            return None
    if r.status_code != 200:
        raise FetchError(_summarize_http_error(r, "processor-status"), status=r.status_code)
    return r.json().get("status")


async def fetch_observations(
    client: httpx.AsyncClient,
    *,
    camera_id: str,
    type_: str,
    limit: int = 200,
) -> list[dict[str, Any]]:
    r = await client.get(
        f"/v1/cameras/{camera_id}/observations/recent",
        params={"type": type_, "limit": str(limit)},
    )
    if r.status_code != 200:
        raise FetchError(_summarize_http_error(r, "observations"), status=r.status_code)
    return r.json().get("observations", [])


async def fetch_perception_snapshot(
    client: httpx.AsyncClient,
    *,
    camera_id: str,
) -> dict[str, Any] | None:
    r = await client.get(f"/v1/cameras/{camera_id}/perception/latest")
    if r.status_code == 404:
        code = r.json().get("error", {}).get("code")
        if code == "perception_snapshot_not_found":
            return None
    if r.status_code != 200:
        raise FetchError(_summarize_http_error(r, "perception"), status=r.status_code)
    return r.json().get("snapshot")


def _summarize_http_error(r: httpx.Response, what: str) -> str:
    """Build a short error hint without leaking response bodies / headers."""
    try:
        code = r.json().get("error", {}).get("code")
    except (ValueError, json.JSONDecodeError):
        code = None
    base = f"{what} fetch failed: HTTP {r.status_code}"
    return f"{base} ({code})" if code else base


# ---------------------------------------------------------------------------
# Pure verdict logic — no I/O. Tested independently from the HTTP layer.
# ---------------------------------------------------------------------------


def check_processor_status(
    row: dict[str, Any] | None,
    *,
    expected_revision: int,
    expected_lines: list[str] | None,
    expected_rois: list[str] | None,
) -> CheckResult:
    """Verifies ``payload.applied_geometry`` against expectations.

    Missing row → BLOCKED (the processor never reported — start a session
    first). Source mismatch / revision mismatch / wrong counts → FAIL.
    """
    result = CheckResult(name="Processor status")
    if row is None:
        result.verdict = Verdict.BLOCKED
        result.add(
            "no processor status row for this camera — start a session "
            "(or pass --stream-session-id if you ran multiple)"
        )
        return result

    payload = row.get("payload") or {}
    applied = payload.get("applied_geometry")
    if not isinstance(applied, dict):
        result.verdict = Verdict.FAIL
        result.add("payload.applied_geometry missing — processor predates P6-C lineage stamping")
        return result

    source = applied.get("source")
    revision = applied.get("revision")
    line_count = applied.get("line_count")
    roi_count = applied.get("roi_count")
    error_code = row.get("error_code")

    failures: list[str] = []
    if source != "operator_saved":
        failures.append(f"source={source!r} (expected 'operator_saved')")
    if revision != expected_revision:
        failures.append(f"revision={revision} (expected {expected_revision})")
    if expected_lines is not None and line_count != len(expected_lines):
        failures.append(f"line_count={line_count} (expected {len(expected_lines)})")
    if expected_rois is not None and roi_count != len(expected_rois):
        failures.append(f"roi_count={roi_count} (expected {len(expected_rois)})")
    if error_code:
        # The P6-C product call: invalid operator geometry degrades to probe
        # default with error_code='geometry_invalid'. Surfacing that as FAIL
        # here is correct — the operator save isn't being applied.
        failures.append(f"error_code={error_code!r}")

    if failures:
        result.verdict = Verdict.FAIL
        for f in failures:
            result.add(f)
        return result

    result.verdict = Verdict.PASS
    result.add(
        f"source=operator_saved revision={revision} line_count={line_count} roi_count={roi_count}"
    )
    result.add(f"stream_session_id={row.get('stream_session_id')}")
    return result


def _filter_window(
    rows: list[dict[str, Any]],
    *,
    since: datetime | None,
    stream_session_id: str | None,
) -> list[dict[str, Any]]:
    """Client-side narrowing — the /observations/recent endpoint accepts only
    ``type`` and ``limit``, so we filter ``observed_at >= since`` and
    ``stream_session_id`` here. Returns rows in the order they came back
    (newest first per the endpoint contract)."""
    filtered: list[dict[str, Any]] = []
    for row in rows:
        if stream_session_id is not None and row.get("stream_session_id") != stream_session_id:
            continue
        if since is not None:
            observed_at = row.get("observed_at")
            if not isinstance(observed_at, str):
                continue
            try:
                ts = datetime.fromisoformat(observed_at.replace("Z", "+00:00"))
            except ValueError:
                continue
            if ts < since:
                continue
        filtered.append(row)
    return filtered


def check_observations(
    rows_by_type: dict[str, list[dict[str, Any]]],
    *,
    expected_revision: int,
    expected_lines: list[str] | None,
    expected_rois: list[str] | None,
    since: datetime | None,
    stream_session_id: str | None,
) -> CheckResult:
    """Verifies geometry lineage on the four observation types.

    Per-row checks:
    - ``payload.source.geometry_source == "operator_saved"``
    - ``payload.source.geometry_revision == expected_revision``
    - line_crossing rows: ``payload.line_id`` ∈ expected_lines (when supplied)
    - ROI/zone/dwell rows: ``payload.roi_id`` ∈ expected_rois (when supplied)

    Verdict:
    - FAIL  — any row shows wrong source / revision / unknown slug, OR any
              expected-supplied slug is absent across all rows of its type
              (lineage promise violated).
    - PARTIAL — line_crossing rows entirely absent in window (operator
                walked parallel; not a P6-C bug). status + ROI lineage still
                proves consumption.
    - PASS — every found row's lineage is correct AND every expected slug
             appears at least once.
    """
    result = CheckResult(name="Observations")
    expected_lines_set = set(expected_lines) if expected_lines else None
    expected_rois_set = set(expected_rois) if expected_rois else None

    seen_line_slugs: set[str] = set()
    seen_roi_slugs: set[str] = set()
    failures: list[str] = []
    counts: dict[str, int] = {}

    for obs_type, rows in rows_by_type.items():
        filtered = _filter_window(rows, since=since, stream_session_id=stream_session_id)
        counts[obs_type] = len(filtered)
        for row in filtered:
            payload = row.get("payload") or {}
            source_block = payload.get("source") or {}
            geo_source = source_block.get("geometry_source")
            geo_revision = source_block.get("geometry_revision")
            if geo_source != "operator_saved":
                failures.append(f"{obs_type} row {row.get('id')}: geometry_source={geo_source!r}")
                continue
            if geo_revision != expected_revision:
                failures.append(
                    f"{obs_type} row {row.get('id')}: geometry_revision={geo_revision} "
                    f"(expected {expected_revision})"
                )
                continue
            if obs_type == _OBS_TYPE_LINE:
                slug = payload.get("line_id")
                if slug:
                    seen_line_slugs.add(slug)
                if expected_lines_set is not None and slug not in expected_lines_set:
                    failures.append(
                        f"{obs_type} row {row.get('id')}: line_id={slug!r} "
                        f"not in expected {sorted(expected_lines_set)}"
                    )
            else:
                # The ROI slug lives under different keys depending on type:
                #   processor.roi_occupancy → payload.roi_id
                #   processor.zone_entered  → payload.zone_id
                #   processor.dwell_started → payload.zone_id
                # All three encode the same operator slug; fall back through
                # both so the verifier doesn't false-FAIL on zone/dwell rows.
                slug = payload.get("roi_id") or payload.get("zone_id")
                if slug:
                    seen_roi_slugs.add(slug)
                if expected_rois_set is not None and slug not in expected_rois_set:
                    failures.append(
                        f"{obs_type} row {row.get('id')}: roi/zone slug={slug!r} "
                        f"not in expected {sorted(expected_rois_set)}"
                    )

    # Expected-but-missing slugs across all rows of the right type = lineage
    # promise violated. ROI absence is FAIL because ROI rows DO fire from
    # standing in the zone; if none came back AND we expected slugs, the
    # operator's ROI saved geometry isn't reaching the runtime.
    missing_rois = (expected_rois_set or set()) - seen_roi_slugs
    if expected_rois_set is not None and missing_rois:
        if counts[_OBS_TYPE_ROI] + counts[_OBS_TYPE_ZONE] + counts[_OBS_TYPE_DWELL] == 0:
            failures.append(
                "no roi/zone/dwell rows in window — operator did not enter the ROI, "
                "or the runtime did not register entry"
            )
        else:
            failures.append(f"expected ROI slugs missing from observations: {sorted(missing_rois)}")

    if failures:
        result.verdict = Verdict.FAIL
        for f in failures:
            result.add(f)
        return result

    # Line absence is PARTIAL when ROI lineage already passed — the most
    # common explanation is a parallel walk that didn't cross perpendicular
    # to a vertical operator line.
    missing_lines = (expected_lines_set or set()) - seen_line_slugs
    if counts[_OBS_TYPE_LINE] == 0:
        result.verdict = Verdict.PARTIAL
        if expected_lines_set is None:
            result.add("no line_crossing rows in window (no --expected-lines supplied)")
        else:
            result.add(
                "no line_crossing rows in window — walk perpendicular across "
                f"each saved line and re-run (expected {sorted(expected_lines_set)})"
            )
    elif expected_lines_set is not None and missing_lines:
        # line_crossing rows exist but some expected slugs never showed —
        # PARTIAL, because the saved lines are in the runtime config but
        # the operator didn't cross them all.
        result.verdict = Verdict.PARTIAL
        result.add(f"expected line slugs not yet crossed: {sorted(missing_lines)}")
    else:
        result.verdict = Verdict.PASS

    for obs_type in _ALL_OBS_TYPES:
        result.add(f"{obs_type}: {counts[obs_type]} rows in window")
    if seen_line_slugs:
        result.add(f"line slugs observed: {sorted(seen_line_slugs)}")
    if seen_roi_slugs:
        result.add(f"roi slugs observed: {sorted(seen_roi_slugs)}")
    return result


def check_perception_snapshot(
    snapshot: dict[str, Any] | None,
    *,
    expected_lines: list[str] | None,
    expected_rois: list[str] | None,
) -> CheckResult:
    """Independent witness — the perception snapshot embeds the applied
    geometry, so the snapshot's line/ROI IDs must match expected slugs.

    Missing snapshot is a soft block on this witness only — it does not
    veto a PASS verdict if processor status + observations already proved
    lineage.
    """
    result = CheckResult(name="Perception snapshot")
    if snapshot is None:
        result.verdict = Verdict.BLOCKED
        result.add(
            "no perception snapshot for this camera yet — analytics mode may "
            "be off or the worker may not have posted one yet"
        )
        return result

    inner = snapshot.get("snapshot") or {}
    geometry = inner.get("geometry") or {}
    snapshot_line_ids = {line.get("id") for line in geometry.get("lines", []) if line.get("id")}
    snapshot_roi_ids = {roi.get("id") for roi in geometry.get("rois", []) if roi.get("id")}

    failures: list[str] = []
    if expected_lines is not None:
        missing = set(expected_lines) - snapshot_line_ids
        if missing:
            failures.append(f"snapshot.geometry.lines missing slugs: {sorted(missing)}")
    if expected_rois is not None:
        missing = set(expected_rois) - snapshot_roi_ids
        if missing:
            failures.append(f"snapshot.geometry.rois missing slugs: {sorted(missing)}")

    if failures:
        result.verdict = Verdict.FAIL
        for f in failures:
            result.add(f)
        return result

    result.verdict = Verdict.PASS
    result.add(f"snapshot.geometry.lines ids: {sorted(snapshot_line_ids)}")
    result.add(f"snapshot.geometry.rois ids: {sorted(snapshot_roi_ids)}")
    return result


def check_runtime_config(
    ini_text: str,
    *,
    expected_lines: list[str] | None,
    expected_rois: list[str] | None,
) -> CheckResult:
    """Optional witness — parses the nvdsanalytics INI from a temp dir on
    the processor host. Verifies expected slugs appear as ``roi-<id>=`` /
    ``line-crossing-<id>=`` keys and that the probe-default ROI is NOT
    present alongside operator slugs (which would indicate resolver leakage).
    """
    result = CheckResult(name="Runtime config")
    lines = ini_text.splitlines()
    roi_keys = {
        line.split("=", 1)[0]
        for line in lines
        if line.startswith("roi-") and "=" in line and not line.startswith("roi-filtering")
    }
    line_keys = {
        line.split("=", 1)[0]
        for line in lines
        if line.startswith("line-crossing-")
        and "=" in line
        and not line.startswith("line-crossing-stream")
    }
    roi_slugs_in_ini = {key[len("roi-") :] for key in roi_keys}
    line_slugs_in_ini = {key[len("line-crossing-") :] for key in line_keys}

    failures: list[str] = []
    if expected_lines is not None:
        missing = set(expected_lines) - line_slugs_in_ini
        if missing:
            failures.append(f"INI missing line-crossing slugs: {sorted(missing)}")
    if expected_rois is not None:
        missing = set(expected_rois) - roi_slugs_in_ini
        if missing:
            failures.append(f"INI missing roi slugs: {sorted(missing)}")
    if _PROBE_DEFAULT_ROI_LINE in ini_text and (expected_rois or expected_lines):
        failures.append(
            "probe-default `roi-entry-roi=64;36;...` present alongside operator geometry"
        )

    if failures:
        result.verdict = Verdict.FAIL
        for f in failures:
            result.add(f)
        return result

    result.verdict = Verdict.PASS
    result.add(f"INI roi slugs: {sorted(roi_slugs_in_ini)}")
    result.add(f"INI line slugs: {sorted(line_slugs_in_ini)}")
    return result


# ---------------------------------------------------------------------------
# Aggregation.
# ---------------------------------------------------------------------------


def aggregate_verdict(checks: list[CheckResult]) -> Verdict:
    """Combine per-witness verdicts into a single CLI verdict.

    Rules (in priority order):
    - any FAIL  → overall FAIL
    - any BLOCKED on processor-status → overall BLOCKED (we can't tell)
    - any PARTIAL → overall PARTIAL
    - otherwise PASS (BLOCKED on the perception-snapshot witness alone does
      not veto PASS — it's a soft witness)
    """
    if any(c.verdict == Verdict.FAIL for c in checks):
        return Verdict.FAIL
    if any(c.verdict == Verdict.BLOCKED and c.name == "Processor status" for c in checks):
        return Verdict.BLOCKED
    if any(c.verdict == Verdict.PARTIAL for c in checks):
        return Verdict.PARTIAL
    return Verdict.PASS


# ---------------------------------------------------------------------------
# Orchestration + CLI.
# ---------------------------------------------------------------------------


async def _run(
    *,
    api_base_url: str,
    api_token: str,
    camera_id: str,
    expected_revision: int,
    expected_lines: list[str] | None,
    expected_rois: list[str] | None,
    since: datetime | None,
    stream_session_id: str | None,
    analytics_config_path: Path | None,
    debug: bool,
) -> Verdict:
    headers = {"Authorization": f"Bearer {api_token}"}
    async with httpx.AsyncClient(base_url=api_base_url, headers=headers, timeout=15.0) as client:
        checks: list[CheckResult] = []

        # 1. Processor status.
        try:
            status_row = await fetch_processor_status(
                client, camera_id=camera_id, stream_session_id=stream_session_id
            )
        except FetchError as exc:
            blocked = CheckResult(name="Processor status", verdict=Verdict.BLOCKED)
            blocked.add(str(exc))
            checks.append(blocked)
            return _finish(checks, debug=debug, raw_payloads={"status_error": str(exc)})
        checks.append(
            check_processor_status(
                status_row,
                expected_revision=expected_revision,
                expected_lines=expected_lines,
                expected_rois=expected_rois,
            )
        )

        # 2. Observations (4 types).
        rows_by_type: dict[str, list[dict[str, Any]]] = {}
        try:
            for t in _ALL_OBS_TYPES:
                rows_by_type[t] = await fetch_observations(client, camera_id=camera_id, type_=t)
        except FetchError as exc:
            blocked = CheckResult(name="Observations", verdict=Verdict.BLOCKED)
            blocked.add(str(exc))
            checks.append(blocked)
            return _finish(
                checks,
                debug=debug,
                raw_payloads={"status": status_row, "observations_error": str(exc)},
            )
        checks.append(
            check_observations(
                rows_by_type,
                expected_revision=expected_revision,
                expected_lines=expected_lines,
                expected_rois=expected_rois,
                since=since,
                stream_session_id=stream_session_id,
            )
        )

        # 3. Perception snapshot (soft witness).
        try:
            snapshot = await fetch_perception_snapshot(client, camera_id=camera_id)
        except FetchError as exc:
            # An auth/server error on snapshot still counts as soft-blocked
            # for that witness — does NOT veto overall PASS by itself.
            snapshot_check = CheckResult(name="Perception snapshot", verdict=Verdict.BLOCKED)
            snapshot_check.add(str(exc))
            checks.append(snapshot_check)
            snapshot = None
        else:
            checks.append(
                check_perception_snapshot(
                    snapshot,
                    expected_lines=expected_lines,
                    expected_rois=expected_rois,
                )
            )

        # 4. Optional runtime-INI parse.
        if analytics_config_path is not None:
            try:
                # Off-load the small INI read so async hygiene linters
                # (ASYNC240) stay happy — the file is small and bounded,
                # but routing through a thread also keeps the script from
                # blocking the event loop if the disk hiccups.
                ini_text = await asyncio.to_thread(
                    analytics_config_path.read_text, encoding="utf-8"
                )
            except OSError as exc:
                ini_check = CheckResult(name="Runtime config", verdict=Verdict.BLOCKED)
                ini_check.add(f"could not read {analytics_config_path}: {exc}")
                checks.append(ini_check)
            else:
                checks.append(
                    check_runtime_config(
                        ini_text,
                        expected_lines=expected_lines,
                        expected_rois=expected_rois,
                    )
                )

        return _finish(
            checks,
            debug=debug,
            raw_payloads={
                "status": status_row,
                "observations": rows_by_type,
                "snapshot": snapshot,
            },
        )


def _finish(
    checks: list[CheckResult],
    *,
    debug: bool,
    raw_payloads: dict[str, Any],
) -> Verdict:
    verdict = aggregate_verdict(checks)
    _print_report(verdict, checks)
    if debug:
        # Dump JSON to stdout *after* the report. ``scrub_rtsp_in_text`` runs
        # over the whole serialized string so any embedded ``rtsp://…`` is
        # masked. Bearer tokens never appear in these payloads (they live in
        # request headers only).
        raw = json.dumps(raw_payloads, default=str, indent=2)
        sys.stdout.write("\n--- debug payloads (rtsp scrubbed) ---\n")
        sys.stdout.write(scrub_rtsp_in_text(raw))
        sys.stdout.write("\n")
    return verdict


_VERDICT_GLYPH = {
    Verdict.PASS: "PASS",
    Verdict.PARTIAL: "PARTIAL",
    Verdict.FAIL: "FAIL",
    Verdict.BLOCKED: "BLOCKED",
}


def _print_report(verdict: Verdict, checks: list[CheckResult]) -> None:
    sys.stdout.write(f"\n=== P6-D verification: {_VERDICT_GLYPH[verdict]} ===\n")
    for check in checks:
        sys.stdout.write(f"\n[{_VERDICT_GLYPH[check.verdict]}] {check.name}\n")
        for line in check.details:
            sys.stdout.write(f"  - {line}\n")
    sys.stdout.write("\n")


def _parse_csv(value: str | None) -> list[str] | None:
    if value is None:
        return None
    return [v.strip() for v in value.split(",") if v.strip()]


def _parse_since(value: str | None) -> datetime | None:
    if value is None:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="verify_geometry_lineage",
        description="P6-D end-to-end geometry lineage verification — exits "
        "0=PASS, 1=PARTIAL, 2=FAIL, 3=BLOCKED.",
    )
    p.add_argument("--camera-id", required=True, help="UUID of the camera to verify.")
    p.add_argument(
        "--expected-revision",
        required=True,
        type=int,
        help="The dashboard-saved geometry revision the processor must be applying.",
    )
    p.add_argument(
        "--expected-lines",
        default=None,
        help="Comma-separated saved line slugs (e.g. 'entry-line,line-1'). "
        "Optional but recommended.",
    )
    p.add_argument(
        "--expected-rois",
        default=None,
        help="Comma-separated saved ROI slugs (e.g. 'roi-1'). Optional but recommended.",
    )
    p.add_argument(
        "--since",
        default=None,
        help="ISO-8601 timestamp lower bound for observations (e.g. '2026-05-27T15:30:00Z'). "
        "Defaults to no lower bound (use --stream-session-id to scope tightly).",
    )
    p.add_argument(
        "--stream-session-id",
        default=None,
        help="UUID of the specific stream session to verify. Narrows both the "
        "processor-status query and observation filtering. Recommended when "
        "multiple processor cycles share the same camera in the time window.",
    )
    p.add_argument(
        "--api-base-url",
        default=os.environ.get("UPSTREAM_API_BASE_URL", "https://api.upstreamcv.com"),
        help="api-server base URL. Defaults to UPSTREAM_API_BASE_URL or production.",
    )
    p.add_argument(
        "--analytics-config-path",
        default=None,
        type=Path,
        help="Optional path to the live nvdsanalytics INI on the processor host "
        "(e.g. /tmp/upstream-p35-analytics.*/analytics_config.txt). "
        "Adds a runtime-config witness when supplied.",
    )
    p.add_argument(
        "--debug",
        action="store_true",
        help="Dump JSON payloads (rtsp:// substrings scrubbed). Never prints "
        "the bearer token; tokens live in request headers only.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    args = _build_parser().parse_args(argv)

    token = os.environ.get("UPSTREAM_API_TOKEN", "").strip()
    if not token:
        sys.stdout.write(
            "\n=== P6-D verification: BLOCKED ===\n"
            "\n[BLOCKED] Environment\n"
            "  - UPSTREAM_API_TOKEN is not set in the environment. Export a "
            "Bearer token (caller-authed, not the processor webhook secret) "
            "and re-run.\n\n"
        )
        return int(Verdict.BLOCKED)

    verdict = asyncio.run(
        _run(
            api_base_url=args.api_base_url,
            api_token=token,
            camera_id=args.camera_id,
            expected_revision=args.expected_revision,
            expected_lines=_parse_csv(args.expected_lines),
            expected_rois=_parse_csv(args.expected_rois),
            since=_parse_since(args.since),
            stream_session_id=args.stream_session_id,
            analytics_config_path=args.analytics_config_path,
            debug=args.debug,
        )
    )
    return int(verdict)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
