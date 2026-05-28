"""P6-D — unit tests for ``scripts/verify_geometry_lineage.py``.

Two layers:

- Pure verdict-logic tests: feed synthetic payloads into the ``check_*``
  helpers and ``aggregate_verdict`` and assert verdicts. No HTTP, no I/O.
- Respx-mocked end-to-end tests: assert the orchestrator calls the right
  URLs with the Bearer header, scrubs ``rtsp://...`` substrings on
  ``--debug``, and surfaces the right verdict / exit code under both happy
  and degraded api-server responses.
"""

from __future__ import annotations

import importlib.util
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx

_SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "verify_geometry_lineage.py"


def _import_script() -> Any:
    """Load the script module by path (it's not under ``app/``, so a normal
    ``import`` won't find it). Registers the module in ``sys.modules``
    because the script defines dataclasses; Python 3.14's dataclasses module
    looks up ``cls.__module__`` via ``sys.modules`` during type-annotation
    resolution and crashes on a missing entry.
    """
    spec = importlib.util.spec_from_file_location("p6d_verify", _SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["p6d_verify"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def verify():  # type: ignore[no-untyped-def]
    return _import_script()


# ---------------------------------------------------------------------------
# Synthetic payload builders.
# ---------------------------------------------------------------------------


def _status_row(
    *,
    source: str = "operator_saved",
    revision: int | None = 9,
    line_count: int = 2,
    roi_count: int = 1,
    error_code: str | None = None,
    extra_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "applied_geometry": {
            "source": source,
            "revision": revision,
            "line_count": line_count,
            "roi_count": roi_count,
        }
    }
    if extra_payload:
        payload.update(extra_payload)
    return {
        "id": "11111111-1111-1111-1111-111111111111",
        "camera_id": "cccccccc-cccc-cccc-cccc-cccccccccccc",
        "stream_session_id": "5bc5dabe-37df-4ff2-961f-8268890bae01",
        "processor_id": "proc-1",
        "runtime": "deepstream",
        "package_id": "deepstream.analytics_probe.v0",
        "status": "processing",
        "fps_input": 30.0,
        "fps_processed": 29.8,
        "last_frame_at": "2026-05-27T15:30:10Z",
        "last_observation_at": "2026-05-27T15:30:11Z",
        "error_code": error_code,
        "error_message": None,
        "payload": payload,
        "created_at": "2026-05-27T15:24:59Z",
        "updated_at": "2026-05-27T15:30:11Z",
    }


def _observation(
    *,
    type_: str,
    line_id: str | None = None,
    roi_id: str | None = None,
    zone_id: str | None = None,
    geometry_source: str = "operator_saved",
    geometry_revision: int = 9,
    observed_at: str = "2026-05-27T15:31:00Z",
    stream_session_id: str = "5bc5dabe-37df-4ff2-961f-8268890bae01",
    obs_id: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "source": {
            "runtime": "deepstream",
            "geometry_source": geometry_source,
            "geometry_revision": geometry_revision,
        }
    }
    if line_id is not None:
        payload["line_id"] = line_id
    if roi_id is not None:
        payload["roi_id"] = roi_id
    if zone_id is not None:
        payload["zone_id"] = zone_id
    return {
        "id": obs_id or "00000000-0000-0000-0000-000000000001",
        "camera_id": "cccccccc-cccc-cccc-cccc-cccccccccccc",
        "stream_session_id": stream_session_id,
        "processor_id": "proc-1",
        "package_id": "deepstream.analytics_probe.v0",
        "type": type_,
        "payload": payload,
        "observed_at": observed_at,
        "created_at": observed_at,
    }


def _snapshot(*, line_ids: list[str], roi_ids: list[str]) -> dict[str, Any]:
    return {
        "snapshot": {
            "geometry": {
                "lines": [{"id": lid, "points": []} for lid in line_ids],
                "rois": [{"id": rid, "points": []} for rid in roi_ids],
            },
            "objects": [],
        }
    }


# ===========================================================================
# check_processor_status
# ===========================================================================


def test_status_pass(verify) -> None:  # type: ignore[no-untyped-def]
    row = _status_row()
    r = verify.check_processor_status(
        row,
        expected_revision=9,
        expected_lines=["entry-line", "line-1"],
        expected_rois=["roi-1"],
    )
    assert r.verdict == verify.Verdict.PASS


def test_status_blocked_when_row_missing(verify) -> None:  # type: ignore[no-untyped-def]
    r = verify.check_processor_status(
        None, expected_revision=9, expected_lines=None, expected_rois=None
    )
    assert r.verdict == verify.Verdict.BLOCKED


def test_status_fail_on_source_default(verify) -> None:  # type: ignore[no-untyped-def]
    """``source = default`` means the processor never picked up operator
    geometry. P6-C failure mode → FAIL, not PARTIAL."""
    row = _status_row(source="default", revision=None)
    r = verify.check_processor_status(
        row, expected_revision=9, expected_lines=None, expected_rois=None
    )
    assert r.verdict == verify.Verdict.FAIL


def test_status_fail_on_revision_mismatch(verify) -> None:  # type: ignore[no-untyped-def]
    row = _status_row(revision=8)
    r = verify.check_processor_status(
        row, expected_revision=9, expected_lines=None, expected_rois=None
    )
    assert r.verdict == verify.Verdict.FAIL


def test_status_fail_on_geometry_invalid_error_code(verify) -> None:  # type: ignore[no-untyped-def]
    """Even with ``source = operator_saved``, a ``geometry_invalid`` error
    code means the resolver fell back to probe defaults. P6-C wires it both
    ways — the lineage block AND the error_code — and verify must catch it."""
    row = _status_row(
        source="fallback_default_on_error",
        revision=9,
        error_code="geometry_invalid",
    )
    r = verify.check_processor_status(
        row, expected_revision=9, expected_lines=None, expected_rois=None
    )
    assert r.verdict == verify.Verdict.FAIL


def test_status_fail_on_missing_applied_geometry(verify) -> None:  # type: ignore[no-untyped-def]
    """A processor that predates P6-C never writes ``applied_geometry``;
    surfacing that as FAIL (not BLOCKED) is right — it means the deployed
    processor doesn't support P6-C, which is exactly what we're verifying."""
    row = _status_row()
    row["payload"] = {}  # drop applied_geometry
    r = verify.check_processor_status(
        row, expected_revision=9, expected_lines=None, expected_rois=None
    )
    assert r.verdict == verify.Verdict.FAIL


def test_status_fail_on_count_mismatch(verify) -> None:  # type: ignore[no-untyped-def]
    row = _status_row(line_count=1)  # operator saved 2 lines, processor sees 1
    r = verify.check_processor_status(
        row,
        expected_revision=9,
        expected_lines=["entry-line", "line-1"],
        expected_rois=["roi-1"],
    )
    assert r.verdict == verify.Verdict.FAIL


# ===========================================================================
# check_observations
# ===========================================================================


def _all_obs(verify, **per_type: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:  # type: ignore[no-untyped-def]
    """Build a {type: rows} map with empty defaults for the four types."""
    return {t: per_type.get(t.split(".")[-1], []) for t in verify._ALL_OBS_TYPES}


def test_observations_pass(verify) -> None:  # type: ignore[no-untyped-def]
    """Mirrors the real wire shapes the analytics worker emits:
    ``processor.roi_occupancy`` uses ``payload.roi_id`` while
    ``processor.zone_entered`` and ``processor.dwell_started`` use
    ``payload.zone_id`` (both encode the same operator slug)."""
    rows_by_type = _all_obs(
        verify,
        line_crossing=[
            _observation(type_=verify._OBS_TYPE_LINE, line_id="entry-line"),
            _observation(type_=verify._OBS_TYPE_LINE, line_id="line-1"),
        ],
        roi_occupancy=[_observation(type_=verify._OBS_TYPE_ROI, roi_id="roi-1")],
        zone_entered=[_observation(type_=verify._OBS_TYPE_ZONE, zone_id="roi-1")],
        dwell_started=[_observation(type_=verify._OBS_TYPE_DWELL, zone_id="roi-1")],
    )
    r = verify.check_observations(
        rows_by_type,
        expected_revision=9,
        expected_lines=["entry-line", "line-1"],
        expected_rois=["roi-1"],
        since=None,
        stream_session_id=None,
    )
    assert r.verdict == verify.Verdict.PASS


def test_observations_handles_per_type_slug_key(verify) -> None:  # type: ignore[no-untyped-def]
    """Pin the slug-key contract per observation type. P6-D-live found the
    verifier originally only checked ``payload.roi_id`` and so false-FAILed
    on zone/dwell rows (which use ``payload.zone_id``). This regression test
    ensures both keys are accepted for the ROI/zone/dwell types."""
    # All three ROI-family types reference the same operator slug, just via
    # a different payload key. Build one row per type using the type's real
    # key shape and confirm the verifier accepts all of them.
    rows_by_type = _all_obs(
        verify,
        roi_occupancy=[_observation(type_=verify._OBS_TYPE_ROI, roi_id="roi-1")],
        zone_entered=[_observation(type_=verify._OBS_TYPE_ZONE, zone_id="roi-1")],
        dwell_started=[_observation(type_=verify._OBS_TYPE_DWELL, zone_id="roi-1")],
    )
    r = verify.check_observations(
        rows_by_type,
        expected_revision=9,
        expected_lines=None,
        expected_rois=["roi-1"],
        since=None,
        stream_session_id=None,
    )
    # Line absence → PARTIAL (no --expected-lines supplied); ROI-family
    # rows must all be accepted as referencing ``roi-1`` regardless of which
    # key the analytics worker chose for that type.
    assert r.verdict == verify.Verdict.PARTIAL
    assert any("roi slugs observed: ['roi-1']" in d for d in r.details)


def test_observations_fail_on_unknown_zone_slug(verify) -> None:  # type: ignore[no-untyped-def]
    """A zone_entered row whose ``payload.zone_id`` is not in the expected
    set is still a real lineage mismatch and must FAIL (not be hidden by
    the roi_id fallback)."""
    rows_by_type = _all_obs(
        verify,
        zone_entered=[_observation(type_=verify._OBS_TYPE_ZONE, zone_id="stale-slug")],
    )
    r = verify.check_observations(
        rows_by_type,
        expected_revision=9,
        expected_lines=None,
        expected_rois=["roi-1"],
        since=None,
        stream_session_id=None,
    )
    assert r.verdict == verify.Verdict.FAIL


def test_observations_partial_when_no_line_rows(verify) -> None:  # type: ignore[no-untyped-def]
    """Most common live failure: ROI lineage proves the geometry is applied,
    but the operator walked parallel to a vertical line so no line_crossing
    rows came back. P6-D treats that as PARTIAL — not a P6-C bug."""
    rows_by_type = _all_obs(
        verify,
        roi_occupancy=[_observation(type_=verify._OBS_TYPE_ROI, roi_id="roi-1")],
        zone_entered=[_observation(type_=verify._OBS_TYPE_ZONE, roi_id="roi-1")],
        dwell_started=[_observation(type_=verify._OBS_TYPE_DWELL, roi_id="roi-1")],
    )
    r = verify.check_observations(
        rows_by_type,
        expected_revision=9,
        expected_lines=["entry-line", "line-1"],
        expected_rois=["roi-1"],
        since=None,
        stream_session_id=None,
    )
    assert r.verdict == verify.Verdict.PARTIAL
    assert any("walk perpendicular" in d for d in r.details)


def test_observations_partial_when_only_some_lines_crossed(verify) -> None:  # type: ignore[no-untyped-def]
    """Operator crossed one of two saved lines — PARTIAL with a clear nudge
    to walk the missing slug."""
    rows_by_type = _all_obs(
        verify,
        line_crossing=[_observation(type_=verify._OBS_TYPE_LINE, line_id="entry-line")],
        roi_occupancy=[_observation(type_=verify._OBS_TYPE_ROI, roi_id="roi-1")],
        zone_entered=[_observation(type_=verify._OBS_TYPE_ZONE, roi_id="roi-1")],
    )
    r = verify.check_observations(
        rows_by_type,
        expected_revision=9,
        expected_lines=["entry-line", "line-1"],
        expected_rois=["roi-1"],
        since=None,
        stream_session_id=None,
    )
    assert r.verdict == verify.Verdict.PARTIAL


def test_observations_fail_on_geometry_source_default(verify) -> None:  # type: ignore[no-untyped-def]
    """A line_crossing row with ``geometry_source = default`` means the
    runtime is using probe defaults; lineage stamping is wrong."""
    rows_by_type = _all_obs(
        verify,
        line_crossing=[
            _observation(
                type_=verify._OBS_TYPE_LINE,
                line_id="entry-line",
                geometry_source="default",
                geometry_revision=None,  # type: ignore[arg-type]
            )
        ],
    )
    r = verify.check_observations(
        rows_by_type,
        expected_revision=9,
        expected_lines=["entry-line"],
        expected_rois=None,
        since=None,
        stream_session_id=None,
    )
    assert r.verdict == verify.Verdict.FAIL


def test_observations_fail_on_revision_mismatch(verify) -> None:  # type: ignore[no-untyped-def]
    rows_by_type = _all_obs(
        verify,
        line_crossing=[
            _observation(type_=verify._OBS_TYPE_LINE, line_id="entry-line", geometry_revision=8)
        ],
    )
    r = verify.check_observations(
        rows_by_type,
        expected_revision=9,
        expected_lines=["entry-line"],
        expected_rois=None,
        since=None,
        stream_session_id=None,
    )
    assert r.verdict == verify.Verdict.FAIL


def test_observations_fail_on_unknown_slug(verify) -> None:  # type: ignore[no-untyped-def]
    """A ``line_id`` not in the expected set means either the saved geometry
    drifted between save and verify, or the processor is emitting a different
    set of lines than the operator authored. Either way: FAIL."""
    rows_by_type = _all_obs(
        verify,
        line_crossing=[_observation(type_=verify._OBS_TYPE_LINE, line_id="stale-slug")],
        roi_occupancy=[_observation(type_=verify._OBS_TYPE_ROI, roi_id="roi-1")],
    )
    r = verify.check_observations(
        rows_by_type,
        expected_revision=9,
        expected_lines=["entry-line", "line-1"],
        expected_rois=["roi-1"],
        since=None,
        stream_session_id=None,
    )
    assert r.verdict == verify.Verdict.FAIL


def test_observations_filters_by_since(verify) -> None:  # type: ignore[no-untyped-def]
    """Rows older than --since must be ignored. If only stale rows exist,
    we should see PARTIAL (window-empty), not PASS."""
    old = "2026-05-26T00:00:00Z"
    rows_by_type = _all_obs(
        verify,
        roi_occupancy=[_observation(type_=verify._OBS_TYPE_ROI, roi_id="roi-1", observed_at=old)],
    )
    r = verify.check_observations(
        rows_by_type,
        expected_revision=9,
        expected_lines=None,
        expected_rois=["roi-1"],
        since=datetime(2026, 5, 27, 15, 0, tzinfo=UTC),
        stream_session_id=None,
    )
    assert r.verdict == verify.Verdict.FAIL  # roi expected, none in window
    assert any("no roi/zone/dwell rows in window" in d for d in r.details)


def test_observations_filters_by_stream_session_id(verify) -> None:  # type: ignore[no-untyped-def]
    """Rows from a different stream_session_id must be ignored — important
    when multiple processor restarts have stamped overlapping observations."""
    rows_by_type = _all_obs(
        verify,
        roi_occupancy=[
            _observation(
                type_=verify._OBS_TYPE_ROI, roi_id="roi-1", stream_session_id="other-session"
            )
        ],
    )
    r = verify.check_observations(
        rows_by_type,
        expected_revision=9,
        expected_lines=None,
        expected_rois=["roi-1"],
        since=None,
        stream_session_id="target-session",
    )
    assert r.verdict == verify.Verdict.FAIL  # target session has no rows


# ===========================================================================
# check_perception_snapshot
# ===========================================================================


def test_snapshot_pass(verify) -> None:  # type: ignore[no-untyped-def]
    snap = _snapshot(line_ids=["entry-line", "line-1"], roi_ids=["roi-1"])
    r = verify.check_perception_snapshot(
        snap, expected_lines=["entry-line", "line-1"], expected_rois=["roi-1"]
    )
    assert r.verdict == verify.Verdict.PASS


def test_snapshot_blocked_when_missing(verify) -> None:  # type: ignore[no-untyped-def]
    r = verify.check_perception_snapshot(None, expected_lines=None, expected_rois=None)
    assert r.verdict == verify.Verdict.BLOCKED


def test_snapshot_fail_on_missing_slug(verify) -> None:  # type: ignore[no-untyped-def]
    snap = _snapshot(line_ids=["entry-line"], roi_ids=["roi-1"])
    r = verify.check_perception_snapshot(
        snap, expected_lines=["entry-line", "line-1"], expected_rois=["roi-1"]
    )
    assert r.verdict == verify.Verdict.FAIL


# ===========================================================================
# check_runtime_config (analytics_config.txt parser)
# ===========================================================================


_LIVE_INI = """\
[property]
enable=1
config-width=1280
config-height=720
[roi-filtering-stream-0]
enable=1
roi-roi-1=384;0;872;0;865;719;393;719
inverse-roi=0
class-id=-1
[line-crossing-stream-0]
enable=1
line-crossing-entry-line=373;0;373;719;453;359;293;359
line-crossing-line-1=884;0;884;719;964;359;804;359
class-id=-1
extended=0
mode=loose
"""


def test_runtime_config_pass(verify) -> None:  # type: ignore[no-untyped-def]
    r = verify.check_runtime_config(
        _LIVE_INI,
        expected_lines=["entry-line", "line-1"],
        expected_rois=["roi-1"],
    )
    assert r.verdict == verify.Verdict.PASS


def test_runtime_config_fail_on_missing_slug(verify) -> None:  # type: ignore[no-untyped-def]
    r = verify.check_runtime_config(
        _LIVE_INI, expected_lines=["entry-line", "missing-line"], expected_rois=["roi-1"]
    )
    assert r.verdict == verify.Verdict.FAIL


def test_runtime_config_fail_when_probe_default_leaks(verify) -> None:  # type: ignore[no-untyped-def]
    """If the resolver leaked the probe-default ROI alongside operator
    slugs, the INI carries both. P6-C says fallback replaces the probe; if
    both appear simultaneously something is wrong."""
    leaked = _LIVE_INI + "\nroi-entry-roi=64;36;1216;36;1216;684;64;684\n"
    r = verify.check_runtime_config(
        leaked, expected_lines=["entry-line", "line-1"], expected_rois=["roi-1"]
    )
    assert r.verdict == verify.Verdict.FAIL


# ===========================================================================
# aggregate_verdict
# ===========================================================================


def test_aggregate_pass(verify) -> None:  # type: ignore[no-untyped-def]
    checks = [
        verify.CheckResult(name="A", verdict=verify.Verdict.PASS),
        verify.CheckResult(name="B", verdict=verify.Verdict.PASS),
    ]
    assert verify.aggregate_verdict(checks) == verify.Verdict.PASS


def test_aggregate_fail_dominates(verify) -> None:  # type: ignore[no-untyped-def]
    checks = [
        verify.CheckResult(name="A", verdict=verify.Verdict.PASS),
        verify.CheckResult(name="B", verdict=verify.Verdict.FAIL),
        verify.CheckResult(name="C", verdict=verify.Verdict.PARTIAL),
    ]
    assert verify.aggregate_verdict(checks) == verify.Verdict.FAIL


def test_aggregate_processor_status_blocked_dominates_partial(verify) -> None:  # type: ignore[no-untyped-def]
    """If we couldn't reach the processor-status witness, we can't credibly
    judge a PARTIAL on the others."""
    checks = [
        verify.CheckResult(name="Processor status", verdict=verify.Verdict.BLOCKED),
        verify.CheckResult(name="Observations", verdict=verify.Verdict.PARTIAL),
    ]
    assert verify.aggregate_verdict(checks) == verify.Verdict.BLOCKED


def test_aggregate_snapshot_blocked_does_not_veto_pass(verify) -> None:  # type: ignore[no-untyped-def]
    """Perception snapshot is a soft witness — missing one shouldn't drop
    an otherwise-clean run from PASS to BLOCKED."""
    checks = [
        verify.CheckResult(name="Processor status", verdict=verify.Verdict.PASS),
        verify.CheckResult(name="Observations", verdict=verify.Verdict.PASS),
        verify.CheckResult(name="Perception snapshot", verdict=verify.Verdict.BLOCKED),
    ]
    assert verify.aggregate_verdict(checks) == verify.Verdict.PASS


def test_aggregate_partial_when_no_fail_no_block(verify) -> None:  # type: ignore[no-untyped-def]
    checks = [
        verify.CheckResult(name="Processor status", verdict=verify.Verdict.PASS),
        verify.CheckResult(name="Observations", verdict=verify.Verdict.PARTIAL),
        verify.CheckResult(name="Perception snapshot", verdict=verify.Verdict.PASS),
    ]
    assert verify.aggregate_verdict(checks) == verify.Verdict.PARTIAL


# ===========================================================================
# End-to-end with respx-mocked HTTP — proves URL routing, Bearer header,
# and exit-code mapping match the spec.
# ===========================================================================


_BASE = "https://api.test"
_CAMERA = "7b8313e5-155c-498f-9cc5-7af613d06074"


def _mock_obs_routes(rows_by_type: dict[str, list[dict[str, Any]]]) -> None:
    for _obs_type, rows in rows_by_type.items():
        respx.get(f"{_BASE}/v1/cameras/{_CAMERA}/observations/recent").mock(
            return_value=httpx.Response(200, json={"observations": rows})
        )
        # Note: respx matches the most-recently-registered route last, so
        # individual per-type filtering is handled by the response side
        # (we serve the same body to any type). The script's logic then
        # treats each type-fetch independently.


@pytest.mark.asyncio
@respx.mock
async def test_end_to_end_pass(verify) -> None:  # type: ignore[no-untyped-def]
    """Full happy path: status returns valid lineage; all four observation
    types return rows; snapshot has expected slugs. PASS, exit 0."""
    respx.get(f"{_BASE}/v1/cameras/{_CAMERA}/processor-status/latest").mock(
        return_value=httpx.Response(200, json={"status": _status_row()})
    )

    def _serve_observations(request: httpx.Request) -> httpx.Response:
        obs_type = request.url.params.get("type")
        if obs_type == verify._OBS_TYPE_LINE:
            return httpx.Response(
                200,
                json={
                    "observations": [
                        _observation(type_=verify._OBS_TYPE_LINE, line_id="entry-line"),
                        _observation(type_=verify._OBS_TYPE_LINE, line_id="line-1"),
                    ]
                },
            )
        if obs_type == verify._OBS_TYPE_ROI:
            return httpx.Response(
                200,
                json={"observations": [_observation(type_=verify._OBS_TYPE_ROI, roi_id="roi-1")]},
            )
        if obs_type == verify._OBS_TYPE_ZONE:
            return httpx.Response(
                200,
                json={"observations": [_observation(type_=verify._OBS_TYPE_ZONE, roi_id="roi-1")]},
            )
        if obs_type == verify._OBS_TYPE_DWELL:
            return httpx.Response(
                200,
                json={"observations": [_observation(type_=verify._OBS_TYPE_DWELL, roi_id="roi-1")]},
            )
        return httpx.Response(200, json={"observations": []})

    respx.get(f"{_BASE}/v1/cameras/{_CAMERA}/observations/recent").mock(
        side_effect=_serve_observations
    )
    respx.get(f"{_BASE}/v1/cameras/{_CAMERA}/perception/latest").mock(
        return_value=httpx.Response(
            200,
            json={
                "snapshot": {
                    "camera_id": _CAMERA,
                    "snapshot": _snapshot(line_ids=["entry-line", "line-1"], roi_ids=["roi-1"])[
                        "snapshot"
                    ],
                }
            },
        )
    )

    verdict = await verify._run(
        api_base_url=_BASE,
        api_token="test-token",
        camera_id=_CAMERA,
        expected_revision=9,
        expected_lines=["entry-line", "line-1"],
        expected_rois=["roi-1"],
        since=None,
        stream_session_id=None,
        analytics_config_path=None,
        debug=False,
    )
    assert verdict == verify.Verdict.PASS

    # Bearer auth header was attached on every request.
    for call in respx.calls:
        assert call.request.headers.get("Authorization") == "Bearer test-token"


@pytest.mark.asyncio
@respx.mock
async def test_end_to_end_blocked_on_status_404(verify) -> None:  # type: ignore[no-untyped-def]
    """No processor status yet → BLOCKED, exit 3. The script must NOT then
    proceed to call observations / snapshot (we already know the verdict)."""
    respx.get(f"{_BASE}/v1/cameras/{_CAMERA}/processor-status/latest").mock(
        return_value=httpx.Response(404, json={"error": {"code": "processor_status_not_found"}})
    )
    # Wire observations + snapshot so they would respond if called; the test
    # passes only if BLOCKED is returned (no observation call required) —
    # see ``aggregate_verdict`` rules for the BLOCKED-on-status veto.
    respx.get(f"{_BASE}/v1/cameras/{_CAMERA}/observations/recent").mock(
        return_value=httpx.Response(200, json={"observations": []})
    )
    respx.get(f"{_BASE}/v1/cameras/{_CAMERA}/perception/latest").mock(
        return_value=httpx.Response(200, json={"snapshot": {"snapshot": {"geometry": {}}}})
    )

    verdict = await verify._run(
        api_base_url=_BASE,
        api_token="test-token",
        camera_id=_CAMERA,
        expected_revision=9,
        expected_lines=None,
        expected_rois=None,
        since=None,
        stream_session_id=None,
        analytics_config_path=None,
        debug=False,
    )
    assert verdict == verify.Verdict.BLOCKED


@pytest.mark.asyncio
@respx.mock
async def test_end_to_end_blocked_on_auth_failure(verify) -> None:  # type: ignore[no-untyped-def]
    """A 401 on processor-status fetch → BLOCKED, exit 3 (auth issue, not a
    lineage failure)."""
    respx.get(f"{_BASE}/v1/cameras/{_CAMERA}/processor-status/latest").mock(
        return_value=httpx.Response(401, json={"error": {"code": "missing_credentials"}})
    )

    verdict = await verify._run(
        api_base_url=_BASE,
        api_token="wrong-token",
        camera_id=_CAMERA,
        expected_revision=9,
        expected_lines=None,
        expected_rois=None,
        since=None,
        stream_session_id=None,
        analytics_config_path=None,
        debug=False,
    )
    assert verdict == verify.Verdict.BLOCKED


@pytest.mark.asyncio
@respx.mock
async def test_end_to_end_stream_session_id_threads_through(verify) -> None:  # type: ignore[no-untyped-def]
    """The ``--stream-session-id`` arg must be passed to the processor-status
    endpoint as a query param, not swallowed."""
    captured: dict[str, Any] = {}

    def _capture(request: httpx.Request) -> httpx.Response:
        captured["params"] = dict(request.url.params)
        return httpx.Response(200, json={"status": _status_row()})

    respx.get(f"{_BASE}/v1/cameras/{_CAMERA}/processor-status/latest").mock(side_effect=_capture)
    respx.get(f"{_BASE}/v1/cameras/{_CAMERA}/observations/recent").mock(
        return_value=httpx.Response(200, json={"observations": []})
    )
    empty_snapshot = _snapshot(line_ids=[], roi_ids=[])["snapshot"]
    respx.get(f"{_BASE}/v1/cameras/{_CAMERA}/perception/latest").mock(
        return_value=httpx.Response(200, json={"snapshot": {"snapshot": empty_snapshot}})
    )

    await verify._run(
        api_base_url=_BASE,
        api_token="test-token",
        camera_id=_CAMERA,
        expected_revision=9,
        expected_lines=None,
        expected_rois=None,
        since=None,
        stream_session_id="session-uuid-here",
        analytics_config_path=None,
        debug=False,
    )
    assert captured["params"]["stream_session_id"] == "session-uuid-here"


# ===========================================================================
# Redaction on --debug output
# ===========================================================================


@pytest.mark.asyncio
@respx.mock
async def test_debug_dump_scrubs_rtsp_urls(verify, capsys) -> None:  # type: ignore[no-untyped-def]
    """If an api-server response contains an RTSP URL (e.g. inside a
    perception snapshot's source block), ``--debug`` must mask the path."""
    respx.get(f"{_BASE}/v1/cameras/{_CAMERA}/processor-status/latest").mock(
        return_value=httpx.Response(200, json={"status": _status_row()})
    )
    respx.get(f"{_BASE}/v1/cameras/{_CAMERA}/observations/recent").mock(
        return_value=httpx.Response(200, json={"observations": []})
    )
    leaky_snapshot = {
        "snapshot": {
            "geometry": {"lines": [{"id": "entry-line"}], "rois": [{"id": "roi-1"}]},
            "debug": {"upstream_url": "rtsp://10.0.0.1:18554/sess_deadbeef0123abcd"},
        }
    }
    respx.get(f"{_BASE}/v1/cameras/{_CAMERA}/perception/latest").mock(
        return_value=httpx.Response(200, json={"snapshot": leaky_snapshot})
    )

    await verify._run(
        api_base_url=_BASE,
        api_token="test-token",
        camera_id=_CAMERA,
        expected_revision=9,
        expected_lines=["entry-line"],
        expected_rois=["roi-1"],
        since=None,
        stream_session_id=None,
        analytics_config_path=None,
        debug=True,
    )
    out = capsys.readouterr().out
    # The session path was redacted (no raw stream key in output).
    assert "sess_deadbeef0123abcd" not in out
    assert "sess_<redacted>" in out
    # Bearer token never appears in any output.
    assert "test-token" not in out


# ===========================================================================
# CLI shim — argparse + exit codes
# ===========================================================================


def test_main_blocked_without_token(verify, monkeypatch, capsys) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.delenv("UPSTREAM_API_TOKEN", raising=False)
    exit_code = verify.main(
        [
            "--camera-id",
            _CAMERA,
            "--expected-revision",
            "9",
        ]
    )
    assert exit_code == int(verify.Verdict.BLOCKED)
    out = capsys.readouterr().out
    assert "UPSTREAM_API_TOKEN" in out


def test_csv_parser_strips_whitespace(verify) -> None:  # type: ignore[no-untyped-def]
    assert verify._parse_csv("a, b ,c") == ["a", "b", "c"]
    assert verify._parse_csv("") == []
    assert verify._parse_csv(None) is None


def test_since_parser_handles_z_suffix(verify) -> None:  # type: ignore[no-untyped-def]
    ts = verify._parse_since("2026-05-27T15:30:00Z")
    assert ts is not None
    assert ts.tzinfo is not None
    assert ts.year == 2026 and ts.minute == 30


def test_verdict_exit_codes(verify) -> None:  # type: ignore[no-untyped-def]
    """Pin the exit-code contract that CI runners branch on."""
    assert int(verify.Verdict.PASS) == 0
    assert int(verify.Verdict.PARTIAL) == 1
    assert int(verify.Verdict.FAIL) == 2
    assert int(verify.Verdict.BLOCKED) == 3
