"""Pure helpers for the P5-E IOU-vs-NvDCF A/B comparison probe.

No Docker / GPU / pyds / gi / network — pure Python, importable under pytest on
any host (split out for the same reason as ``_pyds_probe_helpers.py``). The live
A/B (docs/local-dev.md P5-E runbook) runs the REAL worker twice against the same
scene via ``UPSTREAM_FORCE_PACKAGE_ID`` (IOU package, then NvDCF package) and
captures each run's records as JSON; these helpers aggregate one run into a
summary and diff two runs, so the comparison logic is testable without a GPU.

A "record" is any JSON object captured from a run:

- a processor STATUS payload (carries ``tracker_algorithm`` / ``frames_seen`` /
  ``unique_tracker_ids_count`` / ``fps_processed``),
- an OBSERVATION payload (carries an event ``type`` like
  ``processor.zone_exited``),
- a perception SNAPSHOT (carries per-frame ``objects`` with track ids).

The metrics are honest proxies, not ground-truth MOT scores — see the
docstrings.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

# Per-track events break under track-ID fragmentation; the aggregate counters do
# not. The A/B exists to show NvDCF reaching the per-track events that IOU's
# fragmentation suppressed (zone_exited / line_crossing_object / dwell_exceeded).
PER_TRACK_EVENTS = (
    "processor.line_crossing_object",
    "processor.zone_entered",
    "processor.zone_exited",
    "processor.dwell_started",
    "processor.dwell_updated",
    "processor.dwell_exceeded",
)
AGGREGATE_EVENTS = (
    "processor.line_crossing",
    "processor.roi_occupancy",
)
COMPARISON_EVENTS = PER_TRACK_EVENTS + AGGREGATE_EVENTS

_RTSP_RE = re.compile(r"rtsp://[^\s\"']+", re.IGNORECASE)


def scrub_rtsp(text: str) -> str:
    """Redact any ``rtsp://...`` substrings before printing/logging."""
    return _RTSP_RE.sub("<redacted-rtsp>", text)


def classify_record(rec: dict[str, Any]) -> str:
    """``observation`` / ``status`` / ``snapshot`` / ``unknown`` for one record."""
    type_ = rec.get("type")
    if isinstance(type_, str) and type_.startswith("processor."):
        return "observation"
    if isinstance(rec.get("objects"), list):
        return "snapshot"
    if any(k in rec for k in ("tracker_algorithm", "frames_seen", "unique_tracker_ids_count")):
        return "status"
    return "unknown"


def _extract_json_object(line: str) -> dict[str, Any] | None:
    start = line.find("{")
    if start < 0:
        return None
    try:
        obj = json.loads(line[start:])
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def parse_capture(text: str) -> list[dict[str, Any]]:
    """Parse a capture file into records. Accepts a JSON array, a single JSON
    object (possibly pretty-printed), or JSON-lines / log lines that each embed
    one JSON object (the recommended capture format)."""
    text = text.strip()
    if not text:
        return []
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        data = None
    if isinstance(data, list):
        return [r for r in data if isinstance(r, dict)]
    if isinstance(data, dict):
        return [data]
    records: list[dict[str, Any]] = []
    for line in text.splitlines():
        obj = _extract_json_object(line)
        if obj is not None:
            records.append(obj)
    return records


def _object_track_ids(objects: list[Any]) -> list[str]:
    ids: list[str] = []
    for obj in objects:
        if not isinstance(obj, dict):
            continue
        tid = obj.get("tracker_id", obj.get("track_id"))
        if tid is not None:
            ids.append(str(tid))
    return ids


def _frame_num(snapshot: dict[str, Any]) -> int:
    frame = snapshot.get("frame")
    value = frame.get("frame_num", 0) if isinstance(frame, dict) else snapshot.get("frame_num", 0)
    return int(value) if isinstance(value, (int, float)) else 0


def track_id_stats_from_snapshots(snapshots: list[dict[str, Any]]) -> dict[str, int]:
    """Per-track appearance stats from ordered perception snapshots.

    ``longest_persistence`` = max run of CONSECUTIVE snapshots a single track id
    appears in (a continuity proxy — higher is better). ``switch_proxy`` =
    ``unique_ids - peak_concurrent`` (extra ids beyond the most objects ever seen
    at once — a fragmentation proxy; lower is better). Both are proxies, not
    ground-truth MOT metrics."""
    ordered = sorted(snapshots, key=_frame_num)
    per_snapshot_ids = [set(_object_track_ids(s.get("objects", []))) for s in ordered]
    unique: set[str] = set().union(*per_snapshot_ids) if per_snapshot_ids else set()
    peak_concurrent = max((len(ids) for ids in per_snapshot_ids), default=0)
    longest = 0
    for tid in unique:
        run = best = 0
        for ids in per_snapshot_ids:
            if tid in ids:
                run += 1
                best = max(best, run)
            else:
                run = 0
        longest = max(longest, best)
    return {
        "unique_track_ids": len(unique),
        "longest_persistence": longest,
        "switch_proxy": max(0, len(unique) - peak_concurrent),
        "peak_concurrent": peak_concurrent,
    }


def _max_int(records: list[dict[str, Any]], key: str) -> int:
    vals = [int(r[key]) for r in records if isinstance(r.get(key), (int, float))]
    return max(vals) if vals else 0


def summarize(records: list[dict[str, Any]], label: str) -> dict[str, Any]:
    """Aggregate one run's records into a comparison summary."""
    observations = [r for r in records if classify_record(r) == "observation"]
    statuses = [r for r in records if classify_record(r) == "status"]
    snapshots = [r for r in records if classify_record(r) == "snapshot"]
    counts = Counter(r["type"] for r in observations)
    track = track_id_stats_from_snapshots(snapshots)
    unique = max(track["unique_track_ids"], _max_int(statuses, "unique_tracker_ids_count"))
    fps_vals = [
        float(s["fps_processed"])
        for s in statuses
        if isinstance(s.get("fps_processed"), (int, float))
    ]
    return {
        "label": label,
        "records": len(records),
        "status_samples": len(statuses),
        "snapshot_frames": len(snapshots),
        "total_observations": len(observations),
        "frames_seen": _max_int(statuses, "frames_seen"),
        "fps_processed": round(max(fps_vals), 2) if fps_vals else None,
        "unique_tracker_ids": unique,
        "longest_track_persistence_frames": track["longest_persistence"],
        "track_id_switch_proxy": track["switch_proxy"],
        "event_counts": {t: counts.get(t, 0) for t in COMPARISON_EVENTS},
        "other_event_counts": {
            t: c for t, c in sorted(counts.items()) if t not in COMPARISON_EVENTS
        },
    }


_COMPARISON_METRICS = (
    "frames_seen",
    "fps_processed",
    "unique_tracker_ids",
    "longest_track_persistence_frames",
    "track_id_switch_proxy",
    "total_observations",
)


def compare(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    """Side-by-side diff of two run summaries (a = baseline/IOU, b = NvDCF)."""
    out: dict[str, Any] = {"runs": [a["label"], b["label"]], "metrics": {}, "event_counts": {}}
    for metric in _COMPARISON_METRICS:
        out["metrics"][metric] = {a["label"]: a.get(metric), b["label"]: b.get(metric)}
    for event in COMPARISON_EVENTS:
        out["event_counts"][event] = {
            a["label"]: a["event_counts"].get(event, 0),
            b["label"]: b["event_counts"].get(event, 0),
        }
    out["notes"] = _verdict_notes(a, b)
    return out


def _verdict_notes(a: dict[str, Any], b: dict[str, Any]) -> list[str]:
    notes: list[str] = []
    if a["track_id_switch_proxy"] != b["track_id_switch_proxy"]:
        better = (
            a["label"] if a["track_id_switch_proxy"] < b["track_id_switch_proxy"] else b["label"]
        )
        notes.append(f"lower track_id_switch_proxy (less fragmentation): {better}")
    if a["longest_track_persistence_frames"] != b["longest_track_persistence_frames"]:
        better = (
            a["label"]
            if a["longest_track_persistence_frames"] > b["longest_track_persistence_frames"]
            else b["label"]
        )
        notes.append(f"longer track persistence: {better}")
    for event in (
        "processor.zone_exited",
        "processor.line_crossing",
        "processor.line_crossing_object",
        "processor.dwell_exceeded",
    ):
        ca, cb = a["event_counts"].get(event, 0), b["event_counts"].get(event, 0)
        if ca != cb:
            more = a["label"] if ca > cb else b["label"]
            notes.append(f"more {event}: {more} ({ca} vs {cb})")
    if not notes:
        notes.append("no decisive difference in the captured metrics")
    return notes


# --- DeepStream kitti-track-output → snapshots (clip A/B) ---------------------
#
# The clip A/B (scripts/probe_tracker_ab_clip.sh) runs ``deepstream-app`` against
# a recorded file with ``[tracker]`` enabled and ``kitti-track-output-dir`` set,
# producing one KITTI file per frame. Converting those into snapshot records
# lets the same summarize()/compare() path score tracker fragmentation (no
# observation/analytics path, so only the tracker metrics are populated).

_INT_TOKEN_RE = re.compile(r"-?\d+$")
_DIGITS_RE = re.compile(r"\d+")


def parse_kitti_track_line(line: str) -> tuple[str, str] | None:
    """Extract ``(label, track_id)`` from one DeepStream kitti-track-output line.

    DeepStream's KITTI tracking format is
    ``<label> <track_id> <truncated> <occluded> <alpha> <x1> <y1> <x2> <y2>
    ... <confidence>`` — the tracked object id is the **second token** (the
    later ``occluded`` column is also an integer, so a "last integer" heuristic
    is wrong). Returns ``None`` if the second token is not an integer id
    (e.g. a header/blank line)."""
    tokens = line.split()
    if len(tokens) < 2 or not _INT_TOKEN_RE.fullmatch(tokens[1]):
        return None
    return tokens[0], tokens[1]


def kitti_dir_to_snapshots(directory: str) -> list[dict[str, Any]]:
    """Convert a ``deepstream-app`` kitti-track-output directory into ordered
    snapshot records (one per frame file) consumable by :func:`summarize`."""
    try:
        names = sorted(p.name for p in Path(directory).iterdir() if p.suffix == ".txt")
    except OSError:
        return []
    snapshots: list[dict[str, Any]] = []
    for idx, name in enumerate(names):
        try:
            text = Path(directory, name).read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        objects: list[dict[str, str]] = []
        for line in text.splitlines():
            parsed = parse_kitti_track_line(line)
            if parsed is not None:
                label, track_id = parsed
                objects.append({"label": label, "track_id": track_id})
        digits = _DIGITS_RE.findall(name)
        frame_num = int(digits[-1]) if digits else idx
        snapshots.append({"frame_num": frame_num, "objects": objects})
    return snapshots
