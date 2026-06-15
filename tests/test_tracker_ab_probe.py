"""P5-E — IOU-vs-NvDCF A/B comparison probe: pure-core + script hygiene.

No Docker / GPU / pyds. Loads ``scripts/_tracker_ab_helpers.py`` by path (so
``scripts`` need not be a package, same as test_pyds_probe_helpers) and checks
the comparison aggregation, then lints the probe scripts for the established
hygiene rules (redaction, no tracing, fail-closed gate, no secret/RTSP leak).
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
_HELPERS_PATH = _SCRIPTS / "_tracker_ab_helpers.py"
_spec = importlib.util.spec_from_file_location("_tracker_ab_helpers", _HELPERS_PATH)
assert _spec is not None and _spec.loader is not None
_helpers = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_helpers)

classify_record = _helpers.classify_record
parse_capture = _helpers.parse_capture
summarize = _helpers.summarize
compare = _helpers.compare
scrub_rtsp = _helpers.scrub_rtsp
track_id_stats_from_snapshots = _helpers.track_id_stats_from_snapshots
parse_kitti_track_line = _helpers.parse_kitti_track_line
kitti_dir_to_snapshots = _helpers.kitti_dir_to_snapshots

_URL = "rtsp://127.0.0.1:18554/sess_abcdef0123456789"


# --- pure-core: classify / parse ----------------------------------------------


def test_classify_record() -> None:
    assert classify_record({"type": "processor.zone_exited"}) == "observation"
    assert classify_record({"objects": [], "frame": {"frame_num": 1}}) == "snapshot"
    assert classify_record({"tracker_algorithm": "nvdcf", "frames_seen": 10}) == "status"
    assert classify_record({"unique_tracker_ids_count": 3}) == "status"
    assert classify_record({"hello": "world"}) == "unknown"


def test_parse_capture_accepts_array_and_jsonlines() -> None:
    array = json.dumps([{"type": "processor.zone_entered"}, {"frames_seen": 5}])
    assert len(parse_capture(array)) == 2
    jsonlines = "\n".join(
        [
            '2026-06-15 INFO {"type": "processor.line_crossing"}',
            json.dumps({"frames_seen": 9, "unique_tracker_ids_count": 2}),
            "not json at all",
            "",
        ]
    )
    recs = parse_capture(jsonlines)
    assert len(recs) == 2
    assert recs[0]["type"] == "processor.line_crossing"
    assert parse_capture("") == []


# --- pure-core: track stats ---------------------------------------------------


def test_track_id_stats_persistence_and_switch_proxy() -> None:
    # Track "1" present in 3 consecutive snapshots; "2" only in the last; peak
    # concurrent = 2 (last snapshot) => switch_proxy = unique(2) - peak(2) = 0.
    snapshots = [
        {"frame": {"frame_num": 1}, "objects": [{"track_id": "1"}]},
        {"frame": {"frame_num": 2}, "objects": [{"track_id": "1"}]},
        {"frame": {"frame_num": 3}, "objects": [{"track_id": "1"}, {"track_id": "2"}]},
    ]
    stats = track_id_stats_from_snapshots(snapshots)
    assert stats["unique_track_ids"] == 2
    assert stats["longest_persistence"] == 3
    assert stats["peak_concurrent"] == 2
    assert stats["switch_proxy"] == 0


def test_track_id_switch_proxy_flags_fragmentation() -> None:
    # One person but a fresh id every frame (fragmentation): 3 unique ids, never
    # more than 1 concurrent => switch_proxy = 2.
    frag = [{"frame": {"frame_num": n}, "objects": [{"tracker_id": str(n)}]} for n in (1, 2, 3)]
    stats = track_id_stats_from_snapshots(frag)
    assert stats["unique_track_ids"] == 3
    assert stats["longest_persistence"] == 1
    assert stats["switch_proxy"] == 2


# --- pure-core: summarize + compare -------------------------------------------


def test_summarize_counts_events_and_tracker_metrics() -> None:
    records = parse_capture(
        "\n".join(
            json.dumps(r)
            for r in [
                {"type": "processor.zone_entered"},
                {"type": "processor.zone_exited"},
                {"type": "processor.line_crossing"},
                {"type": "processor.line_crossing"},
                {"frames_seen": 200, "unique_tracker_ids_count": 7, "fps_processed": 24.5},
            ]
        )
    )
    s = summarize(records, "iou")
    assert s["label"] == "iou"
    assert s["frames_seen"] == 200
    assert s["unique_tracker_ids"] == 7
    assert s["fps_processed"] == 24.5
    assert s["event_counts"]["processor.line_crossing"] == 2
    assert s["event_counts"]["processor.zone_exited"] == 1
    assert s["total_observations"] == 4


def test_compare_flags_nvdcf_improvements() -> None:
    iou = summarize(
        parse_capture(
            "\n".join(
                json.dumps(r)
                for r in [
                    {"frame": {"frame_num": 1}, "objects": [{"track_id": "1"}]},
                    {"frame": {"frame_num": 2}, "objects": [{"track_id": "2"}]},  # fragmented
                    {"frame": {"frame_num": 3}, "objects": [{"track_id": "3"}]},
                    {"frames_seen": 90, "unique_tracker_ids_count": 3},
                ]
            )
        ),
        "iou",
    )
    nvdcf = summarize(
        parse_capture(
            "\n".join(
                json.dumps(r)
                for r in [
                    {"frame": {"frame_num": 1}, "objects": [{"track_id": "9"}]},
                    {"frame": {"frame_num": 2}, "objects": [{"track_id": "9"}]},  # continuous
                    {"frame": {"frame_num": 3}, "objects": [{"track_id": "9"}]},
                    {"type": "processor.zone_exited"},
                    {"frames_seen": 90, "unique_tracker_ids_count": 1},
                ]
            )
        ),
        "nvdcf",
    )
    result = compare(iou, nvdcf)
    assert result["runs"] == ["iou", "nvdcf"]
    assert result["metrics"]["unique_tracker_ids"] == {"iou": 3, "nvdcf": 1}
    assert result["event_counts"]["processor.zone_exited"] == {"iou": 0, "nvdcf": 1}
    notes = " ".join(result["notes"])
    assert "less fragmentation): nvdcf" in notes
    assert "more processor.zone_exited: nvdcf" in notes


def test_scrub_rtsp_redacts() -> None:
    assert _URL not in scrub_rtsp(f"connecting to {_URL} now")


# --- pure-core: kitti track-output parsing (clip A/B) -------------------------


def test_parse_kitti_track_line_takes_second_token_as_track_id() -> None:
    # Real DeepStream format: "<label> <track_id> 0.0 0 0.0 <bbox> ... <conf>".
    # The track id is column 2; the later occluded "0" must NOT be picked.
    line = "person 12 0.0 0 0.0 458.4 478.0 474.5 531.1 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.500000"
    assert parse_kitti_track_line(line) == ("person", "12")
    assert parse_kitti_track_line("") is None
    assert parse_kitti_track_line("person") is None  # too few tokens
    assert parse_kitti_track_line("person notanint 0.0") is None  # non-int id column


def test_kitti_dir_to_snapshots(tmp_path) -> None:  # type: ignore[no-untyped-def]
    # Frame 1: track 7. Frame 2: tracks 7 + 8. Files sorted by name = frame order.
    (tmp_path / "out_000001.txt").write_text(
        "car 7 0.0 0 0.0 10.0 10.0 20.0 20.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.50\n"
    )
    (tmp_path / "out_000002.txt").write_text(
        "car 7 0.0 0 0.0 10.0 10.0 20.0 20.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.50\n"
        "car 8 0.0 0 0.0 50.0 50.0 60.0 60.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.50\n"
    )
    snaps = kitti_dir_to_snapshots(str(tmp_path))
    assert [s["frame_num"] for s in snaps] == [1, 2]
    assert [{o["track_id"] for o in s["objects"]} for s in snaps] == [{"7"}, {"7", "8"}]
    # Feeds straight into the existing stats: track 7 persists both frames.
    stats = track_id_stats_from_snapshots(snaps)
    assert stats["unique_track_ids"] == 2
    assert stats["longest_persistence"] == 2
    assert kitti_dir_to_snapshots(str(tmp_path / "missing")) == []


# --- scripts hygiene ----------------------------------------------------------


def test_ab_scripts_syntax_ok() -> None:
    proc = subprocess.run(
        ["bash", "-n", str(_SCRIPTS / "probe_tracker_ab.sh")], capture_output=True, text=True
    )
    assert proc.returncode == 0, proc.stderr


def test_ab_driver_redacts_and_never_traces() -> None:
    body = (_SCRIPTS / "probe_tracker_ab.sh").read_text()
    assert "redact()" in body  # RTSP redaction helper defined...
    assert "redact < " in body  # ...applied to each captured run log...
    assert "| redact" in body  # ...and to the printed comparison
    assert "set -x" not in body  # tracing would leak the stream key via argv
    # The only rtsp:// in the script is the redact() sed pattern itself; the
    # script never embeds a concrete stream key or a secret/auth literal.
    assert "sess_abcdef" not in body
    for needle in ("test-secret-value-xyz", "Authorization"):
        assert needle not in body


def test_ab_driver_fails_closed_without_confirm() -> None:
    """Without CONFIRM_RUN the driver must refuse to launch the real worker and
    exit non-zero, without ever echoing a stream URL."""
    proc = subprocess.run(
        ["bash", str(_SCRIPTS / "probe_tracker_ab.sh")],
        capture_output=True,
        text=True,
        env={"PATH": "/usr/bin:/bin"},
    )
    assert proc.returncode == 3
    combined = proc.stdout + proc.stderr
    assert "CONFIRM_RUN" in combined
    assert "rtsp://" not in combined


def test_compare_cli_runs_and_reports(tmp_path: Path) -> None:
    cap_a = tmp_path / "iou.log"
    cap_b = tmp_path / "nvdcf.log"
    cap_a.write_text(json.dumps({"frames_seen": 100, "unique_tracker_ids_count": 5}) + "\n")
    cap_b.write_text(
        "\n".join(
            [
                json.dumps({"frames_seen": 100, "unique_tracker_ids_count": 2}),
                json.dumps({"type": "processor.zone_exited"}),
            ]
        )
    )
    proc = subprocess.run(
        [
            "python3",
            str(_SCRIPTS / "compare_tracker_ab.py"),
            "iou",
            str(cap_a),
            "nvdcf",
            str(cap_b),
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    report = json.loads(proc.stdout)
    assert report["comparison"]["metrics"]["unique_tracker_ids"] == {"iou": 5, "nvdcf": 2}
    # Bad usage exits 2.
    bad = subprocess.run(
        ["python3", str(_SCRIPTS / "compare_tracker_ab.py"), "only", "one"],
        capture_output=True,
        text=True,
    )
    assert bad.returncode == 2


def test_clip_ab_script_syntax_and_hygiene() -> None:
    proc = subprocess.run(
        ["bash", "-n", str(_SCRIPTS / "probe_tracker_ab_clip.sh")],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    body = (_SCRIPTS / "probe_tracker_ab_clip.sh").read_text()
    assert "redact()" in body
    assert "set -x" not in body
    assert "sess_abcdef" not in body
    for needle in ("test-secret-value-xyz", "Authorization"):
        assert needle not in body


def test_clip_ab_script_fails_closed_on_missing_image_or_files(tmp_path: Path) -> None:
    """With a bogus image / no rendered YAMLs the clip driver exits 2 cleanly
    (no docker pull, no GPU) and never leaks a stream URL."""
    proc = subprocess.run(
        ["bash", str(_SCRIPTS / "probe_tracker_ab_clip.sh")],
        capture_output=True,
        text=True,
        env={"PATH": "/usr/bin:/bin", "IMAGE": "no-such-image:does-not-exist"},
    )
    assert proc.returncode == 2
    assert "rtsp://" not in (proc.stdout + proc.stderr)
