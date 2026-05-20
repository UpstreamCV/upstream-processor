"""Unit tests for the P2 pyds-probe pure helpers (no Docker/GPU/pyds).

The helpers live in ``scripts/_pyds_probe_helpers.py`` so the probe script
can import them inside the DeepStream 8.0 container; here we load the same
file by path so pytest does not require ``scripts`` to be a package.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

_HELPERS_PATH = Path(__file__).resolve().parents[1] / "scripts" / "_pyds_probe_helpers.py"
_spec = importlib.util.spec_from_file_location("_pyds_probe_helpers", _HELPERS_PATH)
assert _spec is not None and _spec.loader is not None
_helpers = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_helpers)

redact_rtsp = _helpers.redact_rtsp
scrub_rtsp_inline = _helpers.scrub_rtsp_inline
bbox_normalized = _helpers.bbox_normalized
build_neutral_object_sample = _helpers.build_neutral_object_sample
UNTRACKED_OBJECT_ID = _helpers.UNTRACKED_OBJECT_ID


_URL = "rtsp://127.0.0.1:18554/sess_abcdef0123456789"


def test_redact_rtsp_masks_stream_key() -> None:
    assert redact_rtsp(_URL) == "rtsp://127.0.0.1:18554/sess_<redacted>"
    assert "abcdef0123456789" not in redact_rtsp(_URL)


def test_redact_rtsp_unparseable_input_is_fully_masked() -> None:
    assert redact_rtsp("not a url") == "<redacted-rtsp>"
    assert redact_rtsp("") == "<redacted-rtsp>"


def test_scrub_rtsp_inline_strips_embedded_urls() -> None:
    line = f"GST_ERROR: connection failed for {_URL} (transport=tcp); second one too: {_URL}?x=1"
    out = scrub_rtsp_inline(line)
    assert "abcdef0123456789" not in out
    assert out.count("sess_<redacted>") == 2


def test_bbox_normalized_roundtrips() -> None:
    n = bbox_normalized(160.0, 144.0, 320.0, 180.0, 1280, 720)
    assert n == {"x": 0.125, "y": 0.2, "w": 0.25, "h": 0.25}


def test_bbox_normalized_handles_zero_frame_dim() -> None:
    # frame_w=0 collapses x/w to None; frame_h=720 keeps y/h normalized.
    n = bbox_normalized(10.0, 10.0, 20.0, 20.0, 0, 720)
    assert n == {
        "x": None,
        "y": round(10.0 / 720, 4),
        "w": None,
        "h": round(20.0 / 720, 4),
    }


def test_build_neutral_object_sample_shape_matches_part_g() -> None:
    s = build_neutral_object_sample(
        class_id=2,
        label="Person",
        confidence=0.91234,
        left=100.0,
        top=50.0,
        width=200.0,
        height=400.0,
        frame_w=1280,
        frame_h=720,
        tracker_id_raw=UNTRACKED_OBJECT_ID,
    )
    assert s == {
        "class_id": 2,
        "label": "Person",
        "confidence": 0.9123,
        "bbox_pixels": {"left": 100.0, "top": 50.0, "width": 200.0, "height": 400.0},
        "bbox_normalized": {
            "x": round(100 / 1280, 4),
            "y": round(50 / 720, 4),
            "w": round(200 / 1280, 4),
            "h": round(400 / 720, 4),
        },
        "tracker_id": None,  # UNTRACKED_OBJECT_ID maps to None
    }


def test_build_neutral_object_sample_keeps_real_tracker_id() -> None:
    s = build_neutral_object_sample(
        class_id=0,
        label="Car",
        confidence=0.42,
        left=1.0,
        top=2.0,
        width=3.0,
        height=4.0,
        frame_w=1280,
        frame_h=720,
        tracker_id_raw=12345,
    )
    assert s["tracker_id"] == 12345


def test_neutral_sample_is_json_serializable() -> None:
    import json

    s = build_neutral_object_sample(
        class_id=1,
        label="",
        confidence=0.0,
        left=0.0,
        top=0.0,
        width=0.0,
        height=0.0,
        frame_w=1280,
        frame_h=720,
        tracker_id_raw=UNTRACKED_OBJECT_ID,
    )
    assert json.loads(json.dumps(s)) == s


def test_untracked_object_id_constant() -> None:
    assert UNTRACKED_OBJECT_ID == 0xFFFFFFFFFFFFFFFF
