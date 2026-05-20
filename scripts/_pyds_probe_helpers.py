"""Pure helpers for the P2 pyds metadata probe.

Split out from ``deepstream_pyds_metadata_probe.py`` so they import without
gi/pyds (testable on the Mac under pytest). The probe imports from this file;
the wrapper shell script mounts both files into the DeepStream 8.0 container.

No I/O, no GStreamer, no DeepStream — pure Python.
"""

from __future__ import annotations

import re
from typing import Any

_RTSP_PATH_RE = re.compile(r"^(rtsp://[^/]+/)(.+)$", re.IGNORECASE)
_RTSP_INLINE_RE = re.compile(r"rtsp://[^\s\"']+", re.IGNORECASE)

# pyds' UNTRACKED_OBJECT_ID (max u64). Matches the DeepStream SDK constant.
UNTRACKED_OBJECT_ID = 0xFFFFFFFFFFFFFFFF


def redact_rtsp(url: str) -> str:
    """``rtsp://host:port/sess_xxx`` -> ``rtsp://host:port/sess_<redacted>``.

    Keeps scheme/host/port; masks the stream-key path. Non-RTSP / unparseable
    input is fully masked."""
    m = _RTSP_PATH_RE.match(url.strip())
    return f"{m.group(1)}sess_<redacted>" if m else "<redacted-rtsp>"


def scrub_rtsp_inline(text: str) -> str:
    """Redact any ``rtsp://...`` substrings embedded in free text (e.g. a
    GStreamer error mentioning the source URI) before printing/logging."""
    return _RTSP_INLINE_RE.sub(lambda m: redact_rtsp(m.group(0)), text)


def bbox_normalized(
    left: float, top: float, width: float, height: float, frame_w: int, frame_h: int
) -> dict[str, float | None]:
    """Normalize a DeepStream pixel bbox to a 0..1 box. Returns ``None``
    components when the frame dimension is non-positive."""
    nx = round(left / frame_w, 4) if frame_w > 0 else None
    ny = round(top / frame_h, 4) if frame_h > 0 else None
    nw = round(width / frame_w, 4) if frame_w > 0 else None
    nh = round(height / frame_h, 4) if frame_h > 0 else None
    return {"x": nx, "y": ny, "w": nw, "h": nh}


def build_neutral_object_sample(
    *,
    class_id: int,
    label: str,
    confidence: float,
    left: float,
    top: float,
    width: float,
    height: float,
    frame_w: int,
    frame_h: int,
    tracker_id_raw: int,
) -> dict[str, Any]:
    """Shape one detection into the **neutral intermediate** form
    (P2 Part G). NOT a final UpStream observation — debug evidence only."""
    tracker_id = None if tracker_id_raw == UNTRACKED_OBJECT_ID else tracker_id_raw
    return {
        "class_id": int(class_id),
        "label": label,
        "confidence": round(float(confidence), 4),
        "bbox_pixels": {
            "left": round(float(left), 2),
            "top": round(float(top), 2),
            "width": round(float(width), 2),
            "height": round(float(height), 2),
        },
        "bbox_normalized": bbox_normalized(left, top, width, height, frame_w, frame_h),
        "tracker_id": tracker_id,
    }
