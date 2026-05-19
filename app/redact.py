"""Redaction helpers for log/error safety.

The internal RTSP URL embeds the per-session stream key
(``rtsp://host:port/sess_<hex>``). The host:port is fine to log; the stream
key is session-identifying and must not be logged or committed. The
processor secret is never constructed here and is scrubbed separately by
``logging_setup.SecretRedactionFilter``.
"""

from __future__ import annotations

import re

_RTSP_PATH_RE = re.compile(r"^(rtsp://[^/]+/)(.+)$", re.IGNORECASE)
_RTSP_INLINE_RE = re.compile(r"rtsp://[^\s\"']+", re.IGNORECASE)


def redact_rtsp(url: str) -> str:
    """``rtsp://127.0.0.1:18554/sess_abc123`` -> ``rtsp://127.0.0.1:18554/sess_<redacted>``.

    Keeps scheme/host/port (useful for debugging reachability), masks the
    path. Non-RTSP / unparseable input is returned fully masked.
    """
    m = _RTSP_PATH_RE.match(url.strip())
    if not m:
        return "<redacted-rtsp>"
    return f"{m.group(1)}sess_<redacted>"


def scrub_rtsp_in_text(text: str) -> str:
    """Redact any ``rtsp://…`` substrings embedded in free text (e.g. a
    ``gst-launch -v`` line that echoes ``location="rtsp://…"``) before it is
    logged or posted to the API."""
    return _RTSP_INLINE_RE.sub(lambda m: redact_rtsp(m.group(0)), text)
