"""Logging with secret redaction.

Port of ``upstream-agent/upstream_agent/logging_setup.py:TokenRedactionFilter``.
The filter is attached to the handler (not loggers) and is seeded with the
literal processor secret so the secret cannot leak even via an exception
string or a third-party log line.
"""

from __future__ import annotations

import logging
import re
import sys

_REDACTED = "***REDACTED***"

# Generic patterns (applied regardless of the concrete secret value).
_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?i)(x-upstream-webhook-secret\s*[:=]\s*)\S+"),
    re.compile(r"(?i)(authorization\s*[:=]\s*)\S+"),
    re.compile(r"(?i)((?:secret|token|password)\s*[=:]\s*)\S+"),
    re.compile(r"eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+"),  # JWT
)


class SecretRedactionFilter(logging.Filter):
    """Scrub the literal secret and credential-shaped substrings from every
    record. Collapses the record to its fully-rendered (redacted) message so
    args cannot re-introduce the secret downstream."""

    def __init__(self, secret: str | None = None) -> None:
        super().__init__()
        self._secret = secret or None

    def _scrub(self, text: str) -> str:
        if self._secret and self._secret in text:
            text = text.replace(self._secret, _REDACTED)
        for pat in _PATTERNS:
            text = pat.sub(
                lambda m: (m.group(1) + _REDACTED) if m.groups() else _REDACTED,
                text,
            )
        return text

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            rendered = record.getMessage()
        except Exception:  # never let logging crash the app
            rendered = str(record.msg)
        record.msg = self._scrub(rendered)
        record.args = ()
        return True


def setup_logging(level: str, secret: str | None = None) -> None:
    handler = logging.StreamHandler(stream=sys.stderr)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    handler.addFilter(SecretRedactionFilter(secret))

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level.upper())
