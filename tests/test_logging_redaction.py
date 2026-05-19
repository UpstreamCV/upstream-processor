from __future__ import annotations

import io
import logging

from app.logging_setup import SecretRedactionFilter, setup_logging

_SECRET = "supersecret-value-123"


def _record(msg: str, *args: object) -> logging.LogRecord:
    return logging.LogRecord(
        name="t",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg=msg,
        args=args,
        exc_info=None,
    )


def test_filter_scrubs_secret_header_token_and_jwt() -> None:
    f = SecretRedactionFilter(_SECRET)
    rec = _record(
        "auth=%s X-Upstream-Webhook-Secret: %s token=abc123 jwt eyJhbGci.payload.sig",
        _SECRET,
        _SECRET,
    )
    assert f.filter(rec) is True
    out = rec.getMessage()
    assert _SECRET not in out
    assert "abc123" not in out
    assert "eyJhbGci.payload.sig" not in out
    assert "REDACTED" in out


def test_filter_handles_no_secret_configured() -> None:
    f = SecretRedactionFilter(None)
    rec = _record("plain message, no creds")
    assert f.filter(rec) is True
    assert rec.getMessage() == "plain message, no creds"


def test_stream_handler_with_filter_emits_redacted() -> None:
    buf = io.StringIO()
    handler = logging.StreamHandler(buf)
    handler.addFilter(SecretRedactionFilter(_SECRET))
    logger = logging.getLogger("redaction-test")
    logger.handlers.clear()
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    logger.info("connecting with secret %s now", _SECRET)
    assert _SECRET not in buf.getvalue()
    assert "REDACTED" in buf.getvalue()


def test_setup_logging_installs_redaction_filter_on_handler() -> None:
    setup_logging("INFO", _SECRET)
    root = logging.getLogger()
    assert any(isinstance(flt, SecretRedactionFilter) for h in root.handlers for flt in h.filters)
