from __future__ import annotations

import os

from app.config import Settings, get_settings


def test_defaults_applied() -> None:
    s = get_settings()
    assert s.api_base_url == "https://api.test"  # from conftest env
    assert s.processor_id == "proc_test_001"
    assert s.poll_interval_seconds == 5.0
    assert s.eligible_session_limit == 50


def test_default_api_base_url_when_unset(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.delenv("UPSTREAM_API_BASE_URL", raising=False)
    get_settings.cache_clear()
    s = Settings()  # type: ignore[call-arg]
    assert s.api_base_url == "https://api.upstreamcv.com"


def test_secret_is_not_revealed_by_repr_or_str() -> None:
    s = get_settings()
    assert os.environ["UPSTREAM_PROCESSOR_WEBHOOK_SECRET"] not in repr(s)
    assert os.environ["UPSTREAM_PROCESSOR_WEBHOOK_SECRET"] not in str(s.processor_webhook_secret)
    # The real value is only obtainable via the explicit getter.
    assert (
        s.processor_webhook_secret.get_secret_value()
        == os.environ["UPSTREAM_PROCESSOR_WEBHOOK_SECRET"]
    )


def test_get_settings_is_cached() -> None:
    assert get_settings() is get_settings()
