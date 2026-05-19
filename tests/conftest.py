from __future__ import annotations

import os
from collections.abc import Iterator
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

import pytest


def _set_env() -> None:
    defaults = {
        "UPSTREAM_PROCESSOR_WEBHOOK_SECRET": "test-secret-value-xyz",
        "UPSTREAM_API_BASE_URL": "https://api.test",
        "UPSTREAM_PROCESSOR_ID": "proc_test_001",
        "UPSTREAM_LOG_LEVEL": "INFO",
    }
    for k, v in defaults.items():
        os.environ.setdefault(k, v)


_set_env()


@pytest.fixture(autouse=True)
def _clear_settings_cache() -> Iterator[None]:
    from app.config import get_settings

    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def settings():  # type: ignore[no-untyped-def]
    from app.config import get_settings

    return get_settings()


@pytest.fixture
def fast_settings():  # type: ignore[no-untyped-def]
    """Settings with sub-millisecond intervals so lifecycle tests are fast."""
    from app.config import get_settings

    return get_settings().model_copy(
        update={
            "poll_interval_seconds": 0.01,
            "mock_observation_interval_seconds": 0.01,
            "status_interval_seconds": 0.01,
            "heartbeat_interval_seconds": 0.01,
            "detection_interval_seconds": 0.01,
            "backoff_initial_seconds": 0.01,
            "backoff_max_seconds": 0.05,
            "final_status_timeout_seconds": 0.5,
            "overall_shutdown_timeout_seconds": 1.0,
            "restart_window_seconds": 1.0,
        }
    )


def eligible_session_dict(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "session_id": str(uuid4()),
        "stream_key": "sess_0123456789abcdef",
        "camera_id": str(uuid4()),
        "account_id": str(uuid4()),
        "internal_rtsp_url": "rtsp://TRIPWIRE.invalid:8554/sess_0123456789abcdef",
        "package_id": "mock.detect.v0",
        "runtime": "mock",
        "approved": {
            "codec": "h264",
            "width": 1280,
            "height": 720,
            "fps": 30,
            "bitrate_kbps": 3000,
        },
        "validation": {
            "passed": True,
            "observed": {"codec": "h264"},
            "expected": {"codec": "h264"},
            "mismatch_fields": [],
            "created_at": datetime.now(tz=UTC).isoformat(),
        },
        "zones_lines": None,
        "started_at": datetime.now(tz=UTC).isoformat(),
        "processor_eligible": True,
    }
    base.update(overrides)
    return base
