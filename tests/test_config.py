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


def test_deepstream_safe_shutdown_defaults() -> None:
    # M8.2B cleanup: DeepStream-safe defaults (env unset in conftest).
    s = get_settings()
    assert s.final_status_timeout_seconds == 15.0
    assert s.overall_shutdown_timeout_seconds == 25.0
    assert s.deepstream_process_stop_timeout_seconds == 5.0


def test_shutdown_timeouts_are_env_overridable(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("UPSTREAM_FINAL_STATUS_TIMEOUT_SECONDS", "7")
    monkeypatch.setenv("UPSTREAM_OVERALL_SHUTDOWN_TIMEOUT_SECONDS", "11")
    monkeypatch.setenv("UPSTREAM_DEEPSTREAM_PROCESS_STOP_TIMEOUT_SECONDS", "2")
    get_settings.cache_clear()
    s = Settings()  # type: ignore[call-arg]
    assert s.final_status_timeout_seconds == 7.0
    assert s.overall_shutdown_timeout_seconds == 11.0
    assert s.deepstream_process_stop_timeout_seconds == 2.0


def test_deepstream_image_defaults_to_8_0() -> None:
    # M8.2C1-B: 8.0 is the single default (7.1 fails inference on Blackwell).
    s = get_settings()
    assert s.deepstream_image == "nvcr.io/nvidia/deepstream:8.0-samples-multiarch"


def test_deepstream_image_is_env_overridable(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv(
        "UPSTREAM_DEEPSTREAM_IMAGE", "nvcr.io/nvidia/deepstream:7.1-samples-multiarch"
    )
    get_settings.cache_clear()
    s = Settings()  # type: ignore[call-arg]
    assert s.deepstream_image == "nvcr.io/nvidia/deepstream:7.1-samples-multiarch"


def test_deepstream_inference_defaults() -> None:
    s = get_settings()
    assert s.deepstream_inference_timeout_seconds == 60.0
    assert s.deepstream_inference_output_dir == ""
    assert s.deepstream_inference_model == "trafficcamnet"
    assert s.deepstream_keep_temp_files is False


def test_deepstream_inference_env_overridable(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("UPSTREAM_DEEPSTREAM_INFERENCE_TIMEOUT_SECONDS", "90")
    monkeypatch.setenv("UPSTREAM_DEEPSTREAM_KEEP_TEMP_FILES", "true")
    get_settings.cache_clear()
    s = Settings()  # type: ignore[call-arg]
    assert s.deepstream_inference_timeout_seconds == 90.0
    assert s.deepstream_keep_temp_files is True


def test_deepstream_metadata_probe_defaults() -> None:
    # P2 metadata-probe runtime: pyds-derived image, 60s probe, 256-slot
    # drop_newest queue, 3 samples per frame.
    s = get_settings()
    assert s.deepstream_pyds_image == "upstream-deepstream:8.0-pyds-v1.2.2"
    assert s.deepstream_metadata_probe_timeout_seconds == 60.0
    assert s.deepstream_metadata_queue_maxsize == 256
    assert s.deepstream_metadata_queue_drop_policy == "drop_newest"
    assert s.deepstream_metadata_sample_limit == 3


def test_deepstream_metadata_probe_env_overridable(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("UPSTREAM_DEEPSTREAM_PYDS_IMAGE", "ghcr.io/example/ds:custom")
    monkeypatch.setenv("UPSTREAM_DEEPSTREAM_METADATA_PROBE_TIMEOUT_SECONDS", "90")
    monkeypatch.setenv("UPSTREAM_DEEPSTREAM_METADATA_QUEUE_MAXSIZE", "512")
    monkeypatch.setenv("UPSTREAM_DEEPSTREAM_METADATA_QUEUE_DROP_POLICY", "drop_oldest")
    monkeypatch.setenv("UPSTREAM_DEEPSTREAM_METADATA_SAMPLE_LIMIT", "8")
    get_settings.cache_clear()
    s = Settings()  # type: ignore[call-arg]
    assert s.deepstream_pyds_image == "ghcr.io/example/ds:custom"
    assert s.deepstream_metadata_probe_timeout_seconds == 90.0
    assert s.deepstream_metadata_queue_maxsize == 512
    assert s.deepstream_metadata_queue_drop_policy == "drop_oldest"
    assert s.deepstream_metadata_sample_limit == 8


def test_perception_snapshot_defaults() -> None:
    # P5-C: enabled by default, ~1/s, 50-object cap.
    s = get_settings()
    assert s.perception_snapshot_enabled is True
    assert s.perception_snapshot_interval_seconds == 1.0
    assert s.perception_snapshot_max_objects == 50


def test_perception_snapshot_env_overridable(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("UPSTREAM_PERCEPTION_SNAPSHOT_ENABLED", "false")
    monkeypatch.setenv("UPSTREAM_PERCEPTION_SNAPSHOT_INTERVAL_SECONDS", "2.5")
    monkeypatch.setenv("UPSTREAM_PERCEPTION_SNAPSHOT_MAX_OBJECTS", "10")
    get_settings.cache_clear()
    s = Settings()  # type: ignore[call-arg]
    assert s.perception_snapshot_enabled is False
    assert s.perception_snapshot_interval_seconds == 2.5
    assert s.perception_snapshot_max_objects == 10
