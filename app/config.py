"""Process configuration.

Mirrors the UpStream settings convention (api-server `config.py`):
``BaseSettings`` with ``env_prefix="UPSTREAM_"``, ``SecretStr`` for the
webhook secret, and a cached ``get_settings()``.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="UPSTREAM_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Brief-mandated env vars ------------------------------------------
    api_base_url: str = "https://api.upstreamcv.com"
    processor_webhook_secret: SecretStr
    processor_id: str = "proc_local_mock_001"
    poll_interval_seconds: float = 5.0
    mock_observation_interval_seconds: float = 5.0
    log_level: str = "INFO"
    eligible_session_limit: int = 50

    # --- Internal tuning knobs (env-overridable, sane defaults) -----------
    status_interval_seconds: float = 5.0
    heartbeat_interval_seconds: float = 10.0
    detection_interval_seconds: float = 5.0

    http_connect_timeout_seconds: float = 5.0
    http_read_timeout_seconds: float = 10.0
    http_write_timeout_seconds: float = 10.0
    http_pool_timeout_seconds: float = 5.0

    # DeepStream-safe shutdown budget (M8.2B cleanup). Docker-backed workers
    # need time for `docker stop` + subprocess cleanup + the final `stopped`
    # status post; the old mock-tuned 3s/8s were too tight in the live run.
    # These are upper bounds — the fast mock runtime exits well within them,
    # so the larger default is safe for mixed runtimes. Env-overridable.
    final_status_timeout_seconds: float = 15.0
    overall_shutdown_timeout_seconds: float = 25.0

    backoff_initial_seconds: float = 1.0
    backoff_factor: float = 2.0
    backoff_max_seconds: float = 30.0
    backoff_jitter: float = 0.2

    max_restarts: int = 5
    restart_window_seconds: float = 60.0

    # --- M8.2B DeepStream decode-probe runtime ---------------------------
    # Dev/test only: force every eligible session to resolve to this package
    # id (e.g. deepstream.decode_probe.v0) without mutating production camera
    # config. Unset => use the session's own package_id.
    force_package_id: str | None = None
    # DeepStream 8.0 by default: 7.1 decodes on Blackwell but FAILS inference
    # (TensorRT 10.3 → "Unsupported SM: 0xc00", M8.2C1-A). 8.0 (TensorRT 10.9
    # / CUDA 12.8) builds the engine and runs nvinfer on the RTX 5070 Ti.
    # 8.0 also decodes fine, so it is safe as the single default for both the
    # decode_probe and inference_probe packages. Env-overridable.
    deepstream_image: str = "nvcr.io/nvidia/deepstream:8.0-samples-multiarch"
    deepstream_rtsp_transport: str = "tcp"
    deepstream_probe_timeout_seconds: float = 30.0
    deepstream_process_stop_timeout_seconds: float = 5.0
    deepstream_docker_network: str = "host"

    # --- M8.2C1-B DeepStream inference-probe runtime ---------------------
    # Bounded nvinfer smoke under the supervisor (TrafficCamNet sample model).
    # Detection evidence is status/log-only (KITTI dump) — NO observations,
    # NO metadata extraction, NO pyds. All env-overridable.
    deepstream_inference_timeout_seconds: float = 60.0
    # Empty => auto temp dir under the system tempdir. If set, the per-run
    # work dir (deepstream-app/PGIE config + KITTI output) is created under
    # this base (useful for post-run inspection on a GPU host).
    deepstream_inference_output_dir: str = ""
    deepstream_inference_model: str = "trafficcamnet"
    deepstream_keep_temp_files: bool = False


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()  # secret + overrides come from env/.env
