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

    # --- P2 DeepStream metadata-probe runtime ---------------------------
    # Bounded pyds metadata extraction under the supervisor. Uses a separate
    # DeepStream 8.0 + pyds image (built from docker/Dockerfile.pyds in the
    # manual P2 probe milestone). Status/debug evidence only — NO
    # observations, NO API writes, NO metadata adapter. Env-overridable.
    deepstream_pyds_image: str = "upstream-deepstream:8.0-pyds-v1.2.2"
    deepstream_metadata_probe_timeout_seconds: float = 60.0
    # Bounded bridge queue between the pyds container stdout reader and the
    # async consumer. ``drop_policy`` is the producer-side eviction policy
    # when the queue is full (``drop_newest`` | ``drop_oldest``).
    deepstream_metadata_queue_maxsize: int = 256
    deepstream_metadata_queue_drop_policy: str = "drop_newest"
    # How many object samples per frame the in-container app emits in its
    # UPSTREAM_METADATA= lines (status/debug evidence; the worker never
    # forwards these to the API).
    deepstream_metadata_sample_limit: int = 3

    # --- P3 DeepStream tracker-probe runtime ----------------------------
    # Tracker-probe mode (deepstream.tracker_probe.v0): same pyds metadata
    # path as metadata_probe but inserts nvtracker between nvinfer and
    # nvvideoconvert so NvDsObjectMeta.object_id becomes a real stable
    # tracker id across frames. Pad probe attaches to the tracker src pad.
    # Status payload gains tracker_enabled / unique_tracker_ids_count /
    # tracker_ids_sample. Still NO observations, NO API writes.
    #
    # Library is in-container path; stable across DS 8.0 patch versions.
    deepstream_tracker_lib_path: str = (
        "/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so"
    )
    # Low-level tracker algorithm. v0 supports the no-ReID IOU tracker which
    # proves stable ids without needing a ReID model. NvDCF/NvSORT/NvDeepSORT
    # are reserved for future tuning.
    deepstream_tracker_algorithm: str = "iou"
    # nvtracker tracker-width/height: must be multiples of 32 for NvDCF /
    # NvDeepSORT (harmless for IOU). 960x544 matches the DS 8.0 sample.
    deepstream_tracker_width: int = 960
    deepstream_tracker_height: int = 544
    # Bounded sample of distinct tracker ids surfaced in the status payload.
    # Keep small — status payload should stay tiny.
    deepstream_tracker_ids_sample_limit: int = 10

    # --- P3.5 DeepStream analytics-probe runtime ------------------------
    # Analytics-probe mode (deepstream.analytics_probe.v0): same pyds path
    # as tracker_probe but inserts an ``nvdsanalytics`` element after
    # ``nvtracker`` and walks ``NvDsUserMeta`` (frame + object) for the
    # ROI / line-crossing / overcrowding / direction metadata that the
    # SDK attaches. v0 ships a hand-authored ROI + line-crossing config
    # targeted at 1280x720; dashboard-driven geometry is a later
    # milestone. Status payload gains analytics_enabled,
    # has_analytics_metadata, roi_metadata_count, line_metadata_count,
    # analytics_labels_sample. Still NO observations, NO API writes.
    #
    # Bounded sample of distinct analytics labels (e.g. ROI / LC labels
    # observed) surfaced in the status payload. Keep small.
    deepstream_analytics_labels_sample_limit: int = 20

    # --- P4-C ROI occupancy emission policy --------------------------
    # ROI counts arrive at frame rate (~30 fps) but we never want
    # per-frame writes. The emit policy is on-first + on-change +
    # interval-elapsed. ``min_interval_seconds`` is the heartbeat
    # cadence when the count is unchanged (default 30s; lower for
    # tests via ``fast_settings``, raise in production if status pings
    # are noisy). Setting to 0 makes every unchanged frame emit — only
    # safe for fakes/tests.
    deepstream_roi_min_interval_seconds: float = 30.0

    # --- P4-C cleanup: ROI debounce / hysteresis ---------------------
    # Number of consecutive frames a new ROI count must hold before
    # the change branch emits. P4-C's live verification showed
    # TrafficCamNet's detection count flaps on edge-case frames
    # (count flips 0 ↔ 1 between adjacent frames even with an
    # operator stationary), and an undamped on-change policy turns
    # each flap into a real observation. Setting >= 2 suppresses
    # short flaps; 5 (production default) requires ~166 ms of
    # stability at 30 fps before a change is committed.
    #
    # The first-seen path and the interval heartbeat are unchanged.
    # During a pending change, the heartbeat reports the LAST stable
    # count, not the unstable candidate (so consumers cannot mistake
    # a heartbeat for a real state change).
    #
    # Setting to 1 reproduces the P4-C immediate-change behavior
    # (used by ``fast_settings`` so existing emit tests don't need
    # to override). Values < 1 are rejected at adapter construction.
    deepstream_roi_change_stability_frames: int = 5

    # --- P4-D zone transition emission policy ------------------------
    # Per-track zone transitions (processor.zone_entered /
    # processor.zone_exited). State is keyed per (track_id, zone_id)
    # and updated only on observed frames; total object disappearance
    # is treated as a detector/tracker miss, not a real exit (no
    # exit event is emitted from absence alone). A new membership
    # value (inside or outside) must hold for this many consecutive
    # OBSERVED frames before the transition commits and an event is
    # emitted. Default 5 matches the ROI-occupancy debounce default —
    # ~166 ms of stability at 30 fps. Setting to 1 reproduces
    # immediate-transition behavior. Values < 1 fail at adapter
    # construction.
    deepstream_zone_membership_stability_frames: int = 5

    # --- P4-D follow-up: analytics ROI geometry + class filter -------
    # P4-D live verification revealed that the operator's bbox bottom-
    # center (nvdsanalytics' reference point for ROI containment) was
    # at y=719 on a 720-tall frame while the previous ROI ended at
    # y=648 (90% of 720) — so the object was outside the ROI and
    # ``roi_labels`` was never attached. ``class-id=-1`` was already
    # in the config; the actual root cause was geometry.
    #
    # ``roi_margin_pct`` is the inset on each side of the frame for
    # the probe-default rectangular ROI. 0.05 → ROI covers the
    # central 90% x 90% of the frame. P4-D used the implicit 20%
    # horizontal + 10% vertical margin and a foot point at the bottom
    # edge fell outside; the new looser default is more forgiving for
    # camera-above-operator scenes. Range [0, 0.49]; values outside
    # fail at adapter construction. Production-tuned scenes are
    # expected to ship per-camera geometry via the dashboard milestone.
    deepstream_analytics_roi_margin_pct: float = 0.05

    # ``analytics_class_ids`` is the nvdsanalytics per-section class
    # filter. ``"all"`` (the probe default) and ``"-1"`` both map to
    # ``class-id=-1`` (every detected class counted). A single integer
    # like ``"2"`` maps to ``class-id=2`` (one class only). The
    # nvdsanalytics property parser accepts a SINGLE int per section
    # so multi-class lists are not supported in v0; multi-class
    # production tuning will need per-class sections, a later
    # milestone. Values are validated at adapter construction.
    deepstream_analytics_class_ids: str = "all"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()  # secret + overrides come from env/.env
