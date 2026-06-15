"""DeepStream worker — decode probe (M8.2B) + inference probe (M8.2C1-B).

Implements the existing ``Runtime`` interface so the supervisor treats it
exactly like the mock worker. The resolved package's ``supports`` selects
the mode:

- ``decode_probe`` (``deepstream.decode_probe.v0``, M8.2B) — the M8.2A-proven
  GStreamer NVDEC decode pipeline. **Decode only.**
- ``inference_probe`` (``deepstream.inference_probe.v0``, M8.2C1-B) — the
  M8.2C1-A-proven DeepStream 8.0 ``deepstream-app`` nvinfer pipeline
  (TrafficCamNet sample model). Inference runs; detection evidence is
  status/log-only (KITTI dump). **No observations, no metadata extraction.**

Both map onto the M8.0 status lifecycle:

    starting -> attached (source/pipeline up) -> processing (fps/inference)
             -> stopped (graceful) | failed (process / TensorRT-SM error)

Neither emits observations. The RTSP URL is never logged (only its redacted
form); the secret is never handled here. **No pyds, no pad probes, no
MetadataSource, no metadata adapter, no tracking, no nvdsanalytics.**
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import re
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Any
from uuid import UUID

from app.api_client.base import ProcessorApi
from app.backoff import Backoff, wait_or_event
from app.clock import now_utc
from app.config import Settings
from app.errors import (
    TRANSIENT_API_ERRORS,
    ApiSessionGoneError,
    ApiValidationError,
)
from app.geometry.resolver import ResolvedGeometry, resolve_session_geometry
from app.observations.bridge_queue import BoundedBridgeQueue, DropPolicy
from app.observations.metadata_models import NeutralFrameMetadata
from app.observations.observation_adapter import (
    DwellState,
    LineCrossingObjectState,
    LineCrossingState,
    RoiOccupancyState,
    ZoneMembershipState,
    deepstream_source_block,
)
from app.observations.schemas import EligibleSession
from app.observations.writer import ObservationWriter
from app.packages.registries import (
    find_model_by_alias,
    find_runnable_deepstream_tracker_by_profile_id,
    load_tracker_registry,
)
from app.packages.resolver import PackageSpec
from app.perception.builder import build_perception_snapshot
from app.perception.fanout_publisher import FanoutPublisher, build_fanout_envelope
from app.perception.schemas import PerceptionSnapshotPayload
from app.processing.class_filter import filter_objects_by_class
from app.processing.resolver import (
    DEFAULT_APPLIED,
    ResolvedProcessingConfig,
    resolve_session_processing_config,
)
from app.redact import redact_rtsp, scrub_rtsp_in_text
from app.runtimes.base import Runtime, WorkerState
from app.runtimes.deepstream_runtime.inference import (
    AnalyticsGeometry,
    DeepStreamInferenceProcessManager,
    compute_analytics_geometry,
)
from app.runtimes.deepstream_runtime.metadata_probe import (
    DeepStreamMetadataProcessManager,
)
from app.runtimes.deepstream_runtime.process_manager import (
    DeepStreamProcessManager,
    ProcessManager,
)
from app.runtimes.metadata_source import BridgeQueueMetadataSource
from app.supervisor.processor_status import build_status_request

_log = logging.getLogger("upstream_processor.deepstream.worker")

_TERMINAL_STATUS_ATTEMPTS = 3
_READLINE_POLL_SECONDS = 0.5
# Bound on draining the background perception sender at consumer teardown so
# the terminal status post reflects the final perception counters. Best-effort:
# a stuck upload is abandoned after this so teardown can never wedge.
_PERCEPTION_FLUSH_TIMEOUT = 5.0
# gst -e on SIGINT, or `timeout` self-terminating the bounded probe, are
# graceful endings (124 = timeout, 130 = SIGINT, 137 = SIGKILL after stop).
_CLEAN_EXIT_CODES = frozenset({0, 124, 130, 137})

_DECODE_MODE = "decode_probe"
_INFERENCE_MODE = "inference_probe"
_METADATA_MODE = "metadata_probe"
_TRACKER_MODE = "tracker_probe"
_ANALYTICS_MODE = "analytics_probe"

# Structured stdout markers emitted by the in-container pyds metadata app
# (``app/runtimes/deepstream_runtime/pyds_metadata_app.py``). Keep in lock-
# step with the constants defined there.
_META_PREFIX = "UPSTREAM_METADATA="
_SUMMARY_PREFIX = "UPSTREAM_METADATA_SUMMARY="

_FPS_RE = re.compile(
    r"rendered:\s*(\d+),\s*dropped:\s*(\d+),.*?average:\s*([\d.]+)",
    re.IGNORECASE,
)


def _parse_fps(line: str) -> tuple[int, int, float] | None:
    m = _FPS_RE.search(line)
    if m is None:
        return None
    return int(m.group(1)), int(m.group(2)), float(m.group(3))


# --- inference-probe (M8.2C1-B) log markers, from the proven M8.2C1-A run ---
# deepstream-app PERF line: "**PERF:  30.00 (29.98)" → (instant, avg). The
# "**PERF:  FPS 0 (Avg)" header has no decimal and is intentionally skipped.
_PERF_RE = re.compile(r"\*\*PERF:\s+([0-9]+\.[0-9]+)\s+\(([0-9]+\.[0-9]+)\)")
# nvinfer built or loaded a usable TensorRT engine (success).
_ENGINE_OK_RE = re.compile(
    r"serialize cuda engine to file:.*successfully"
    r"|deserialized trt engine"
    r"|Use deserialized engine model"
    r"|Loaded engine",
    re.IGNORECASE,
)
# Source/pipeline/nvinfer is up — earliest attach marker (appears well before
# the engine finishes building).
_ATTACH_RE = re.compile(
    r"Trying to create engine from model files"
    r"|deserialize engine from file"
    r"|NvDsInferContext"
    r"|Pipeline running"
    r"|nvstreammux",
    re.IGNORECASE,
)
# Hard TensorRT / GPU-architecture failure (the DS 7.1 / Blackwell case).
_HARD_FAIL_RE = re.compile(
    r"Unsupported SM"
    r"|buildSerializedNetwork:.*Error"
    r"|failed to build (cuda|network) engine"
    r"|Segmentation fault"
    r"|core dump",
    re.IGNORECASE,
)


def _parse_perf(line: str) -> float | None:
    """Average fps from a deepstream-app ``**PERF:`` line, or ``None``."""
    m = _PERF_RE.search(line)
    if m is None:
        return None
    return float(m.group(2))


def parse_metadata_line(line: str) -> tuple[str, str] | None:
    """Pure parser for the metadata-probe stdout protocol.

    Returns ``("frame", "<json>")`` for ``UPSTREAM_METADATA=`` lines,
    ``("summary", "<json>")`` for ``UPSTREAM_METADATA_SUMMARY=`` lines,
    or ``None`` for any other line. JSON validation happens in the caller
    so an invalid payload doesn't kill the reader."""
    if line.startswith(_SUMMARY_PREFIX):
        return ("summary", line[len(_SUMMARY_PREFIX) :])
    if line.startswith(_META_PREFIX):
        return ("frame", line[len(_META_PREFIX) :])
    return None


@dataclass
class _MetadataCounters:
    """Worker-side aggregate counters fed by the metadata consumer task.

    ``unique_tracker_ids`` / ``tracker_ids_sample`` are populated only when
    the producer emits real ``tracker_id`` values (P3 tracker_probe mode);
    in metadata_probe they stay empty and the corresponding payload fields
    are suppressed by the worker."""

    frames_seen: int = 0
    frames_with_objects: int = 0
    objects_seen: int = 0
    class_counts: Counter[int] = field(default_factory=Counter)
    label_counts: Counter[str] = field(default_factory=Counter)
    has_bbox: bool = False
    has_confidence: bool = False
    has_tracker_id: bool = False
    sample_count: int = 0
    unique_tracker_ids: set[str] = field(default_factory=set)
    tracker_ids_sample: list[str] = field(default_factory=list)
    tracker_ids_sample_limit: int = 10
    # P3.5 analytics counters; populated only when the producer attached
    # nvdsanalytics user-meta (analytics_probe mode).
    has_analytics_metadata: bool = False
    roi_metadata_count: int = 0
    line_metadata_count: int = 0
    overcrowding_metadata_count: int = 0
    direction_metadata_count: int = 0
    objects_with_roi_metadata: int = 0
    objects_with_line_metadata: int = 0
    analytics_labels: set[str] = field(default_factory=set)
    analytics_labels_sample: list[str] = field(default_factory=list)
    analytics_labels_sample_limit: int = 20
    # P4-B observation-emit counters. Populated only when the worker has
    # an ObservationWriter AND the analytics mode is active. Surfaced in
    # the status payload so an operator can confirm writes flowed without
    # querying the api-server.
    observations_emitted: int = 0
    line_crossing_observations_emitted: int = 0
    line_crossing_object_observations_emitted: int = 0
    # P4-C: ROI occupancy emit counter (parallel to the line-crossing
    # one). Errors increment the shared ``observation_write_errors``
    # counter — the writer error path does not distinguish.
    roi_occupancy_observations_emitted: int = 0
    # P4-D: per-track zone transition counters. Same shared error
    # counter; the worker error path doesn't distinguish per-type.
    zone_entered_observations_emitted: int = 0
    zone_exited_observations_emitted: int = 0
    # P4-E: per-type dwell counters. Same shared error counter
    # (writer failures land in observation_write_errors); a separate
    # dwell_write_errors_count is also surfaced for the operator
    # so they can tell if the dwell tick path is silently dropping
    # events vs the zone path.
    dwell_started_observations_emitted: int = 0
    dwell_updated_observations_emitted: int = 0
    dwell_exceeded_observations_emitted: int = 0
    dwell_write_errors: int = 0
    observation_write_errors: int = 0
    last_observation_at: datetime | None = None
    # P5-C perception-snapshot counters. Deliberately separate from the
    # observation counters above — perception is a dedicated latest-state
    # path that posts directly via the API client, not the writer.
    perception_snapshots_posted: int = 0
    perception_snapshot_write_errors: int = 0
    last_perception_snapshot_at: datetime | None = None
    # P5-F fanout hot-path counters (mirrors of FanoutPublisher state so
    # the status payload stays a plain-counters affair).
    fanout_published: int = 0
    fanout_publish_errors: int = 0
    fanout_subscribers: int = 0

    def absorb(self, frame: NeutralFrameMetadata) -> None:
        self.frames_seen += 1
        if frame.objects:
            self.frames_with_objects += 1
            self.sample_count += 1  # one bounded sample per frame-with-objects
        for obj in frame.objects:
            self.objects_seen += 1
            self.class_counts[obj.class_id] += 1
            if obj.label:
                self.label_counts[obj.label] += 1
            if obj.confidence is not None and obj.confidence > 0.0:
                self.has_confidence = True
            if (
                obj.bbox_pixels is not None
                and obj.bbox_pixels.width > 0
                and obj.bbox_pixels.height > 0
            ):
                self.has_bbox = True
            if obj.tracker_id is not None:
                self.has_tracker_id = True
                if obj.tracker_id not in self.unique_tracker_ids:
                    self.unique_tracker_ids.add(obj.tracker_id)
                    if len(self.tracker_ids_sample) < self.tracker_ids_sample_limit:
                        self.tracker_ids_sample.append(obj.tracker_id)
            self._absorb_object_analytics(obj.debug)
        self._absorb_frame_analytics(frame.source)

    # ------------------------------------------------------------------
    # P3.5 helpers: fold the per-frame / per-object analytics blobs the
    # in-container app drops under ``frame.source["analytics"]`` /
    # ``object.debug["analytics"]`` into bounded worker-side counters.
    # All keys are optional — the producer only sets the ones it found.
    # ------------------------------------------------------------------

    def _absorb_object_analytics(self, debug: dict[str, Any]) -> None:
        if not debug:
            return
        analytics = debug.get("analytics")
        if not isinstance(analytics, dict) or not analytics:
            return
        self.has_analytics_metadata = True
        roi = analytics.get("roi_labels") or []
        lc = analytics.get("line_labels") or []
        oc = analytics.get("overcrowding_labels") or []
        direction = analytics.get("direction") or ""
        if roi:
            self.objects_with_roi_metadata += 1
            self._record_labels(roi)
        if lc:
            self.objects_with_line_metadata += 1
            self._record_labels(lc)
        if oc:
            self._record_labels(oc)
        if direction:
            self.direction_metadata_count += 1
            self._record_labels([direction])

    def _absorb_frame_analytics(self, source: dict[str, Any]) -> None:
        if not source:
            return
        analytics = source.get("analytics")
        if not isinstance(analytics, dict) or not analytics:
            return
        self.has_analytics_metadata = True
        roi_counts = analytics.get("roi_counts") or {}
        lc_cum = analytics.get("line_crossing_cumulative") or {}
        oc = analytics.get("overcrowding_status") or {}
        if roi_counts:
            self.roi_metadata_count += 1
            self._record_labels(roi_counts.keys())
        if lc_cum:
            self.line_metadata_count += 1
            self._record_labels(lc_cum.keys())
        if oc:
            self.overcrowding_metadata_count += 1
            self._record_labels(oc.keys())

    def _record_labels(self, labels: Any) -> None:
        for raw in labels:
            label = str(raw)
            if label in self.analytics_labels:
                continue
            self.analytics_labels.add(label)
            if len(self.analytics_labels_sample) < self.analytics_labels_sample_limit:
                self.analytics_labels_sample.append(label)


@dataclass
class _MetadataReaderState:
    """Bookkeeping the reader task needs to share with the parent on exit."""

    hard_fail_code: str | None = None
    last_err: str | None = None
    last_summary: dict[str, Any] | None = None
    malformed_lines: int = 0


def _queue_stats_dict(queue: BoundedBridgeQueue) -> dict[str, Any]:
    """Plain-dict view of ``BridgeQueueStats`` for embedding in status
    payloads (pydantic-friendly, JSON-safe)."""
    s = queue.stats()
    return {
        "accepted": s.accepted,
        "dropped": s.dropped,
        "dropped_newest": s.dropped_newest,
        "dropped_oldest": s.dropped_oldest,
        "consumed": s.consumed,
        "current_size": s.current_size,
        "closed": s.closed,
    }


class DeepStreamWorker(Runtime):
    def __init__(
        self,
        *,
        session: EligibleSession,
        settings: Settings,
        api: ProcessorApi,
        package: PackageSpec,
        writer: ObservationWriter | None = None,
        pm_factory: Callable[[], ProcessManager] | None = None,
    ) -> None:
        self._session = session
        self._settings = settings
        self._api = api
        self._package = package
        # P5-B: the package's registry model wins over the global
        # ``deepstream_inference_model`` setting (the env stays a dev/probe
        # default for packages that declare no model, e.g. decode_probe).
        # For the TrafficCamNet packages both resolve to the same registry
        # entry, so the rendered configs are unchanged; the status payload's
        # ``model`` field now reports the registry id (e.g.
        # ``trafficcamnet_v1``) instead of the env alias.
        self._effective_model: str = package.model_id or settings.deepstream_inference_model
        # P5-D: the canonical registry id of the effective model — used for
        # the processing-config lineage stamp (``applied.model_key``) and the
        # status payload's ``model_id``, so a YOLOX worker stamps
        # ``yolox_s_v1`` and an env alias like ``trafficcamnet`` stamps
        # ``trafficcamnet_v1``. Falls back to the raw name for models the
        # registry doesn't know (the lineage then carries what was asked).
        _model_spec = find_model_by_alias(self._effective_model)
        self._effective_model_id: str = (
            _model_spec.model_id if _model_spec is not None else self._effective_model
        )
        # P5-E: the running tracker is now PACKAGE-DRIVEN, mirroring the
        # package-driven model above. When the package declares a
        # ``tracker_profile_id`` we derive the algorithm + ll-lib / width /
        # height from the resolved registry profile; the global
        # ``deepstream_tracker_algorithm`` setting is only a fallback for
        # packages that declare no tracker profile. Resolution is restart-time
        # only, like the model. The IOU packages declare
        # ``deepstream_iou_fast_v1`` whose registry values equal the Settings
        # defaults, so their rendered tracker_config.yml is byte-identical.
        _tracker_spec = (
            find_runnable_deepstream_tracker_by_profile_id(package.tracker_profile_id)
            if package.tracker_profile_id is not None
            else None
        )
        if _tracker_spec is not None:
            _ds = _tracker_spec.deepstream
            self._effective_tracker_algorithm: str = _tracker_spec.algorithm
            self._effective_tracker_lib_path: str = _ds.get(
                "ll_lib_path_default", settings.deepstream_tracker_lib_path
            )
            self._effective_tracker_width: int = int(
                _ds.get("tracker_width", settings.deepstream_tracker_width)
            )
            self._effective_tracker_height: int = int(
                _ds.get("tracker_height", settings.deepstream_tracker_height)
            )
        elif package.tracker_profile_id is not None:
            # Declared but NOT runnable (non-runnable status / no authored
            # settings). The resolver already gates status, so this is a
            # defensive belt-and-suspenders for an empty-settings profile. Fail
            # CLOSED: keep the declared profile's algorithm so
            # ``build_tracker_config`` raises clearly at process start (the
            # worker then posts ``failed``) rather than silently running IOU.
            _raw = load_tracker_registry().get(package.tracker_profile_id)
            self._effective_tracker_algorithm = (
                _raw.algorithm if _raw is not None else package.tracker_profile_id
            )
            self._effective_tracker_lib_path = settings.deepstream_tracker_lib_path
            self._effective_tracker_width = settings.deepstream_tracker_width
            self._effective_tracker_height = settings.deepstream_tracker_height
        else:
            # No package tracker profile: global Settings fallback (the
            # pre-P5-E behavior; used by tracker-disabled / profile-less
            # packages).
            self._effective_tracker_algorithm = settings.deepstream_tracker_algorithm
            self._effective_tracker_lib_path = settings.deepstream_tracker_lib_path
            self._effective_tracker_width = settings.deepstream_tracker_width
            self._effective_tracker_height = settings.deepstream_tracker_height
        # P4-B: only the analytics mode uses the writer (today, only for
        # ``processor.line_crossing``). decode_probe / inference_probe /
        # metadata_probe / tracker_probe modes ignore it. Optional so unit
        # tests that exercise the non-analytics modes don't have to build
        # a writer when they don't care.
        self._writer = writer
        self._pm_factory = pm_factory or self._default_pm_factory
        self._pm: ProcessManager | None = None
        self._stop = asyncio.Event()
        self._state = WorkerState.INIT
        # Set only while ``_drive_metadata_probe`` is running, so
        # ``request_stop`` can close the bridge queue directly and wake a
        # consumer parked in ``asyncio.to_thread(queue.Queue.get)`` without
        # waiting for the reader's next poll iteration.
        self._metadata_queue: BoundedBridgeQueue | None = None
        self._backoff = Backoff(
            initial=settings.backoff_initial_seconds,
            factor=settings.backoff_factor,
            maximum=settings.backoff_max_seconds,
            jitter=settings.backoff_jitter,
        )
        # P6-C: resolve operator-saved geometry once. Only analytics-mode
        # workers consume geometry (line crossings, ROI occupancy, etc.); the
        # other modes ignore it. Resolution is restart-time only — a mid-
        # session revision bump is picked up on the next eligible session.
        self._resolved_geometry: ResolvedGeometry | None = None
        if self._mode == _ANALYTICS_MODE:
            fallback = compute_analytics_geometry(
                frame_width=session.approved.width,
                frame_height=session.approved.height,
                roi_margin_pct=settings.deepstream_analytics_roi_margin_pct,
            )
            self._resolved_geometry = resolve_session_geometry(
                session.geometry_config,
                frame_width=session.approved.width,
                frame_height=session.approved.height,
                fallback=fallback,
            )
            _log.info(
                "geometry resolved: camera_id=%s stream_session_id=%s source=%s "
                "revision=%s line_count=%d roi_count=%d%s",
                session.camera_id,
                session.session_id,
                self._resolved_geometry.source,
                self._resolved_geometry.revision,
                self._resolved_geometry.line_count,
                self._resolved_geometry.roi_count,
                (
                    f" error_reason={self._resolved_geometry.error_reason!r}"
                    if self._resolved_geometry.error_reason
                    else ""
                ),
            )
        # M7B/M7C: resolve the operator-saved processing config once. Done
        # defensively right next to the geometry resolve — the resolver never
        # raises. model/tracker stay at the worker's current defaults
        # (stamp-only). M7C: confidence_threshold is genuinely
        # applied, but ONLY in the modes that actually render it into a PGIE
        # config — the metadata/tracker/analytics modes that construct a
        # DeepStreamMetadataProcessManager in ``_default_pm_factory``. The
        # boolean below is the gate; the resolved APPLIED value is then the
        # single source of truth forwarded to that process manager so the
        # stamped lineage and the live PGIE config can never disagree. Resolved
        # unconditionally (not gated on mode) so every DeepStream worker carries
        # the lineage; a bad/diverging config must NOT flag a worker error — it
        # lives only in the lineage block.
        # M7D: the same per-object modes that can render confidence into the
        # PGIE config are the ones that produce the per-object output paths
        # (perception snapshot + zone/dwell), so they can apply selected_classes
        # by POST-INFERENCE filtering of those paths.
        per_object_modes = self._mode in (
            _METADATA_MODE,
            _TRACKER_MODE,
            _ANALYTICS_MODE,
        )
        confidence_apply_supported = per_object_modes
        selected_classes_apply_supported = per_object_modes
        # M7E: processing_fps is applied as an OUTPUT-CADENCE throttle on the
        # perception-snapshot emit interval (1/fps replaces the global
        # perception_snapshot_interval_seconds for this session). That seam
        # only exists in the analytics mode with perception snapshots enabled;
        # everywhere else the request stays stamp-only
        # (processing_fps_requested_not_applied). The DeepStream pipeline /
        # nvinfer interval is deliberately untouched, and the event-driven
        # observation emits are NOT throttled (dropping real events is not a
        # cadence) — the resolver stamps the honest
        # processing_fps_output_cadence_only caveat whenever fps is applied.
        processing_fps_apply_supported = self._mode == _ANALYTICS_MODE and bool(
            settings.perception_snapshot_enabled
        )
        self._resolved_processing_config: ResolvedProcessingConfig = (
            resolve_session_processing_config(
                session.processing_config,
                # P5-D: the lineage's APPLIED model is the worker's effective
                # registry model (package-driven since P5-B), so a YOLOX
                # worker stamps yolox_s_v1 and an operator request for a
                # different model is honestly recorded as
                # model_switch_requested_not_applied.
                # P5-E: the APPLIED tracker is now the worker's effective
                # tracker algorithm (package-driven), so it reflects what
                # actually runs (iou for the IOU packages, nvdcf for the NvDCF
                # package) instead of the static "iou" default. An operator
                # tracker request that diverges is still recorded as
                # tracker_requested_not_applied (tracker stays package-driven,
                # not operator-applied).
                defaults=replace(
                    DEFAULT_APPLIED,
                    model_key=self._effective_model_id,
                    tracker=self._effective_tracker_algorithm,
                ),
                confidence_apply_supported=confidence_apply_supported,
                selected_classes_apply_supported=selected_classes_apply_supported,
                processing_fps_apply_supported=processing_fps_apply_supported,
            )
        )
        # M7D: single source of truth for the per-object class filter — the
        # resolved APPLIED list ([] = no filter; filter_objects_by_class treats
        # an empty list as "return unchanged"). Per-worker only; no global state.
        # P5-D: when the operator has NOT narrowed classes, the PACKAGE's
        # declared supported_classes become the filter. Live-found: a COCO-80
        # detector without this floods the zone/dwell paths with confidence-
        # flapping background objects (books/tv/chairs) inside the ROI —
        # 12 zone_entered / 11 dwell_exceeded from furniture in one 190s run.
        # TrafficCamNet packages declare no supported_classes, so they are
        # byte-identical to pre-P5-D. The lineage stamp is NOT touched: the
        # package bound is package metadata, not an operator request.
        operator_selected = tuple(self._resolved_processing_config.applied.selected_classes)
        self._selected_classes: tuple[str, ...] = operator_selected or tuple(
            package.supported_classes
        )
        # P5-D: single-class packages also narrow the nvdsanalytics AGGREGATE
        # counts (ROI occupancy / line counters) to that class — the
        # nvdsanalytics parser accepts exactly one int per section, which is
        # precisely the person-only people-analytics case. Multi-class and
        # class-less packages keep the env setting (default "all"), unchanged.
        self._analytics_class_ids: str = self._package_analytics_class_ids()
        # M7E: single source of truth for the perception-snapshot cadence —
        # the resolved APPLIED fps (1/fps), falling back to the global setting
        # when no fps is applied. Per-worker only (no global Settings
        # mutation, no cross-camera leakage); defensively guarded so a
        # non-positive value can never produce a zero/negative interval.
        applied_fps = self._resolved_processing_config.applied.processing_fps
        self._perception_interval_seconds: float = (
            1.0 / float(applied_fps)
            if applied_fps is not None and applied_fps > 0
            else float(settings.perception_snapshot_interval_seconds)
        )
        _log.info(
            "processing config resolved: camera_id=%s stream_session_id=%s source=%s "
            "revision=%s status=%s warnings=%s%s",
            session.camera_id,
            session.session_id,
            self._resolved_processing_config.source,
            self._resolved_processing_config.revision,
            self._resolved_processing_config.status,
            list(self._resolved_processing_config.warnings),
            (
                f" error_reason={self._resolved_processing_config.error_reason!r}"
                if self._resolved_processing_config.error_reason
                else ""
            ),
        )

    # --- Runtime interface ------------------------------------------------

    @property
    def session_id(self) -> UUID:
        return self._session.session_id

    @property
    def state(self) -> WorkerState:
        return self._state

    def request_stop(self) -> None:
        """Idempotent: sets the stop event AND closes the active metadata
        bridge queue if one exists. Closing the queue wakes any consumer
        parked in ``asyncio.to_thread(queue.Queue.get)`` immediately
        (otherwise the consumer would only wake when the reader's next
        poll iteration sees ``self._stop`` and closes the queue in its
        ``finally``)."""
        self._stop.set()
        queue = self._metadata_queue
        if queue is not None:
            queue.close()

    def _package_analytics_class_ids(self) -> str:
        """The nvdsanalytics ``class-id`` value for this worker: the model's
        class id when the package declares exactly ONE supported class that
        the model registry maps (e.g. person → 0 for YOLOX), else the env
        setting (``deepstream_analytics_class_ids``, default "all")."""
        classes = self._package.supported_classes
        if len(classes) != 1:
            return self._settings.deepstream_analytics_class_ids
        spec = find_model_by_alias(self._effective_model)
        if spec is None:
            return self._settings.deepstream_analytics_class_ids
        class_id = spec.class_id_map.get(classes[0])
        if class_id is None:
            return self._settings.deepstream_analytics_class_ids
        return str(class_id)

    @property
    def _mode(self) -> str:
        """``analytics_probe`` (P3.5) / ``tracker_probe`` (P3) /
        ``metadata_probe`` (P2) / ``inference_probe`` (M8.2C1-B) /
        ``decode_probe`` (M8.2B), from the resolved package's
        ``supports``. Order matters: a package that advertises multiple
        supports falls through to the most specific mode
        (analytics > tracker > metadata > inference > decode)."""
        if self._package.supports.get("analytics_probe"):
            return _ANALYTICS_MODE
        if self._package.supports.get("tracker_probe"):
            return _TRACKER_MODE
        if self._package.supports.get("metadata_probe"):
            return _METADATA_MODE
        if self._package.supports.get("inference_probe"):
            return _INFERENCE_MODE
        return _DECODE_MODE

    def _default_pm_factory(self) -> ProcessManager:
        if self._mode in (_METADATA_MODE, _TRACKER_MODE, _ANALYTICS_MODE):
            tracker_enabled = self._mode in (_TRACKER_MODE, _ANALYTICS_MODE)
            analytics_enabled = self._mode == _ANALYTICS_MODE
            # P6-C: forward the resolved operator-saved (or fallback) geometry
            # to the analytics process manager. None outside of analytics mode.
            analytics_geometry: AnalyticsGeometry | None = (
                self._resolved_geometry.geometry
                if analytics_enabled and self._resolved_geometry is not None
                else None
            )
            return DeepStreamMetadataProcessManager(
                rtsp_url=self._session.internal_rtsp_url,
                image=self._settings.deepstream_pyds_image,
                network=self._settings.deepstream_docker_network,
                rtsp_transport=self._settings.deepstream_rtsp_transport,
                probe_timeout_seconds=self._settings.deepstream_metadata_probe_timeout_seconds,
                stop_timeout_seconds=self._settings.deepstream_process_stop_timeout_seconds,
                model=self._effective_model,
                model_store_dir=self._settings.deepstream_model_store_dir,
                output_base_dir=self._settings.deepstream_inference_output_dir,
                keep_temp_files=self._settings.deepstream_keep_temp_files,
                sample_limit=self._settings.deepstream_metadata_sample_limit,
                stream_session_id=str(self._session.session_id),
                width=self._session.approved.width,
                height=self._session.approved.height,
                tracker_enabled=tracker_enabled,
                # P5-E: package-driven tracker (see __init__). Falls back to the
                # global Settings values only when the package declares no
                # tracker profile.
                tracker_algorithm=self._effective_tracker_algorithm,
                tracker_lib_path=self._effective_tracker_lib_path,
                tracker_width=self._effective_tracker_width,
                tracker_height=self._effective_tracker_height,
                analytics_enabled=analytics_enabled,
                analytics_roi_margin_pct=(self._settings.deepstream_analytics_roi_margin_pct),
                analytics_class_ids=self._analytics_class_ids,
                analytics_geometry=analytics_geometry,
                gpus=self._settings.deepstream_gpus,
                # M7C: the resolved APPLIED confidence threshold (a float when
                # applied for this mode, None otherwise) — the single source of
                # truth shared with the stamped lineage so they can't diverge.
                confidence_threshold=self._resolved_processing_config.applied.confidence_threshold,
            )
        if self._mode == _INFERENCE_MODE:
            return DeepStreamInferenceProcessManager(
                rtsp_url=self._session.internal_rtsp_url,
                image=self._settings.deepstream_image,
                network=self._settings.deepstream_docker_network,
                rtsp_transport=self._settings.deepstream_rtsp_transport,
                probe_timeout_seconds=self._settings.deepstream_inference_timeout_seconds,
                stop_timeout_seconds=self._settings.deepstream_process_stop_timeout_seconds,
                model=self._settings.deepstream_inference_model,
                output_base_dir=self._settings.deepstream_inference_output_dir,
                keep_temp_files=self._settings.deepstream_keep_temp_files,
                width=self._session.approved.width,
                height=self._session.approved.height,
                gpus=self._settings.deepstream_gpus,
            )
        return DeepStreamProcessManager(
            rtsp_url=self._session.internal_rtsp_url,
            image=self._settings.deepstream_image,
            network=self._settings.deepstream_docker_network,
            rtsp_transport=self._settings.deepstream_rtsp_transport,
            probe_timeout_seconds=self._settings.deepstream_probe_timeout_seconds,
            stop_timeout_seconds=self._settings.deepstream_process_stop_timeout_seconds,
            gpus=self._settings.deepstream_gpus,
        )

    async def run(self) -> None:
        try:
            if not self._session.internal_rtsp_url:
                self._state = WorkerState.FAILED
                await self._post_best_effort(
                    "failed",
                    error_code="no_rtsp_url",
                    error_message="eligible session missing internal_rtsp_url",
                )
                return

            self._state = WorkerState.STARTING
            if not await self._post_gating("starting"):
                return
            _log.info(
                "deepstream worker session=%s package=%s mode=%s url=%s",
                self.session_id,
                self._package.package_id,
                self._mode,
                redact_rtsp(self._session.internal_rtsp_url),
            )
            self._pm = self._pm_factory()
            await self._pm.start()

            if self._mode in (_METADATA_MODE, _TRACKER_MODE, _ANALYTICS_MODE):
                await self._drive_metadata_probe()
            elif self._mode == _INFERENCE_MODE:
                await self._drive_inference()
            else:
                await self._drive_decode()
        except ApiSessionGoneError:
            self._state = WorkerState.FAILED
            _log.info("session %s gone; stopping quietly", self.session_id)
            await self._safe_stop_pm()
        except ApiValidationError as exc:
            self._state = WorkerState.FAILED
            _log.error("contract error for session %s: %s", self.session_id, exc)
            await self._safe_stop_pm()
            await self._post_best_effort(
                "failed", error_code="validation_error", error_message=str(exc)
            )
        except asyncio.CancelledError:
            await self._safe_stop_pm()
            raise
        except Exception as exc:  # never crash the supervisor
            self._state = WorkerState.FAILED
            _log.exception("deepstream worker session=%s crashed", self.session_id)
            await self._safe_stop_pm()
            await self._post_best_effort(
                "failed", error_code="worker_error", error_message=repr(exc)
            )

    # --- decode-probe driver (M8.2B, unchanged behaviour) ----------------

    async def _drive_decode(self) -> None:
        assert self._pm is not None
        attached = False
        last_proc_post = 0.0
        last_fps: float | None = None
        last_err: str | None = None

        while not self._stop.is_set():
            try:
                line = await asyncio.wait_for(self._pm.readline(), timeout=_READLINE_POLL_SECONDS)
            except TimeoutError:
                continue
            if line is None:
                break  # decode process ended
            if "ERROR" in line:
                last_err = scrub_rtsp_in_text(line)[:480]
            parsed = _parse_fps(line)
            if parsed is None:
                continue
            frames, dropped, fps = parsed
            last_fps = fps
            if not attached and frames > 0:
                attached = True
                self._state = WorkerState.ATTACHED
                if not await self._post_gating("attached", fps_processed=fps):
                    break
                self._state = WorkerState.PROCESSING
            if attached:
                now = time.monotonic()
                if now - last_proc_post >= self._settings.status_interval_seconds:
                    last_proc_post = now
                    await self._post(
                        "processing",
                        transient_ok=True,
                        fps_processed=fps,
                        last_frame_at=now_utc(),
                        payload={
                            "runtime": "deepstream",
                            "frames": frames,
                            "dropped": dropped,
                            "decoder": "nvv4l2decoder",
                        },
                    )

        await self._safe_stop_pm()
        rc = self._pm.returncode if self._pm else None
        graceful = self._stop.is_set() or rc in _CLEAN_EXIT_CODES
        if graceful:
            self._state = WorkerState.STOPPED
            await self._post_best_effort("stopped", fps_processed=last_fps)
        else:
            self._state = WorkerState.FAILED
            await self._post_best_effort(
                "failed",
                error_code="deepstream_exit",
                error_message=(f"decode process exited rc={rc}; last_error={last_err or 'n/a'}"),
            )

    # --- inference-probe driver (M8.2C1-B) -------------------------------

    def _infer_payload(
        self,
        *,
        engine_built: bool,
        fps: float | None,
        frames: int | None = None,
        detections: int | None = None,
    ) -> dict[str, Any]:
        """Status payload for the inference probe. All fields are
        status/debug evidence only — NOT metadata extraction."""
        p: dict[str, Any] = {
            "runtime": "deepstream",
            "mode": _INFERENCE_MODE,
            "inference_enabled": True,
            "inference_engine_built": engine_built,
            "model": self._settings.deepstream_inference_model,
            "image": self._settings.deepstream_image,
        }
        if fps is not None:
            p["fps_processed"] = fps
        if frames is not None:
            p["detection_frames_count"] = frames
        if detections is not None:
            p["detection_count"] = detections
        return p

    def _kitti_stats(self) -> tuple[int | None, int | None]:
        """Best-effort KITTI (frame_count, detection_lines) from the process
        manager, if it exposes them. Status evidence only."""
        fn = getattr(self._pm, "kitti_stats", None)
        if not callable(fn):
            return (None, None)
        try:
            frames, dets = fn()
            return (int(frames), int(dets))
        except Exception:  # evidence is best-effort, never fatal
            return (None, None)

    async def _drive_inference(self) -> None:
        assert self._pm is not None
        attached = False
        processing = False
        engine_built = False
        last_proc_post = 0.0
        last_fps: float | None = None
        last_err: str | None = None
        hard_fail_code: str | None = None

        while not self._stop.is_set():
            try:
                line = await asyncio.wait_for(self._pm.readline(), timeout=_READLINE_POLL_SECONDS)
            except TimeoutError:
                continue
            if line is None:
                break  # inference process ended
            safe = scrub_rtsp_in_text(line)

            if hard_fail_code is None and _HARD_FAIL_RE.search(line):
                hard_fail_code = (
                    "deepstream_unsupported_sm"
                    if "unsupported sm" in line.lower()
                    else "deepstream_trt_error"
                )
                last_err = safe[:480]
                continue  # let the process die; we will post `failed`
            if "ERROR" in line and last_err is None:
                last_err = safe[:480]
            if not engine_built and _ENGINE_OK_RE.search(line):
                engine_built = True
            if not attached and (_ATTACH_RE.search(line) or engine_built):
                attached = True
                self._state = WorkerState.ATTACHED
                if not await self._post_gating("attached"):
                    break

            fps = _parse_perf(line)
            if fps is not None:
                last_fps = fps

            evidence = engine_built or fps is not None
            if attached and evidence and not processing:
                processing = True
                self._state = WorkerState.PROCESSING
                last_proc_post = time.monotonic()
                await self._post(
                    "processing",
                    transient_ok=True,
                    fps_processed=last_fps,
                    last_frame_at=now_utc(),
                    payload=self._infer_payload(engine_built=engine_built, fps=last_fps),
                )
            elif processing:
                now = time.monotonic()
                if now - last_proc_post >= self._settings.status_interval_seconds:
                    last_proc_post = now
                    await self._post(
                        "processing",
                        transient_ok=True,
                        fps_processed=last_fps,
                        last_frame_at=now_utc(),
                        payload=self._infer_payload(engine_built=engine_built, fps=last_fps),
                    )

        frames, dets = self._kitti_stats()
        await self._safe_stop_pm()
        rc = self._pm.returncode if self._pm else None
        graceful = hard_fail_code is None and (self._stop.is_set() or rc in _CLEAN_EXIT_CODES)
        payload = self._infer_payload(
            engine_built=engine_built,
            fps=last_fps,
            frames=frames,
            detections=dets,
        )
        if graceful:
            self._state = WorkerState.STOPPED
            await self._post_best_effort("stopped", fps_processed=last_fps, payload=payload)
        else:
            self._state = WorkerState.FAILED
            await self._post_best_effort(
                "failed",
                error_code=hard_fail_code or "deepstream_exit",
                error_message=(f"inference process rc={rc}; last_error={last_err or 'n/a'}"),
                payload=payload,
            )

    # --- metadata-probe driver (P2) --------------------------------------

    def _metadata_payload(
        self,
        *,
        counters: _MetadataCounters,
        queue_stats: dict[str, Any],
        last_app_summary: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Status payload for the metadata / tracker probe. Includes
        counters, queue stats, and the in-container app's terminal summary
        if available. Deliberately small — no per-frame samples are
        forwarded. In ``tracker_probe`` mode the payload also carries
        ``tracker_enabled`` / ``unique_tracker_ids_count`` /
        ``tracker_ids_sample`` so the operator can confirm stable IDs
        without scraping container logs."""
        tracker_enabled = self._mode in (_TRACKER_MODE, _ANALYTICS_MODE)
        analytics_enabled = self._mode == _ANALYTICS_MODE
        p: dict[str, Any] = {
            "runtime": "deepstream",
            "mode": self._mode,
            "metadata_enabled": True,
            "metadata_source": "pyds_pad_probe",
            # P5-B: the EFFECTIVE model (package model_id when declared,
            # else the global setting) so a YOLO probe is distinguishable
            # from the TrafficCamNet probes in status evidence.
            "model": self._effective_model,
            # P5-D: canonical registry component ids for package-readiness
            # evidence. Bounded scalars only.
            "model_id": self._effective_model_id,
            "tracker_profile_id": self._package.tracker_profile_id,
            "analytics_profile_id": self._package.analytics_profile_id,
            "image": self._settings.deepstream_pyds_image,
            "frames_seen": counters.frames_seen,
            "frames_with_objects": counters.frames_with_objects,
            "objects_seen": counters.objects_seen,
            "class_counts": dict(counters.class_counts),
            "label_counts": dict(counters.label_counts),
            "has_bbox": counters.has_bbox,
            "has_confidence": counters.has_confidence,
            "has_tracker_id": counters.has_tracker_id,
            "sample_count": counters.sample_count,
            "queue_stats": queue_stats,
        }
        if tracker_enabled:
            p["tracker_enabled"] = True
            p["tracker_algorithm"] = self._effective_tracker_algorithm
            p["unique_tracker_ids_count"] = len(counters.unique_tracker_ids)
            p["tracker_ids_sample"] = list(counters.tracker_ids_sample)
        if analytics_enabled:
            p["analytics_enabled"] = True
            p["has_analytics_metadata"] = counters.has_analytics_metadata
            p["roi_metadata_count"] = counters.roi_metadata_count
            p["line_metadata_count"] = counters.line_metadata_count
            p["overcrowding_metadata_count"] = counters.overcrowding_metadata_count
            p["direction_metadata_count"] = counters.direction_metadata_count
            p["objects_with_roi_metadata"] = counters.objects_with_roi_metadata
            p["objects_with_line_metadata"] = counters.objects_with_line_metadata
            p["analytics_labels_count"] = len(counters.analytics_labels)
            p["analytics_labels_sample"] = list(counters.analytics_labels_sample)
            # P4-D follow-up: live-debug diagnostics. When zone events
            # don't emit, the operator needs to distinguish "no detection"
            # from "wrong class" from "no ROI tag" from "adapter
            # suppressed". The first three are pre-adapter; the fourth
            # is a P4-D regression. Surface enough of the upstream state
            # in the status payload to triage without re-running.
            # P5-D: the EFFECTIVE class filter (package-derived for
            # single-class packages), not the raw env setting.
            p["analytics_class_ids"] = self._analytics_class_ids
            p["analytics_roi_margin_pct"] = float(
                self._settings.deepstream_analytics_roi_margin_pct
            )
            # ``object_class_id_counts`` is the count of OBJECTS (not
            # frames) per detected class_id seen by the worker; it
            # would already be in ``class_counts`` for metadata_probe
            # mode but we explicitly surface it under an analytics-
            # specific name here so the live triage flow doesn't need
            # to look at two different keys.
            p["analytics_object_class_id_counts"] = {
                str(k): int(v) for k, v in counters.class_counts.items()
            }
            # ``objects_with_roi_labels_count`` is the count of
            # OBJECTS that had a non-empty per-object roiStatus list
            # (i.e., ``roi_labels`` was attached). Equivalent to
            # ``objects_with_roi_metadata`` but the name matches the
            # diagnostic checklist in docs/local-dev.md so an operator
            # can grep without rederiving.
            p["objects_with_roi_labels_count"] = counters.objects_with_roi_metadata
            p["roi_labels_sample"] = list(counters.analytics_labels_sample)
            # P4-B: observation-emit counters only land in the status
            # payload when the worker actually has a writer (analytics
            # + writer present). The flag stays False so an operator can
            # tell the difference between "no writer wired" and "writer
            # wired but no events yet". Counters are omitted when no
            # writer is present, keeping the payload tight.
            if self._writer is not None:
                p["observations_enabled"] = True
                p["observations_emitted_count"] = counters.observations_emitted
                p["line_crossing_observations_emitted"] = (
                    counters.line_crossing_observations_emitted
                )
                p["line_crossing_object_observations_emitted"] = (
                    counters.line_crossing_object_observations_emitted
                )
                # P4-C: parallel ROI occupancy counter so operators can
                # tell line vs ROI throughput apart from the status row.
                p["roi_occupancy_observations_emitted"] = (
                    counters.roi_occupancy_observations_emitted
                )
                # P4-C cleanup: surface the debounce configuration so an
                # operator can tell "no observations because steady scene"
                # apart from "no observations because debounce is too
                # aggressive". Counts >1 actively suppress short flaps.
                p["roi_change_stability_frames"] = int(
                    self._settings.deepstream_roi_change_stability_frames
                )
                # P4-D: per-track zone transition counters + the
                # configured membership stability threshold. ``_enabled``
                # mirrors ``observations_enabled``'s shape — True when
                # the analytics worker has a writer, regardless of
                # whether any events have flowed yet.
                p["zone_transition_observations_enabled"] = True
                p["zone_entered_observations_emitted"] = counters.zone_entered_observations_emitted
                p["zone_exited_observations_emitted"] = counters.zone_exited_observations_emitted
                p["zone_membership_stability_frames"] = int(
                    self._settings.deepstream_zone_membership_stability_frames
                )
                # P4-E: dwell counters + configured timers. ``_enabled``
                # mirrors ``observations_enabled`` since dwell lives or
                # dies with the writer + analytics mode. The active
                # session count is a triage signal — non-zero means
                # the worker has been seeing stable zone_entered events
                # without a matching zone_exited yet.
                p["dwell_observations_enabled"] = True
                p["dwell_started_observations_emitted"] = (
                    counters.dwell_started_observations_emitted
                )
                p["dwell_updated_observations_emitted"] = (
                    counters.dwell_updated_observations_emitted
                )
                p["dwell_exceeded_observations_emitted"] = (
                    counters.dwell_exceeded_observations_emitted
                )
                p["dwell_write_errors_count"] = counters.dwell_write_errors
                p["dwell_update_interval_seconds"] = float(
                    self._settings.deepstream_dwell_update_interval_seconds
                )
                p["dwell_exceeded_threshold_seconds"] = float(
                    self._settings.deepstream_dwell_exceeded_threshold_seconds
                )
                p["observation_write_errors_count"] = counters.observation_write_errors
                p["last_observation_at"] = (
                    counters.last_observation_at.isoformat()
                    if counters.last_observation_at is not None
                    else None
                )
            else:
                p["observations_enabled"] = False
            # P5-C: perception-snapshot counters. Independent of the writer
            # (perception posts directly via the API client), so surfaced
            # whenever the analytics mode is active. ``_enabled`` lets an
            # operator tell "off by config" apart from "on but nothing
            # posted yet".
            p["perception_snapshots_enabled"] = bool(self._settings.perception_snapshot_enabled)
            if self._settings.perception_snapshot_enabled:
                p["perception_snapshots_posted"] = counters.perception_snapshots_posted
                p["perception_snapshot_write_errors_count"] = (
                    counters.perception_snapshot_write_errors
                )
                # M7E: report the EFFECTIVE per-session cadence (1/fps when an
                # operator processing_fps is applied), not the global default.
                p["perception_snapshot_interval_seconds"] = float(self._perception_interval_seconds)
                p["last_perception_snapshot_at"] = (
                    counters.last_perception_snapshot_at.isoformat()
                    if counters.last_perception_snapshot_at is not None
                    else None
                )
            # P5-F: fanout hot-path evidence. ``_enabled`` distinguishes
            # "not configured" from "configured but nothing published yet".
            fanout_configured = bool(
                self._settings.fanout_publish_url
                and self._settings.fanout_publish_secret is not None
            )
            p["fanout_publish_enabled"] = fanout_configured
            if fanout_configured:
                p["fanout_published_count"] = counters.fanout_published
                p["fanout_publish_errors_count"] = counters.fanout_publish_errors
                p["fanout_subscribers"] = counters.fanout_subscribers
        if last_app_summary is not None:
            p["app_summary"] = last_app_summary
        return p

    async def _finalize_metadata_probe(
        self,
        *,
        queue: BoundedBridgeQueue,
        counters: _MetadataCounters,
        reader_state: _MetadataReaderState,
    ) -> None:
        """Terminal cleanup for the metadata probe. Called from a shielded
        ``finally`` so it runs to completion even if the worker task is
        being cancelled (e.g., supervisor force-stop after the per-worker
        budget). Idempotent: closes the queue, stops the PM, posts the
        terminal status. ``DeepStreamMetadataProcessManager.stop()`` is
        itself idempotent so a second call (from the outer ``run()``'s
        ``except`` handlers) is a no-op."""
        queue.close()
        await self._safe_stop_pm()
        rc = self._pm.returncode if self._pm else None
        payload = self._metadata_payload(
            counters=counters,
            queue_stats=_queue_stats_dict(queue),
            last_app_summary=reader_state.last_summary,
        )
        graceful = reader_state.hard_fail_code is None and (
            self._stop.is_set() or rc in _CLEAN_EXIT_CODES
        )
        if graceful:
            self._state = WorkerState.STOPPED
            await self._post_best_effort("stopped", payload=payload)
        else:
            self._state = WorkerState.FAILED
            await self._post_best_effort(
                "failed",
                error_code=reader_state.hard_fail_code or "deepstream_exit",
                error_message=(
                    f"metadata process rc={rc}; last_error={reader_state.last_err or 'n/a'}"
                ),
                payload=payload,
            )

    async def _drive_metadata_probe(self) -> None:
        assert self._pm is not None

        policy = (
            DropPolicy.DROP_OLDEST
            if self._settings.deepstream_metadata_queue_drop_policy == "drop_oldest"
            else DropPolicy.DROP_NEWEST
        )
        queue = BoundedBridgeQueue(
            maxsize=int(self._settings.deepstream_metadata_queue_maxsize),
            drop_policy=policy,
        )
        # Expose the queue to ``request_stop`` so a SIGINT-driven stop wakes
        # the consumer immediately (instead of waiting for the reader's next
        # poll iteration to close the queue from its own ``finally``).
        self._metadata_queue = queue
        source = BridgeQueueMetadataSource(queue)
        counters = _MetadataCounters(
            tracker_ids_sample_limit=int(self._settings.deepstream_tracker_ids_sample_limit),
            analytics_labels_sample_limit=int(
                self._settings.deepstream_analytics_labels_sample_limit
            ),
        )
        reader_state = _MetadataReaderState()

        # P4-B/P4-C: only the analytics mode emits observations today,
        # and only when the supervisor handed us a writer. Both branches
        # build the state machines + the per-emit ``source`` block once
        # so the consumer's hot path stays allocation-light.
        observations_enabled = self._mode == _ANALYTICS_MODE and self._writer is not None
        line_state: LineCrossingState | None = (
            LineCrossingState(stream_session_id=self._session.session_id)
            if observations_enabled
            else None
        )
        # Side task (per-object line crossing): rising-edge per-(track,
        # line) crossing evidence from lcStatus, emitted ALONGSIDE the
        # aggregate cumulative-counter events above. Same gate.
        line_object_state: LineCrossingObjectState | None = (
            LineCrossingObjectState(stream_session_id=self._session.session_id)
            if observations_enabled
            else None
        )
        # P4-C: ROI occupancy state runs in parallel; emit policy is
        # first-seen + on-change + ``min_interval_seconds`` heartbeat
        # (never per-frame). The interval is env-overridable via
        # ``UPSTREAM_DEEPSTREAM_ROI_MIN_INTERVAL_SECONDS`` (default 30s).
        # P4-C cleanup: a debounce of ``change_stability_frames`` frames
        # (default 5, env-overridable via
        # ``UPSTREAM_DEEPSTREAM_ROI_CHANGE_STABILITY_FRAMES``) suppresses
        # short detection-count flaps before a change is committed.
        roi_state: RoiOccupancyState | None = (
            RoiOccupancyState(
                stream_session_id=self._session.session_id,
                emit_on_change=True,
                min_interval_seconds=self._settings.deepstream_roi_min_interval_seconds,
                change_stability_frames=self._settings.deepstream_roi_change_stability_frames,
            )
            if observations_enabled
            else None
        )
        # P4-D: per-track zone transitions. Same gate as line/ROI state;
        # stability threshold from settings (default 5; tests override
        # via ``fast_settings`` to 1). Disappearance does NOT emit
        # zone_exited — see ``ZoneMembershipState``'s docstring.
        zone_state: ZoneMembershipState | None = (
            ZoneMembershipState(
                stream_session_id=self._session.session_id,
                membership_stability_frames=(
                    self._settings.deepstream_zone_membership_stability_frames
                ),
            )
            if observations_enabled
            else None
        )
        # P4-E: DwellState ingests zone transition events from
        # ``zone_state`` (via ``_emit_zone_transitions``) and emits
        # dwell_started immediately + dwell_updated periodically +
        # dwell_exceeded once. Same observations_enabled gate; the
        # state always exists when zone state does (and vice versa).
        dwell_state: DwellState | None = (
            DwellState(
                stream_session_id=self._session.session_id,
                update_interval_seconds=(self._settings.deepstream_dwell_update_interval_seconds),
                exceeded_threshold_seconds=(
                    self._settings.deepstream_dwell_exceeded_threshold_seconds
                ),
            )
            if observations_enabled
            else None
        )
        source_block: dict[str, Any] = (
            deepstream_source_block(
                package_id=self._package.package_id,
                tracker_algorithm=self._effective_tracker_algorithm,
                # P6-C: stamp every emitted line-crossing / ROI observation
                # with the geometry lineage so downstream consumers can
                # correlate counts back to a specific dashboard save.
                geometry_source=(
                    self._resolved_geometry.source if self._resolved_geometry is not None else None
                ),
                geometry_revision=(
                    self._resolved_geometry.revision
                    if self._resolved_geometry is not None
                    else None
                ),
                # M8: this single source_block is threaded into ALL emits
                # (line / ROI / zone / dwell), so stamping the SAME status
                # lineage here makes every observation carry it uniformly. The
                # line/ROI aggregate counts stay all-class — the honest
                # selected_classes_analytics_counts_not_filtered caveat rides
                # inside this block's warnings. Additive only; no count change.
                applied_processing_config=self._resolved_processing_config.status_payload(),
            )
            if observations_enabled
            else {}
        )
        # P5-C: low-rate latest perception snapshot. Analytics mode only;
        # does NOT require a writer (perception posts directly via the API
        # client). The ROI/line geometry is frame-size-invariant for the
        # session, so it is computed once here and reused every snapshot.
        # P6-C: the snapshot embeds the *applied* geometry — operator-saved
        # when available, otherwise probe-default — so the dashboard overlay
        # matches the rules nvdsanalytics is running with.
        perception_enabled = (
            self._mode == _ANALYTICS_MODE and self._settings.perception_snapshot_enabled
        )
        # P5-F: the fanout hot path is analytics-mode-only like the cold
        # path, enabled only when BOTH url + secret are configured. It
        # reuses the same snapshot builder/geometry so the dashboard's one
        # renderer serves both paths (and geometry rides EVERY hot frame).
        fanout_publisher: FanoutPublisher | None = None
        if (
            self._mode == _ANALYTICS_MODE
            and self._settings.fanout_publish_url
            and self._settings.fanout_publish_secret is not None
        ):
            fanout_publisher = FanoutPublisher(
                url=self._settings.fanout_publish_url,
                secret=self._settings.fanout_publish_secret.get_secret_value(),
                hot_interval_seconds=1.0 / max(self._settings.fanout_publish_hz, 0.1),
                idle_interval_seconds=1.0 / max(self._settings.fanout_publish_idle_hz, 0.01),
                timeout_seconds=self._settings.fanout_publish_timeout_seconds,
            )
        perception_geometry: dict[str, Any] = (
            self._resolved_geometry.geometry.to_json()
            if (perception_enabled or fanout_publisher is not None)
            and self._resolved_geometry is not None
            else {}
        )

        # P5-C follow-up: decouple the perception upload from the frame loop.
        # The loop builds a snapshot (cheap, local) and drops it in a single
        # coalescing slot; this background task drains the slot and does the
        # ~330ms POST round-trip OFF the hot path. Without it the loop awaited
        # every upload inline, so frame consumption (and thus observation
        # sampling) was pinned to network-round-trip rate (~3 FPS) even with
        # an idle GPU and a 30 FPS source. "Coalescing" = only the freshest
        # snapshot is ever sent; intermediate ones are dropped if a POST is in
        # flight, so the overlay always shows the latest frame, never a
        # backlog. ``fatal`` relays an ``ApiSessionGoneError`` back to the loop
        # so a vanished session still ends the worker (as the old inline path
        # did). The slot/events are created here (enclosing scope) so the
        # teardown ``finally`` below can flush the sender before the terminal
        # status post reads the final perception counters.
        perception_slot: dict[str, Any] = {"pending": None, "fatal": None}
        perception_wakeup = asyncio.Event()
        perception_stop = asyncio.Event()

        # P5-F: the fanout sender mirrors the perception sender exactly —
        # depth-1 coalescing slot, background task, drain-then-exit — so a
        # slow/dead sidecar can never throttle the frame loop. Unlike the
        # perception sender there is no fatal relay: fanout failures are
        # best-effort by contract (publisher counts + drops to idle cadence).
        fanout_slot: dict[str, Any] = {"pending": None}
        fanout_wakeup = asyncio.Event()
        fanout_stop = asyncio.Event()

        async def _fanout_sender() -> None:
            assert fanout_publisher is not None  # gated at task creation
            while True:
                await fanout_wakeup.wait()
                fanout_wakeup.clear()
                envelope = fanout_slot["pending"]
                fanout_slot["pending"] = None
                if envelope is not None:
                    await fanout_publisher.publish(envelope)  # never raises
                    counters.fanout_published = fanout_publisher.published
                    counters.fanout_publish_errors = fanout_publisher.errors
                    counters.fanout_subscribers = fanout_publisher.subscribers
                if fanout_stop.is_set() and fanout_slot["pending"] is None:
                    return

        async def _perception_sender() -> None:
            while True:
                await perception_wakeup.wait()
                perception_wakeup.clear()
                payload = perception_slot["pending"]
                perception_slot["pending"] = None
                if payload is not None:
                    try:
                        await self._post_perception_payload(payload, counters)
                    except ApiSessionGoneError as exc:
                        perception_slot["fatal"] = exc
                        return
                # Drain-then-exit: when asked to stop, send the final pending
                # snapshot (above) before returning so it is not lost.
                if perception_stop.is_set() and perception_slot["pending"] is None:
                    return

        async def _reader() -> None:
            assert self._pm is not None
            try:
                while not self._stop.is_set():
                    try:
                        line = await asyncio.wait_for(
                            self._pm.readline(), timeout=_READLINE_POLL_SECONDS
                        )
                    except TimeoutError:
                        continue
                    if line is None:
                        break
                    safe = scrub_rtsp_in_text(line)
                    if reader_state.hard_fail_code is None and _HARD_FAIL_RE.search(line):
                        reader_state.hard_fail_code = (
                            "deepstream_unsupported_sm"
                            if "unsupported sm" in line.lower()
                            else "deepstream_trt_error"
                        )
                        reader_state.last_err = safe[:480]
                        continue
                    parsed = parse_metadata_line(line)
                    if parsed is None:
                        if "ERROR" in line and reader_state.last_err is None:
                            reader_state.last_err = safe[:480]
                        continue
                    kind, payload_json = parsed
                    try:
                        payload = json.loads(payload_json)
                    except json.JSONDecodeError:
                        reader_state.malformed_lines += 1
                        continue
                    if kind == "summary":
                        reader_state.last_summary = payload
                        continue
                    try:
                        frame = NeutralFrameMetadata.model_validate(payload)
                    except Exception:
                        reader_state.malformed_lines += 1
                        continue
                    queue.put_from_thread(frame)
            finally:
                queue.close()

        async def _consumer() -> None:
            last_proc_post = 0.0
            last_perception_post = 0.0
            last_fanout_post = 0.0
            attached = False
            processing = False
            async for frame in source:
                # Relay a session-gone raised by the background perception
                # sender so the worker ends quietly (old inline behavior).
                if perception_slot["fatal"] is not None:
                    raise perception_slot["fatal"]
                # M7D: status debug counters stay RAW (what the detector
                # actually produced) — intentionally NOT class-filtered.
                counters.absorb(frame)
                if line_state is not None:
                    await self._emit_line_crossings(
                        frame=frame,
                        line_state=line_state,
                        source_block=source_block,
                        counters=counters,
                    )
                if line_object_state is not None:
                    await self._emit_line_crossing_objects(
                        frame=frame,
                        line_object_state=line_object_state,
                        source_block=source_block,
                        counters=counters,
                    )
                if roi_state is not None:
                    await self._emit_roi_occupancies(
                        frame=frame,
                        roi_state=roi_state,
                        source_block=source_block,
                        counters=counters,
                    )
                if zone_state is not None:
                    await self._emit_zone_transitions(
                        frame=frame,
                        zone_state=zone_state,
                        source_block=source_block,
                        counters=counters,
                        dwell_state=dwell_state,
                    )
                # P4-E: tick the dwell state machine AFTER zone events
                # are processed for this frame so a freshly-emitted
                # dwell_started's interval-counter starts at the right
                # ``last_update_at``. Even when no zone events fire
                # this frame, the tick may emit dwell_updated /
                # dwell_exceeded for sessions that started on prior
                # frames.
                if dwell_state is not None:
                    await self._emit_dwell_tick(
                        frame=frame,
                        dwell_state=dwell_state,
                        source_block=source_block,
                        counters=counters,
                    )
                # P5-C: low-rate latest perception snapshot. Posted AFTER
                # the observation emits so a perception failure can never
                # disturb the observation path. At most once per interval;
                # the first qualifying frame posts immediately.
                if perception_enabled:
                    now_mono = time.monotonic()
                    # M7E: the effective interval is per-session — 1/fps when
                    # an operator processing_fps is applied, else the global
                    # default (see __init__).
                    if now_mono - last_perception_post >= self._perception_interval_seconds:
                        last_perception_post = now_mono
                        payload = self._build_perception_payload(
                            frame=frame,
                            geometry=perception_geometry,
                            counters=counters,
                        )
                        if payload is not None:
                            # Hand off to the background sender (non-blocking):
                            # the frame loop never waits on the upload. The
                            # slot coalesces — a snapshot already waiting is
                            # overwritten by this fresher one.
                            perception_slot["pending"] = payload
                            perception_wakeup.set()
                # P5-F: fanout hot path. Same snapshot builder, separate
                # coalescing slot, dynamic cadence: 1/hot_hz while the
                # sidecar reports watchers, 1/idle_hz otherwise (the
                # publisher owns that decision). Placed AFTER the
                # observation emits + cold-path snapshot so a fanout
                # problem can never disturb either.
                if fanout_publisher is not None:
                    now_mono = time.monotonic()
                    if now_mono - last_fanout_post >= fanout_publisher.interval_seconds:
                        last_fanout_post = now_mono
                        payload = self._build_perception_payload(
                            frame=frame,
                            geometry=perception_geometry,
                            counters=counters,
                        )
                        if payload is not None:
                            fanout_slot["pending"] = build_fanout_envelope(
                                camera_id=str(self._session.camera_id),
                                stream_session_id=str(payload.stream_session_id),
                                processor_id=payload.processor_id,
                                package_id=payload.package_id,
                                observed_at_iso=payload.observed_at.isoformat(),
                                snapshot=payload.snapshot,
                            )
                            fanout_wakeup.set()
                if not attached:
                    attached = True
                    self._state = WorkerState.ATTACHED
                    if not await self._post_gating("attached"):
                        return
                if not processing:
                    processing = True
                    self._state = WorkerState.PROCESSING
                    last_proc_post = time.monotonic()
                    await self._post(
                        "processing",
                        transient_ok=True,
                        last_frame_at=now_utc(),
                        payload=self._metadata_payload(
                            counters=counters,
                            queue_stats=_queue_stats_dict(queue),
                            last_app_summary=reader_state.last_summary,
                        ),
                    )
                    continue
                now = time.monotonic()
                if now - last_proc_post >= self._settings.status_interval_seconds:
                    last_proc_post = now
                    await self._post(
                        "processing",
                        transient_ok=True,
                        last_frame_at=now_utc(),
                        payload=self._metadata_payload(
                            counters=counters,
                            queue_stats=_queue_stats_dict(queue),
                            last_app_summary=reader_state.last_summary,
                        ),
                    )

        # Run reader + consumer concurrently. Cleanup is in a shielded
        # ``finally`` so docker stop + workdir cleanup + the terminal
        # status post happen even if the worker task is cancelled (e.g.,
        # supervisor force-stop after the per-worker budget expires).
        # The background perception sender runs alongside (not inside the
        # gather: it idles until the consumer ends, then is drained below).
        sender_task = asyncio.create_task(_perception_sender()) if perception_enabled else None
        fanout_task = (
            asyncio.create_task(_fanout_sender()) if fanout_publisher is not None else None
        )
        try:
            results = await asyncio.gather(_reader(), _consumer(), return_exceptions=True)
            for r in results:
                if isinstance(r, BaseException) and not isinstance(r, asyncio.CancelledError):
                    # Surface unexpected failure but keep cleanup going.
                    _log.exception("metadata-probe task crashed", exc_info=r)
        finally:
            # Flush the perception sender BEFORE finalize: signal stop + wake
            # it so it drains the final pending snapshot, then await (bounded)
            # so the terminal status post reflects the final perception
            # counters. A stuck upload is abandoned after the timeout so
            # teardown can never wedge.
            if sender_task is not None:
                perception_stop.set()
                perception_wakeup.set()
                try:
                    await asyncio.wait_for(
                        asyncio.shield(sender_task), timeout=_PERCEPTION_FLUSH_TIMEOUT
                    )
                except TimeoutError:
                    sender_task.cancel()
            # P5-F: same drain-then-close for the fanout sender; then close
            # the publisher's HTTP client. Best-effort end to end.
            if fanout_task is not None:
                fanout_stop.set()
                fanout_wakeup.set()
                try:
                    await asyncio.wait_for(
                        asyncio.shield(fanout_task), timeout=_PERCEPTION_FLUSH_TIMEOUT
                    )
                except TimeoutError:
                    fanout_task.cancel()
            if fanout_publisher is not None:
                with contextlib.suppress(Exception):
                    await fanout_publisher.aclose()
            try:
                await asyncio.shield(
                    self._finalize_metadata_probe(
                        queue=queue,
                        counters=counters,
                        reader_state=reader_state,
                    )
                )
            finally:
                self._metadata_queue = None

    # --- P5-C: low-rate latest perception snapshot -----------------------

    def _build_perception_payload(
        self,
        *,
        frame: NeutralFrameMetadata,
        geometry: dict[str, Any],
        counters: _MetadataCounters,
    ) -> PerceptionSnapshotPayload | None:
        """Build one latest-perception-snapshot payload — NO network.

        Cheap + local so it runs inline in the frame loop; the ~330ms upload
        round-trip is handed to the background sender
        (``_post_perception_payload``) so it never throttles frame
        consumption. A build failure (defensive builder edge case) is counted
        + swallowed (best-effort) and returns ``None``; it must NEVER crash
        the consumer loop or disturb the observation path.

        Unlike the observation emits this does NOT go through
        ``ObservationWriter`` — a perception snapshot is not an observation;
        it posts directly to the dedicated perception-snapshot endpoint.
        """
        observed_at = frame.frame.timestamp or now_utc()
        # M7D: apply the per-camera selected_classes by post-inference filtering
        # of the per-object snapshot. ``self._selected_classes`` is the resolved
        # APPLIED list ([] -> filter returns the objects unchanged, so this is
        # byte-identical to today when no classes are selected). Build the
        # snapshot from a COPY carrying the filtered objects — never mutate the
        # shared frame in place, since counters / line-crossing / ROI-occupancy
        # consume the same frame and must stay raw/aggregate.
        filtered_objects = filter_objects_by_class(frame.objects, self._selected_classes)
        snapshot_frame = (
            frame
            if filtered_objects is frame.objects
            else frame.model_copy(update={"objects": list(filtered_objects)})
        )
        try:
            snapshot = build_perception_snapshot(
                metadata=snapshot_frame,
                geometry=geometry,
                camera_id=self._session.camera_id,
                stream_session_id=self._session.session_id,
                processor_id=self._settings.processor_id,
                package_id=self._package.package_id,
                runtime="deepstream",
                metadata_source="pyds_pad_probe",
                analytics_plugin="nvdsanalytics",
                tracker_algorithm=self._effective_tracker_algorithm,
                observed_at=observed_at,
                max_objects=self._settings.perception_snapshot_max_objects,
                # M8: stamp the SAME requested-vs-applied lineage carried on
                # processor status (single source of truth) into the snapshot
                # source block. Additive metadata only — status_payload() is
                # pure/read-only, so no objects/geometry/count behavior changes.
                applied_processing_config=self._resolved_processing_config.status_payload(),
            )
        except Exception as exc:
            counters.perception_snapshot_write_errors += 1
            _log.warning(
                "perception snapshot build failed for %s: %s",
                self.session_id,
                exc,
            )
            return None
        return PerceptionSnapshotPayload(
            stream_session_id=self._session.session_id,
            processor_id=self._settings.processor_id,
            package_id=self._package.package_id,
            observed_at=observed_at,
            snapshot=snapshot,
        )

    async def _post_perception_payload(
        self,
        payload: PerceptionSnapshotPayload,
        counters: _MetadataCounters,
    ) -> None:
        """POST one perception snapshot — the network round-trip.

        Runs in the background sender task, NOT the frame loop, so the
        ~330ms round-trip never blocks frame consumption. Best-effort:
        ``ApiSessionGoneError`` propagates (the session is gone — the sender
        relays it to the loop to end the worker quietly); every other error
        (transient API, contract drift) is counted + swallowed so it can never
        disturb the observation path.
        """
        try:
            await self._api.post_perception_snapshot(payload)
        except ApiSessionGoneError:
            raise
        except Exception as exc:
            counters.perception_snapshot_write_errors += 1
            _log.warning(
                "perception snapshot post failed for %s: %s",
                self.session_id,
                exc,
            )
            return
        counters.perception_snapshots_posted += 1
        counters.last_perception_snapshot_at = payload.observed_at

    # --- P4-B: per-frame line-crossing emission --------------------------

    async def _emit_line_crossings(
        self,
        *,
        frame: NeutralFrameMetadata,
        line_state: LineCrossingState,
        source_block: dict[str, Any],
        counters: _MetadataCounters,
    ) -> None:
        """Pull the analytics frame-level ``line_crossing_cumulative``
        block from one consumed frame, hand it to the adapter state, and
        emit one ``processor.line_crossing`` observation per increment.

        Caller has already gated on ``self._mode == _ANALYTICS_MODE`` and
        ``self._writer is not None`` (line_state is only built in that
        case), so this method assumes both. Transient write errors are
        absorbed (mirroring ``_post``'s ``transient_ok=True`` shape).
        ``ApiSessionGoneError`` propagates — same semantics as elsewhere
        in the worker: the session is gone, end quietly via the outer
        handler. ``ApiValidationError`` is logged + counted but does NOT
        crash the consumer (a wire-contract drift between processor and
        api-server is unrecoverable in flight; the next worker spawn
        will use a fresh schema)."""
        assert self._writer is not None  # gated by the caller
        # M7D: intentionally NOT class-filtered. These are nvdsanalytics
        # AGGREGATE per-line cumulative counts across ALL classes, computed
        # in-container; they are not per-class filterable post-inference (out of
        # M7D scope — see the selected_classes_analytics_counts_not_filtered
        # lineage warning).
        analytics = frame.source.get("analytics") if isinstance(frame.source, dict) else None
        if not isinstance(analytics, dict):
            return
        cumulative_by_line = analytics.get("line_crossing_cumulative")
        if not isinstance(cumulative_by_line, dict) or not cumulative_by_line:
            return
        occurred_at = frame.frame.timestamp or now_utc()
        # Coerce to ``dict[str, int]`` defensively — the producer should
        # already match this, but the validation gate is permissive about
        # the inner ``dict[str, Any]`` shape on ``frame.source``.
        cumulative_clean: dict[str, int] = {
            str(k): int(v) for k, v in cumulative_by_line.items() if isinstance(v, int | float)
        }
        if not cumulative_clean:
            return
        events = line_state.absorb_cumulative(
            cumulative_by_line=cumulative_clean,
            occurred_at=occurred_at,
            source=source_block,
            debug={
                "frame_num": int(frame.frame.frame_num),
                "geometry_source": "probe_default",
                "raw_line_crossing_cumulative": dict(cumulative_clean),
            },
        )
        for event in events:
            try:
                await self._writer.emit(
                    stream_session_id=self._session.session_id,
                    obs_type="processor.line_crossing",
                    payload=event.to_payload(),
                    package_id=self._package.package_id,
                    observed_at=event.occurred_at,
                )
            except ApiSessionGoneError:
                # Outer ``run()`` handler ends the worker quietly.
                raise
            except ApiValidationError as exc:
                counters.observation_write_errors += 1
                _log.error(
                    "wire contract drift on observation emit for %s: %s",
                    self.session_id,
                    exc,
                )
                continue
            except TRANSIENT_API_ERRORS as exc:
                counters.observation_write_errors += 1
                _log.warning(
                    "transient observation emit for %s: %s",
                    self.session_id,
                    exc,
                )
                continue
            counters.observations_emitted += 1
            counters.line_crossing_observations_emitted += 1
            counters.last_observation_at = event.occurred_at

    # --- side task: per-object line crossing emission ---------------------

    async def _emit_line_crossing_objects(
        self,
        *,
        frame: NeutralFrameMetadata,
        line_object_state: LineCrossingObjectState,
        source_block: dict[str, Any],
        counters: _MetadataCounters,
    ) -> None:
        """Per-OBJECT line-crossing evidence, emitted alongside the
        aggregate cumulative events (``_emit_line_crossings``) for the
        same frame. Only tracked objects participate — an untracked
        object's crossing is still counted by the aggregate path, just
        never attributed. ``known_lines`` is the frame's cumulative
        counter keys plus the object's own labels, so a label that
        clears re-arms the (track, line) pair (rising-edge dedupe — see
        ``LineCrossingObjectState``). Error handling mirrors
        ``_emit_line_crossings`` exactly."""
        assert self._writer is not None  # gated by the caller
        analytics = frame.source.get("analytics") if isinstance(frame.source, dict) else None
        cumulative = (
            analytics.get("line_crossing_cumulative") if isinstance(analytics, dict) else None
        )
        known_lines: set[str] = (
            {str(k) for k in cumulative} if isinstance(cumulative, dict) else set()
        )
        occurred_at = frame.frame.timestamp or now_utc()
        for obj in frame.objects:
            if obj.tracker_id is None:
                continue
            obj_analytics = obj.debug.get("analytics") if isinstance(obj.debug, dict) else None
            if not isinstance(obj_analytics, dict):
                continue
            raw_labels = obj_analytics.get("line_labels")
            line_labels: set[str] = (
                {str(label) for label in raw_labels if isinstance(label, str)}
                if isinstance(raw_labels, list)
                else set()
            )
            if not line_labels and not known_lines:
                # No crossing labels and no line universe — nothing to
                # emit and nothing to re-arm for this sighting.
                continue
            direction_raw = obj_analytics.get("direction")
            events = line_object_state.absorb_object(
                track_id=obj.tracker_id,
                line_labels=line_labels,
                known_lines=known_lines,
                occurred_at=occurred_at,
                class_id=obj.class_id,
                class_label=obj.label,
                confidence=obj.confidence,
                analytics_direction_label=(
                    direction_raw if isinstance(direction_raw, str) and direction_raw else None
                ),
                source=source_block,
                debug={
                    "frame_num": int(frame.frame.frame_num),
                    "geometry_source": "probe_default",
                },
            )
            for event in events:
                try:
                    await self._writer.emit(
                        stream_session_id=self._session.session_id,
                        obs_type="processor.line_crossing_object",
                        payload=event.to_payload(),
                        package_id=self._package.package_id,
                        observed_at=event.occurred_at,
                    )
                except ApiSessionGoneError:
                    raise
                except ApiValidationError as exc:
                    counters.observation_write_errors += 1
                    _log.error(
                        "wire contract drift on line_crossing_object emit for %s: %s",
                        self.session_id,
                        exc,
                    )
                    continue
                except TRANSIENT_API_ERRORS as exc:
                    counters.observation_write_errors += 1
                    _log.warning(
                        "transient line_crossing_object emit for %s: %s",
                        self.session_id,
                        exc,
                    )
                    continue
                counters.observations_emitted += 1
                counters.line_crossing_object_observations_emitted += 1
                counters.last_observation_at = event.occurred_at

    # --- P4-C: per-frame ROI occupancy emission --------------------------

    async def _emit_roi_occupancies(
        self,
        *,
        frame: NeutralFrameMetadata,
        roi_state: RoiOccupancyState,
        source_block: dict[str, Any],
        counters: _MetadataCounters,
    ) -> None:
        """Pull the analytics frame-level ``roi_counts`` block from one
        consumed frame, hand it to the ROI adapter state, and emit one
        ``processor.roi_occupancy`` observation per state-change /
        first-seen / interval heartbeat.

        Caller has already gated on ``self._mode == _ANALYTICS_MODE`` and
        ``self._writer is not None`` (roi_state is only built in that
        case), so this method assumes both. Error handling mirrors
        ``_emit_line_crossings`` exactly: transient writer errors are
        absorbed, ``ApiSessionGoneError`` propagates so the outer
        ``run()`` handler ends quietly, ``ApiValidationError`` is logged
        + counted (wire-contract drift; do not crash). The deterministic
        ``idempotency_key`` on every event makes a future server-side
        UNIQUE constraint safe to add."""
        assert self._writer is not None  # gated by the caller
        # ``total_count`` stays the nvdsanalytics AGGREGATE per-ROI occupancy
        # counter (cross-class, computed in-container — the M7D posture).
        # Side task (workflow condition payload enrichment): ``class_counts``
        # is now populated from a SECOND, object-attributed source — each
        # object's per-object ``analytics.roi_labels`` x its class label.
        # Only labeled objects with analytics metadata are counted, so the
        # breakdown can legitimately sum below ``total_count`` (unlabeled or
        # metadata-less objects); it is never padded to match. When no object
        # carries usable metadata the breakdown is omitted entirely
        # (empty ``class_counts``), never guessed.
        analytics = frame.source.get("analytics") if isinstance(frame.source, dict) else None
        if not isinstance(analytics, dict):
            return
        roi_counts = analytics.get("roi_counts")
        if not isinstance(roi_counts, dict) or not roi_counts:
            return
        occurred_at = frame.frame.timestamp or now_utc()
        # Coerce to ``dict[str, int]`` defensively — the producer should
        # already match this shape, but ``frame.source`` is permissive
        # about inner ``Any`` types.
        roi_counts_clean: dict[str, int] = {
            str(k): int(v) for k, v in roi_counts.items() if isinstance(v, int | float)
        }
        if not roi_counts_clean:
            return
        class_counts_by_roi: dict[str, dict[str, int]] = {}
        for obj in frame.objects:
            if not obj.label:
                continue
            obj_analytics = obj.debug.get("analytics") if isinstance(obj.debug, dict) else None
            if not isinstance(obj_analytics, dict):
                continue
            raw_labels = obj_analytics.get("roi_labels")
            if not isinstance(raw_labels, list):
                continue
            for roi_label in raw_labels:
                if not isinstance(roi_label, str):
                    continue
                per_roi = class_counts_by_roi.setdefault(roi_label, {})
                per_roi[obj.label] = per_roi.get(obj.label, 0) + 1
        events = roi_state.absorb_roi_counts(
            roi_counts=roi_counts_clean,
            class_counts_by_roi=class_counts_by_roi,
            occurred_at=occurred_at,
            source=source_block,
            debug={
                "frame_num": int(frame.frame.frame_num),
                "geometry_source": "probe_default",
                "raw_roi_counts": dict(roi_counts_clean),
            },
        )
        for event in events:
            try:
                await self._writer.emit(
                    stream_session_id=self._session.session_id,
                    obs_type="processor.roi_occupancy",
                    payload=event.to_payload(),
                    package_id=self._package.package_id,
                    observed_at=event.occurred_at,
                )
            except ApiSessionGoneError:
                # Outer ``run()`` handler ends the worker quietly.
                raise
            except ApiValidationError as exc:
                counters.observation_write_errors += 1
                _log.error(
                    "wire contract drift on ROI observation emit for %s: %s",
                    self.session_id,
                    exc,
                )
                continue
            except TRANSIENT_API_ERRORS as exc:
                counters.observation_write_errors += 1
                _log.warning(
                    "transient ROI observation emit for %s: %s",
                    self.session_id,
                    exc,
                )
                continue
            counters.observations_emitted += 1
            counters.roi_occupancy_observations_emitted += 1
            counters.last_observation_at = event.occurred_at

    # --- P4-D: per-track zone transition emission ------------------------

    async def _emit_zone_transitions(
        self,
        *,
        frame: NeutralFrameMetadata,
        zone_state: ZoneMembershipState,
        source_block: dict[str, Any],
        counters: _MetadataCounters,
        dwell_state: DwellState | None = None,
    ) -> None:
        """Walk each tracked object in the frame, hand its
        ``roi_labels`` to ``ZoneMembershipState`` along with the set of
        known zones (= keys of ``frame.source.analytics.roi_counts``),
        and emit one ``processor.zone_entered`` / ``processor.zone_exited``
        observation per committed transition.

        Skips objects without a ``tracker_id`` (zone transitions are
        meaningless without a stable identity) and objects whose
        ``debug.analytics.roi_labels`` is missing or malformed. Bails
        early if the frame has no ``roi_counts`` (no zones configured
        for this frame). Error handling mirrors
        ``_emit_roi_occupancies`` exactly: ``ApiSessionGoneError``
        propagates so the outer ``run()`` ends quietly;
        ``ApiValidationError`` / ``TRANSIENT_API_ERRORS`` are logged
        + counted + continue."""
        assert self._writer is not None  # gated by the caller
        analytics = frame.source.get("analytics") if isinstance(frame.source, dict) else None
        if not isinstance(analytics, dict):
            return
        roi_counts = analytics.get("roi_counts")
        if not isinstance(roi_counts, dict) or not roi_counts:
            return
        zone_set: set[str] = {str(k) for k in roi_counts}
        occurred_at = frame.frame.timestamp or now_utc()
        # M7D: apply the per-camera selected_classes to this PER-OBJECT path
        # (dwell follows zone automatically). [] -> filter returns frame.objects
        # unchanged (byte-identical to today). Track continuity is preserved
        # because an object's class is stable across its track lifetime.
        objects = filter_objects_by_class(frame.objects, self._selected_classes)
        for obj in objects:
            if obj.tracker_id is None:
                continue
            obj_analytics = obj.debug.get("analytics") if isinstance(obj.debug, dict) else None
            if not isinstance(obj_analytics, dict):
                continue
            raw_labels = obj_analytics.get("roi_labels")
            if not isinstance(raw_labels, list):
                continue
            roi_labels: set[str] = {str(label) for label in raw_labels if isinstance(label, str)}
            events = zone_state.absorb_object(
                track_id=obj.tracker_id,
                zone_set=zone_set,
                roi_labels=roi_labels,
                occurred_at=occurred_at,
                class_id=obj.class_id,
                class_label=obj.label,
                confidence=obj.confidence,
                source=source_block,
                debug={
                    "frame_num": int(frame.frame.frame_num),
                    "geometry_source": "probe_default",
                    "roi_labels": sorted(roi_labels),
                },
            )
            for event in events:
                try:
                    await self._writer.emit(
                        stream_session_id=self._session.session_id,
                        obs_type=event.type,  # type: ignore[arg-type]
                        payload=event.to_payload(),
                        package_id=self._package.package_id,
                        observed_at=event.occurred_at,
                    )
                except ApiSessionGoneError:
                    raise
                except ApiValidationError as exc:
                    counters.observation_write_errors += 1
                    _log.error(
                        "wire contract drift on zone transition emit for %s: %s",
                        self.session_id,
                        exc,
                    )
                    continue
                except TRANSIENT_API_ERRORS as exc:
                    counters.observation_write_errors += 1
                    _log.warning(
                        "transient zone transition emit for %s: %s",
                        self.session_id,
                        exc,
                    )
                    continue
                counters.observations_emitted += 1
                if event.type == "processor.zone_entered":
                    counters.zone_entered_observations_emitted += 1
                else:
                    counters.zone_exited_observations_emitted += 1
                counters.last_observation_at = event.occurred_at
                # P4-E: feed the zone event into DwellState. For
                # zone_entered this produces a dwell_started event we
                # emit immediately; for zone_exited it silently closes
                # the dwell session.
                if dwell_state is not None:
                    dwell_events = dwell_state.absorb_zone_event(
                        event,
                        source=source_block,
                        debug={
                            "frame_num": int(frame.frame.frame_num),
                            "geometry_source": "probe_default",
                        },
                    )
                    for dev in dwell_events:
                        await self._emit_one_dwell_event(dev, counters=counters)

    async def _emit_dwell_tick(
        self,
        *,
        frame: NeutralFrameMetadata,
        dwell_state: DwellState,
        source_block: dict[str, Any],
        counters: _MetadataCounters,
    ) -> None:
        """Per-frame dwell tick: emit periodic ``dwell_updated`` events
        and the single-shot ``dwell_exceeded`` event for any active
        session whose timers have elapsed.

        Bails early if there are no active dwell sessions (most
        frames). ``occurred_at`` is taken from the frame's timestamp
        when available so unit tests can drive deterministic clock
        progression; live frames fall back to ``now_utc()``."""
        assert self._writer is not None  # gated by the caller
        if dwell_state.active_session_count() == 0:
            return
        occurred_at = frame.frame.timestamp or now_utc()
        # Side task (workflow condition payload enrichment): the CURRENT
        # frame's per-track detection confidence. Tracks not detected this
        # frame are simply absent — their dwell events omit confidence
        # rather than carrying a stale value.
        confidence_by_track: dict[str, float] = {
            obj.tracker_id: float(obj.confidence)
            for obj in frame.objects
            if obj.tracker_id is not None and obj.confidence is not None
        }
        events = dwell_state.tick(
            occurred_at=occurred_at,
            confidence_by_track=confidence_by_track,
            source=source_block,
            debug={
                "frame_num": int(frame.frame.frame_num),
                "geometry_source": "probe_default",
            },
        )
        for event in events:
            await self._emit_one_dwell_event(event, counters=counters)

    async def _emit_one_dwell_event(
        self,
        event: Any,  # DwellEvent — typed loosely to avoid top-level forward ref
        *,
        counters: _MetadataCounters,
    ) -> None:
        """Single-event emit path used by both the zone-event-driven
        path (dwell_started) and the per-frame tick path
        (dwell_updated / dwell_exceeded). Error handling mirrors the
        other emit helpers exactly."""
        assert self._writer is not None
        try:
            await self._writer.emit(
                stream_session_id=self._session.session_id,
                obs_type=event.type,
                payload=event.to_payload(),
                package_id=self._package.package_id,
                observed_at=event.occurred_at,
            )
        except ApiSessionGoneError:
            raise
        except ApiValidationError as exc:
            counters.observation_write_errors += 1
            counters.dwell_write_errors += 1
            _log.error(
                "wire contract drift on dwell emit (%s) for %s: %s",
                event.type,
                self.session_id,
                exc,
            )
            return
        except TRANSIENT_API_ERRORS as exc:
            counters.observation_write_errors += 1
            counters.dwell_write_errors += 1
            _log.warning(
                "transient dwell emit (%s) for %s: %s",
                event.type,
                self.session_id,
                exc,
            )
            return
        counters.observations_emitted += 1
        if event.type == "processor.dwell_started":
            counters.dwell_started_observations_emitted += 1
        elif event.type == "processor.dwell_updated":
            counters.dwell_updated_observations_emitted += 1
        elif event.type == "processor.dwell_exceeded":
            counters.dwell_exceeded_observations_emitted += 1
        counters.last_observation_at = event.occurred_at

    # --- helpers ----------------------------------------------------------

    async def _safe_stop_pm(self) -> None:
        if self._pm is None:
            return
        try:
            await self._pm.stop()
        except Exception as exc:  # stop must never raise out of teardown
            _log.warning("error stopping decode process: %s", exc)

    def _build(self, status: str, **fields: Any) -> Any:
        # P6-C: every status post from an analytics worker carries the
        # geometry lineage block in ``payload.applied_geometry``. On invalid
        # operator-saved geometry the worker also surfaces
        # ``error_code="geometry_invalid"`` — but only when the caller didn't
        # set a more-specific error (container crash, model load failure,
        # etc.), so runtime errors keep their own taxonomy.
        #
        # M7B: every status post also carries the processing-config lineage
        # block in ``payload.applied_processing_config`` (requested-vs-applied,
        # STAMP ONLY). Unlike geometry, a bad/diverging processing config does
        # NOT set ``error_code`` — M7B applies nothing, so it must not flag a
        # worker error; it lives only in the lineage block.
        if self._resolved_geometry is not None or self._resolved_processing_config is not None:
            payload = dict(fields.pop("payload", None) or {})
            if self._resolved_geometry is not None:
                payload.setdefault("applied_geometry", self._resolved_geometry.status_payload())
            payload.setdefault(
                "applied_processing_config",
                self._resolved_processing_config.status_payload(),
            )
            fields["payload"] = payload
            if (
                self._resolved_geometry is not None
                and self._resolved_geometry.source == "fallback_default_on_error"
            ):
                fields.setdefault("error_code", "geometry_invalid")
                if self._resolved_geometry.error_reason:
                    fields.setdefault("error_message", self._resolved_geometry.error_reason)
        return build_status_request(
            session=self._session,
            processor_id=self._settings.processor_id,
            status=status,  # type: ignore[arg-type]  # caller passes a valid literal
            runtime=self._package.runtime,
            package_id=self._package.package_id,
            **fields,
        )

    async def _post(self, status: str, *, transient_ok: bool = False, **fields: Any) -> None:
        try:
            await self._api.post_processor_status(self._build(status, **fields))
            self._backoff.reset()
        except TRANSIENT_API_ERRORS as exc:
            if not transient_ok:
                raise
            _log.warning(
                "transient posting status=%s for %s: %s",
                status,
                self.session_id,
                exc,
            )

    async def _post_gating(self, status: str, **fields: Any) -> bool:
        while not self._stop.is_set():
            try:
                await self._api.post_processor_status(self._build(status, **fields))
                self._backoff.reset()
                return True
            except TRANSIENT_API_ERRORS as exc:
                _log.warning(
                    "transient posting status=%s for %s: %s",
                    status,
                    self.session_id,
                    exc,
                )
                if await wait_or_event(self._stop, self._backoff.next_delay()):
                    return False
        return False

    async def _post_best_effort(
        self,
        status: str,
        *,
        fps_processed: float | None = None,
        error_code: str | None = None,
        error_message: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> None:
        for _ in range(_TERMINAL_STATUS_ATTEMPTS):
            try:
                await self._api.post_processor_status(
                    self._build(
                        status,
                        fps_processed=fps_processed,
                        error_code=error_code,
                        error_message=error_message,
                        payload=payload,
                    )
                )
                return
            except ApiSessionGoneError:
                return
            except TRANSIENT_API_ERRORS as exc:
                _log.info(
                    "transient terminal status=%s for %s: %s",
                    status,
                    self.session_id,
                    exc,
                )
                await asyncio.sleep(0.1)
            except Exception as exc:  # terminal post must not raise
                _log.warning(
                    "giving up terminal status=%s for %s: %s",
                    status,
                    self.session_id,
                    exc,
                )
                return
