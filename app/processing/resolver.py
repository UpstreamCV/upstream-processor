"""Resolve the eligible-session ``processing_config`` envelope into a
requested-vs-applied lineage block.

Three resolution outcomes (mirrors ``geometry/resolver.py``):

- ``operator_saved`` — envelope parsed. The requested values are recorded
  and compared field-by-field against what the runtime *actually* applies
  (the static defaults below). Any divergence that is NOT applied is recorded
  in ``warnings``.
- ``default`` — envelope is ``None`` (camera has no ``camera_processing_configs``
  row). ``status`` is ``applied_defaults`` with all-null requested.
- ``fallback_default_on_error`` — envelope is present but doesn't parse.
  ``error_reason`` carries a short, secret-free hint.

M7C: ``confidence_threshold`` is now genuinely applied — but ONLY when the
caller passes ``confidence_apply_supported=True`` (the DeepStream
metadata/tracker/analytics modes, which actually render the value into the
PGIE ``pre-cluster-threshold``). When applied, it is echoed into ``applied``
and produces NO warning; when the mode cannot apply it, it stays in the
defaults and emits ``confidence_threshold_requested_not_applied``. Because
both ``partially_applied`` (some applied, some not) and ``applied`` (request
matches/applies cleanly) can now occur, both statuses are emitted here.

M7D: ``selected_classes`` is now genuinely applied — but ONLY when the caller
passes ``selected_classes_apply_supported=True`` (the same per-object modes).
It is applied by POST-INFERENCE filtering of the PER-OBJECT output paths (the
perception snapshot + the zone/dwell observation paths); the detector still
runs across all classes. When applied, the operator list is echoed into
``applied.selected_classes`` and the honest warning
``selected_classes_analytics_counts_not_filtered`` is added — because the
nvdsanalytics AGGREGATE counts (line-crossing / ROI-occupancy) remain across
ALL classes and are intentionally NOT class-filtered (out of M7D scope). That
warning means a selected-classes-applied config is always at LEAST
``partially_applied``. When the mode cannot apply it (or no support), the list
stays at the defaults (``[]``) and emits ``selected_classes_requested_not_applied``
exactly as before.

M7E: ``processing_fps`` is now genuinely applied — but ONLY when the caller
passes ``processing_fps_apply_supported=True`` (the analytics mode with
perception snapshots enabled, which actually derives its snapshot cadence from
the value). It is applied as an OUTPUT-CADENCE throttle: the perception
snapshot emit interval becomes ``1/fps``. The DeepStream pipeline keeps
receiving and inferring frames at the source rate (no nvinfer interval /
GStreamer / pipeline change), and the line/zone/dwell observations are
event-driven (throttling them would drop real events) so they are NOT
throttled. That honesty rides as the always-on caveat
``processing_fps_output_cadence_only`` whenever fps is applied — so, like an
applied selected_classes, an fps-applied config is always at LEAST
``partially_applied``. When the mode cannot apply it, the request stays at the
defaults and emits ``processing_fps_requested_not_applied`` exactly as before.

Everything ELSE remains STAMP ONLY: ``model_key`` / ``tracker`` are never
applied — any meaningful request for them is recorded as a
``*_requested_not_applied`` warning and ``applied`` keeps the worker's current
runtime defaults.

Unlike the geometry resolver, a bad/diverging processing config does NOT flag
a worker ``error_code``: it lives only in this lineage block (the caller wires
it as ``payload.applied_processing_config`` and sets no error).

Pure / side-effect free: no I/O, no logging. Logging is the caller's job.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Literal

from pydantic import ValidationError

from app.processing.session_processing import SessionProcessingConfigEnvelope

ResolvedProcessingConfigSource = Literal["operator_saved", "default", "fallback_default_on_error"]


@dataclass(frozen=True)
class RequestedProcessingConfig:
    """What the operator saved (or all-null when there is no/invalid config)."""

    model_key: str | None
    selected_classes: list[str] | None
    tracker: str | None
    confidence_threshold: float | None
    processing_fps: int | None


@dataclass(frozen=True)
class AppliedProcessingConfig:
    """What the runtime actually applies. Starts from the static DeepStream
    defaults; M7C may override ``confidence_threshold``, M7D
    ``selected_classes``, and M7E ``processing_fps`` (output cadence) with the
    operator value when the mode can apply them. ``model_key`` / ``tracker``
    stay at defaults."""

    model_key: str
    selected_classes: list[str]
    tracker: str | None
    confidence_threshold: float | None
    processing_fps: int | None


# The deepstream worker's REAL current default behavior: trafficcamnet, all
# classes (empty list == no class filter), iou tracker, no operator
# threshold/fps override. M7C may replace ``confidence_threshold`` here with
# the operator value when the mode can apply it; everything else is reported
# as-is regardless of what was requested.
DEFAULT_APPLIED = AppliedProcessingConfig(
    model_key="trafficcamnet_v1",
    selected_classes=[],
    tracker="iou",
    confidence_threshold=None,
    processing_fps=None,
)

# Lineage-only static map: which runtime a model_key would route to. Used
# only to flag a requested model that has no known runtime mapping; it does
# NOT influence any applied value (M7B applies nothing).
MODEL_KEY_RUNTIME = {
    "trafficcamnet_v1": "deepstream",
    "mock_detect_v0": "mock",
}


@dataclass(frozen=True)
class ResolvedProcessingConfig:
    """The requested-vs-applied lineage the worker stamps onto status."""

    source: ResolvedProcessingConfigSource
    revision: int | None
    requested: RequestedProcessingConfig
    applied: AppliedProcessingConfig
    status: str
    warnings: tuple[str, ...] = field(default_factory=tuple)
    # Populated only when ``source == "fallback_default_on_error"``. Plain
    # English, no PII / no raw envelope contents — safe to log and surface.
    error_reason: str | None = None

    def status_payload(self) -> dict[str, Any]:
        """Shape injected into ``ProcessorStatusPayload.payload`` for lineage."""
        return {
            "source": self.source,
            "revision": self.revision,
            "status": self.status,
            "reason": self.error_reason,
            "requested": {
                "model_key": self.requested.model_key,
                "selected_classes": self.requested.selected_classes,
                "tracker": self.requested.tracker,
                "confidence_threshold": self.requested.confidence_threshold,
                "processing_fps": self.requested.processing_fps,
            },
            "applied": {
                "model_key": self.applied.model_key,
                "selected_classes": list(self.applied.selected_classes),
                "tracker": self.applied.tracker,
                "confidence_threshold": self.applied.confidence_threshold,
                "processing_fps": self.applied.processing_fps,
            },
            "warnings": list(self.warnings),
        }


_NULL_REQUESTED = RequestedProcessingConfig(
    model_key=None,
    selected_classes=None,
    tracker=None,
    confidence_threshold=None,
    processing_fps=None,
)


def resolve_session_processing_config(
    envelope: dict[str, Any] | None,
    *,
    defaults: AppliedProcessingConfig = DEFAULT_APPLIED,
    confidence_apply_supported: bool = False,
    selected_classes_apply_supported: bool = False,
    processing_fps_apply_supported: bool = False,
) -> ResolvedProcessingConfig:
    """Resolve the wire envelope into a ``ResolvedProcessingConfig``.

    ``confidence_apply_supported`` (default ``False``, which preserves the
    pre-M7C stamp-only behavior for callers/tests that don't pass it) tells
    the resolver the running mode can actually write the operator
    ``confidence_threshold`` into the PGIE config. When ``True`` and a
    threshold is requested, it is applied (echoed into ``applied`` with no
    warning); otherwise it is recorded as
    ``confidence_threshold_requested_not_applied``.

    ``selected_classes_apply_supported`` (default ``False``, which preserves
    the M7B/M7C behavior for callers/tests that don't pass it) tells the
    resolver the running mode can apply the operator ``selected_classes`` by
    POST-INFERENCE filtering of the per-object output paths. When ``True`` and
    a non-empty list is requested, it is applied (echoed into
    ``applied.selected_classes``) and the honest
    ``selected_classes_analytics_counts_not_filtered`` warning is added
    (aggregate line/ROI counts stay across all classes — out of scope);
    otherwise a non-empty request is recorded as
    ``selected_classes_requested_not_applied``.

    ``processing_fps_apply_supported`` (default ``False``, which preserves the
    pre-M7E stamp-only behavior for callers/tests that don't pass it) tells
    the resolver the running mode actually derives its perception-snapshot
    output cadence from the operator ``processing_fps``. When ``True`` and a
    valid fps is requested, it is applied (echoed into
    ``applied.processing_fps``) with the honest always-on caveat
    ``processing_fps_output_cadence_only`` (the throttle changes the snapshot
    emit cadence only — inference keeps running at the source rate and the
    event-driven observations are not throttled); otherwise a request is
    recorded as ``processing_fps_requested_not_applied``.

    See module docstring. NEVER raises.
    """
    if envelope is None:
        return ResolvedProcessingConfig(
            source="default",
            revision=None,
            requested=_NULL_REQUESTED,
            applied=defaults,
            status="applied_defaults",
            warnings=(),
        )
    try:
        parsed = SessionProcessingConfigEnvelope.model_validate(envelope)
    except ValidationError as exc:
        # Surface a short, secret-free hint without leaking the raw envelope.
        errors = exc.errors()
        if errors:
            first = errors[0]
            loc = ".".join(str(p) for p in first.get("loc", ()))
            msg = first.get("msg", "validation failed")
        else:
            loc = ""
            msg = "validation failed"
        reason = (
            f"invalid processing config envelope: {loc}: {msg}"
            if loc
            else f"invalid processing config envelope: {msg}"
        )
        return ResolvedProcessingConfig(
            source="fallback_default_on_error",
            revision=None,
            requested=_NULL_REQUESTED,
            applied=defaults,
            status="fallback_default_on_error",
            warnings=(),
            error_reason=reason[:200],
        )

    config = parsed.config
    requested = RequestedProcessingConfig(
        model_key=config.model_key,
        selected_classes=list(config.selected_classes),
        tracker=config.tracker,
        confidence_threshold=config.confidence_threshold,
        processing_fps=config.processing_fps,
    )

    # Per-field divergence vs what the runtime actually applies (defaults).
    # Everything except confidence_threshold (M7C), selected_classes (M7D)
    # and processing_fps (M7E) is still stamp-only: a meaningful request is
    # recorded as "*_requested_not_applied" and never applied.
    warnings: list[str] = []
    applied_count = 0

    # M7E: processing_fps is genuinely applied (perception-snapshot output
    # cadence = 1/fps) when the mode supports it. An applied fps ALWAYS also
    # adds the honest output-cadence caveat — inference keeps running at the
    # source rate and event-driven observations are not throttled — so an
    # fps-applied config is always at least partially_applied. The envelope
    # schema bounds fps to 1..30; the > 0 check is defensive belt-and-braces
    # so a future schema relaxation can never produce a zero/negative cadence.
    applied_fps = defaults.processing_fps
    if config.processing_fps is not None:
        if processing_fps_apply_supported and config.processing_fps > 0:
            applied_fps = config.processing_fps
            applied_count += 1
            warnings.append("processing_fps_output_cadence_only")
        else:
            warnings.append("processing_fps_requested_not_applied")
    if config.model_key != defaults.model_key:
        warnings.append("model_switch_requested_not_applied")
        if config.model_key not in MODEL_KEY_RUNTIME:
            warnings.append("model_key_no_runtime_mapping")
    if config.tracker not in (None, defaults.tracker):
        warnings.append("tracker_requested_not_applied")

    # M7C: confidence_threshold is genuinely applied when the mode supports it.
    applied_confidence = defaults.confidence_threshold
    if config.confidence_threshold is not None:
        if confidence_apply_supported:
            applied_confidence = config.confidence_threshold
            applied_count += 1
        else:
            warnings.append("confidence_threshold_requested_not_applied")

    # M7D: selected_classes is genuinely applied (post-inference per-object
    # filtering) when the mode supports it. An applied list ALWAYS also adds the
    # honest analytics-counts warning, because the nvdsanalytics aggregate
    # line/ROI counts remain across all classes (out of M7D scope) — so a
    # selected-classes-applied config is always at least partially_applied.
    applied_selected = list(defaults.selected_classes)
    if config.selected_classes:
        if selected_classes_apply_supported:
            applied_selected = list(config.selected_classes)
            applied_count += 1
            warnings.append("selected_classes_analytics_counts_not_filtered")
        else:
            warnings.append("selected_classes_requested_not_applied")

    applied = replace(
        defaults,
        confidence_threshold=applied_confidence,
        selected_classes=applied_selected,
        processing_fps=applied_fps,
    )

    if warnings and applied_count > 0:
        status = "partially_applied"
    elif warnings:
        status = "requested_not_applied"
    elif applied_count > 0:
        status = "applied"
    else:
        status = "applied_defaults"
    return ResolvedProcessingConfig(
        source="operator_saved",
        revision=parsed.revision,
        requested=requested,
        applied=applied,
        status=status,
        warnings=tuple(warnings),
    )
