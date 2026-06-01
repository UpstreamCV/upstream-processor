"""Resolve the eligible-session ``processing_config`` envelope into a
requested-vs-applied lineage block. STAMP ONLY (M7B).

Three resolution outcomes (mirrors ``geometry/resolver.py``):

- ``operator_saved`` — envelope parsed. The requested values are recorded
  and compared field-by-field against what the runtime *actually* applies
  (the static defaults below). Any divergence is recorded in ``warnings``
  and flips ``status`` to ``requested_not_applied``; nothing is applied.
- ``default`` — envelope is ``None`` (camera has no ``camera_processing_configs``
  row). ``status`` is ``applied_defaults`` with all-null requested.
- ``fallback_default_on_error`` — envelope is present but doesn't parse.
  ``error_reason`` carries a short, secret-free hint.

M7B applies NOTHING from the config — ``applied`` is ALWAYS the worker's
current runtime defaults and never echoes a requested value. Unlike the
geometry resolver, a bad/diverging processing config does NOT flag a worker
``error_code``: it lives only in this lineage block (the caller wires it as
``payload.applied_processing_config`` and sets no error). ``partially_applied``
is reserved for a later milestone and is NEVER emitted here.

Pure / side-effect free: no I/O, no logging. Logging is the caller's job.
"""

from __future__ import annotations

from dataclasses import dataclass, field
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
    """What the runtime actually applies. In M7B this is always the static
    DeepStream defaults — the config is never applied."""

    model_key: str
    selected_classes: list[str]
    tracker: str | None
    confidence_threshold: float | None
    processing_fps: int | None


# The deepstream worker's REAL current behavior: trafficcamnet, all classes
# (empty list == no class filter), iou tracker, no operator threshold/fps
# override. M7B applies nothing, so this is what every ``applied`` block
# reports regardless of what was requested.
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
) -> ResolvedProcessingConfig:
    """Resolve the wire envelope into a ``ResolvedProcessingConfig``.

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
    # M7B applies nothing, so any meaningful request is "requested_not_applied".
    warnings: list[str] = []
    if config.selected_classes and config.selected_classes != []:
        warnings.append("selected_classes_requested_not_applied")
    if config.confidence_threshold is not None:
        warnings.append("confidence_threshold_requested_not_applied")
    if config.processing_fps is not None:
        warnings.append("processing_fps_requested_not_applied")
    if config.model_key != defaults.model_key:
        warnings.append("model_switch_requested_not_applied")
        if config.model_key not in MODEL_KEY_RUNTIME:
            warnings.append("model_key_no_runtime_mapping")
    if config.tracker not in (None, defaults.tracker):
        warnings.append("tracker_requested_not_applied")

    # NEVER "partially_applied" in M7B — only requested_not_applied (something
    # diverges) or applied_defaults (request matches the defaults exactly).
    status = "requested_not_applied" if warnings else "applied_defaults"
    return ResolvedProcessingConfig(
        source="operator_saved",
        revision=parsed.revision,
        requested=requested,
        applied=defaults,
        status=status,
        warnings=tuple(warnings),
    )
