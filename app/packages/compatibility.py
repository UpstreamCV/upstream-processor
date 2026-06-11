"""Processing P5 — package ↔ registry compatibility validation.

``resolver.resolve`` calls ``validate_package_compatibility`` after its
runtime check, so an invalid model/tracker/analytics combination is
rejected at package-resolution time (the worker posts ``failed`` with a
clear message) instead of mis-running a pipeline.

Status gating (the core rule):

- a ``supported`` package may only reference ``supported`` components;
- ``probe`` / ``experimental`` packages may reference ``probe`` /
  ``experimental`` / ``supported`` components;
- ``planned`` components (no runnable artifacts) and ``deprecated``
  components are NEVER referencable, regardless of package status.

Deliberately not overbuilt: the checks below exist to stop invalid
combinations and to give future product-facing packages a place to hang
richer rules — not to model every possible constraint today.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from app.errors import RegistryValidationError
from app.packages.registries import (
    COMPUTE_TIERS,
    PRICING_TIERS,
    RUNNABLE_COMPONENT_STATUSES,
    AnalyticsProfileSpec,
    ModelSpec,
    TrackerProfileSpec,
    load_analytics_registry,
    load_model_registry,
    load_tracker_registry,
)

if TYPE_CHECKING:  # runtime import would cycle (resolver imports this module)
    from app.packages.resolver import PackageSpec

PACKAGE_STATUSES = frozenset({"probe", "experimental", "supported", "deprecated"})

# Which component statuses a package of a given status may reference.
_ALLOWED_COMPONENT_STATUSES: dict[str, frozenset[str]] = {
    "supported": frozenset({"supported"}),
    "experimental": RUNNABLE_COMPONENT_STATUSES,
    "probe": RUNNABLE_COMPONENT_STATUSES,
    # A deprecated package keeps resolving (existing sessions must not
    # break) under the probe/experimental component rule.
    "deprecated": RUNNABLE_COMPONENT_STATUSES,
}

# Package runtime -> the model runtime_support token that runtime requires.
_MODEL_RUNTIME_REQUIREMENT: dict[str, str] = {"deepstream": "deepstream_nvinfer"}

# Outputs that are only attributable with a tracker id (per-track events).
OUTPUTS_REQUIRING_TRACKER = frozenset(
    {
        "processor.line_crossing_object",
        "processor.zone_entered",
        "processor.zone_exited",
        "processor.dwell_started",
        "processor.dwell_updated",
        "processor.dwell_exceeded",
    }
)


def _check_component_status(
    *, package: PackageSpec, kind: str, component_id: str, status: str
) -> None:
    allowed = _ALLOWED_COMPONENT_STATUSES[package.status]
    if status not in allowed:
        raise RegistryValidationError(
            f"package {package.package_id!r} (status {package.status!r}) may not reference "
            f"{kind} {component_id!r} with status {status!r}; allowed component statuses: "
            f"{sorted(allowed)}"
        )


def validate_package_compatibility(
    spec: PackageSpec,
    *,
    models: dict[str, ModelSpec] | None = None,
    trackers: dict[str, TrackerProfileSpec] | None = None,
    analytics: dict[str, AnalyticsProfileSpec] | None = None,
) -> None:
    """Raise ``RegistryValidationError`` if ``spec`` references unknown or
    incompatible registry components. The ``models``/``trackers``/
    ``analytics`` overrides exist for tests; production callers use the
    cached registries."""
    models = load_model_registry() if models is None else models
    trackers = load_tracker_registry() if trackers is None else trackers
    analytics = load_analytics_registry() if analytics is None else analytics

    pid = spec.package_id
    if spec.status not in PACKAGE_STATUSES:
        raise RegistryValidationError(
            f"package {pid!r}: invalid status {spec.status!r}; allowed: {sorted(PACKAGE_STATUSES)}"
        )
    if spec.compute_tier is not None and spec.compute_tier not in COMPUTE_TIERS:
        raise RegistryValidationError(
            f"package {pid!r}: invalid compute_tier {spec.compute_tier!r}"
        )
    if spec.pricing_tier is not None and spec.pricing_tier not in PRICING_TIERS:
        raise RegistryValidationError(
            f"package {pid!r}: invalid pricing_tier {spec.pricing_tier!r}"
        )

    model: ModelSpec | None = None
    if spec.model_id is not None:
        model = models.get(spec.model_id)
        if model is None:
            raise RegistryValidationError(
                f"package {pid!r} references unknown model_id {spec.model_id!r}; "
                f"known: {sorted(models)}"
            )
        _check_component_status(
            package=spec, kind="model", component_id=model.model_id, status=model.status
        )
        required = _MODEL_RUNTIME_REQUIREMENT.get(spec.runtime)
        if required is None:
            raise RegistryValidationError(
                f"package {pid!r} (runtime {spec.runtime!r}) cannot reference a model; "
                f"model runtime mappings exist for: {sorted(_MODEL_RUNTIME_REQUIREMENT)}"
            )
        if required not in model.runtime_support:
            raise RegistryValidationError(
                f"package {pid!r} requires model runtime support {required!r} but model "
                f"{model.model_id!r} supports only {sorted(model.runtime_support)}"
            )

    if spec.supported_classes:
        if model is None:
            raise RegistryValidationError(
                f"package {pid!r} declares supported_classes but references no model_id"
            )
        extra = set(spec.supported_classes) - set(model.supported_classes)
        if extra:
            raise RegistryValidationError(
                f"package {pid!r} supported_classes {sorted(extra)} are not supported by "
                f"model {model.model_id!r}"
            )

    tracker: TrackerProfileSpec | None = None
    if spec.tracker_profile_id is not None:
        tracker = trackers.get(spec.tracker_profile_id)
        if tracker is None:
            raise RegistryValidationError(
                f"package {pid!r} references unknown tracker_profile_id "
                f"{spec.tracker_profile_id!r}; known: {sorted(trackers)}"
            )
        _check_component_status(
            package=spec,
            kind="tracker profile",
            component_id=tracker.tracker_profile_id,
            status=tracker.status,
        )
        if tracker.runtime != spec.runtime:
            raise RegistryValidationError(
                f"package {pid!r} (runtime {spec.runtime!r}) references tracker profile "
                f"{tracker.tracker_profile_id!r} for runtime {tracker.runtime!r}"
            )

    profile: AnalyticsProfileSpec | None = None
    if spec.analytics_profile_id is not None:
        profile = analytics.get(spec.analytics_profile_id)
        if profile is None:
            raise RegistryValidationError(
                f"package {pid!r} references unknown analytics_profile_id "
                f"{spec.analytics_profile_id!r}; known: {sorted(analytics)}"
            )
        _check_component_status(
            package=spec,
            kind="analytics profile",
            component_id=profile.analytics_profile_id,
            status=profile.status,
        )
        if profile.runtime != spec.runtime:
            raise RegistryValidationError(
                f"package {pid!r} (runtime {spec.runtime!r}) references analytics profile "
                f"{profile.analytics_profile_id!r} for runtime {profile.runtime!r}"
            )
        if profile.requires_tracker and tracker is None:
            raise RegistryValidationError(
                f"package {pid!r}: analytics profile {profile.analytics_profile_id!r} "
                "requires a tracker but the package declares no tracker_profile_id"
            )
        unsupported = set(spec.outputs) - set(profile.supported_outputs)
        if unsupported:
            raise RegistryValidationError(
                f"package {pid!r} declares outputs {sorted(unsupported)} not supported by "
                f"analytics profile {profile.analytics_profile_id!r}"
            )

    tracked_outputs = set(spec.outputs) & OUTPUTS_REQUIRING_TRACKER
    if tracked_outputs and tracker is None:
        raise RegistryValidationError(
            f"package {pid!r} declares per-track outputs {sorted(tracked_outputs)} but no "
            "tracker_profile_id"
        )
