"""Processing P5 — model / tracker / analytics-profile registries.

These registries sit BELOW the package layer:

    model_registry + tracker_registry + analytics_profile_registry
        → package_registry (product-facing bundles, ``resolver.py``)
        → resolved runtime config
        → DeepStream worker

Models, tracker profiles, and analytics profiles are curated lower-level
building blocks. Packages combine them (plus observation policy and
compute/pricing metadata) into product-facing use cases; compatibility
between a package and its referenced components is enforced by
``app.packages.compatibility``.

Registry data is NOT customer-facing yet. ``status`` gates what may run:

- ``probe``        — proven in an Upstream probe milestone; dev/probe use.
- ``supported``    — production-supported (license + deployment verified).
- ``experimental`` — runnable but unverified; dev/probe packages only.
- ``planned``      — placeholder; no runnable artifacts. NEVER resolvable.
- ``deprecated``   — retired; NEVER resolvable.

Compute/pricing fields (``compute_tier`` / ``pricing_tier`` /
``estimated_cost_weight`` / ``observation_volume_risk``) are PLACEHOLDER
metadata for future processing-usage metering — they are not billed and not
final pricing.

Loaders are strict: duplicate ids, missing required fields, and invalid
enum values raise ``RegistryValidationError`` at load time so a registry
typo fails fast instead of mis-running a worker. Parsing is split into pure
``parse_*`` functions (unit-testable on plain dicts) and cached ``load_*``
file readers (mirrors ``resolver.load_registry``).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
from typing import Any

from app.errors import RegistryValidationError

# --- enums --------------------------------------------------------------

COMPONENT_STATUSES = frozenset({"probe", "supported", "experimental", "planned", "deprecated"})
# Statuses that may actually run (planned has no artifacts; deprecated is
# retired). Resolution helpers below only match runnable entries.
RUNNABLE_COMPONENT_STATUSES = frozenset({"probe", "supported", "experimental"})

MODEL_TASKS = frozenset(
    {"object_detection", "instance_segmentation", "pose_estimation", "classification"}
)
MODEL_RUNTIME_SUPPORT = frozenset(
    {"deepstream_nvinfer", "deepstream_nvinferserver", "python", "triton"}
)
TRACKER_RUNTIMES = frozenset({"deepstream", "python"})
ANALYTICS_RUNTIMES = frozenset({"deepstream"})
COMPUTE_TIERS = frozenset({"low", "medium", "high", "very_high"})
PRICING_TIERS = frozenset({"free_probe", "lite", "balanced", "accurate", "custom"})
OBSERVATION_VOLUME_RISKS = frozenset({"low", "medium", "high"})
ANALYTICS_CONFIG_MODES = frozenset({"generated_ini"})

# --- specs ----------------------------------------------------------------


@dataclass(frozen=True)
class ModelDeepStreamSpec:
    """DeepStream-specific deployment facts for one model."""

    config_template: str
    parser: str
    labels_file: str
    engine_strategy: str
    supports_int8: bool
    supports_fp16: bool
    requires_custom_parser: bool
    model_artifact_path: str
    known_working_image: str | None
    blackwell_supported: bool | None
    # P5-B: when set, the model's artifacts live OUTSIDE the DeepStream image
    # on the GPU host, under ``<deepstream_model_store_dir>/<artifact_subdir>``
    # (bind-mounted into the container at the same absolute path).
    # ``model_artifact_path`` then documents the DEFAULT effective path.
    # ``None`` = artifacts are baked into the image at ``model_artifact_path``
    # (the TrafficCamNet sample case).
    artifact_subdir: str | None = None


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    display_name: str
    family: str
    task: str
    version: str
    status: str
    license: str
    license_notes: str
    source: str
    supported_classes: tuple[str, ...]
    class_id_map: dict[str, int]
    runtime_support: tuple[str, ...]
    # Operator-facing names that map onto this model (e.g. the
    # ``UPSTREAM_DEEPSTREAM_INFERENCE_MODEL`` setting values).
    aliases: tuple[str, ...]
    deepstream: ModelDeepStreamSpec | None
    input: dict[str, int]
    default_confidence_threshold: float | None
    recommended_confidence_threshold: float | None
    compute_tier: str
    pricing_tier: str
    default_target_fps: int | None
    recommended_resolution: str | None
    gpu_required: bool
    estimated_cost_weight: float
    observation_volume_risk: str
    notes: str


@dataclass(frozen=True)
class TrackerProfileSpec:
    tracker_profile_id: str
    display_name: str
    runtime: str
    family: str
    status: str
    algorithm: str
    requires_reid: bool
    # Model dependency beyond the primary detector (e.g. a ReID model id);
    # None for trackers that only need detector boxes.
    requires_model: str | None
    deepstream: dict[str, Any]
    best_for: tuple[str, ...]
    not_recommended_for: tuple[str, ...]
    compute_tier: str
    estimated_cost_weight: float
    # Low-level config values, section -> key -> scalar, in render order.
    # For DeepStream trackers this is the exact ll-config YAML content.
    settings: dict[str, dict[str, Any]]
    notes: str


@dataclass(frozen=True)
class AnalyticsProfileSpec:
    analytics_profile_id: str
    display_name: str
    runtime: str
    status: str
    supported_outputs: tuple[str, ...]
    supported_shapes: tuple[str, ...]
    requires_tracker: bool
    supports_line_crossing: bool
    supports_roi_occupancy: bool
    supports_zone_transitions: bool
    supports_dwell: bool
    supports_direction: bool
    config_mode: str
    compute_tier: str
    estimated_cost_weight: float
    observation_volume_risk: str
    notes: str


# --- strict field extraction helpers ---------------------------------------


def _req_str(entry: dict[str, Any], key: str, ctx: str) -> str:
    value = entry.get(key)
    if not isinstance(value, str) or not value:
        raise RegistryValidationError(f"{ctx}: missing/invalid required string field {key!r}")
    return value


def _opt_str(entry: dict[str, Any], key: str, ctx: str) -> str | None:
    value = entry.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise RegistryValidationError(f"{ctx}: field {key!r} must be a string or null")
    return value


def _req_bool(entry: dict[str, Any], key: str, ctx: str) -> bool:
    value = entry.get(key)
    if not isinstance(value, bool):
        raise RegistryValidationError(f"{ctx}: missing/invalid required boolean field {key!r}")
    return value


def _opt_bool(entry: dict[str, Any], key: str, ctx: str) -> bool | None:
    value = entry.get(key)
    if value is None:
        return None
    if not isinstance(value, bool):
        raise RegistryValidationError(f"{ctx}: field {key!r} must be a boolean or null")
    return value


def _opt_number(entry: dict[str, Any], key: str, ctx: str) -> float | None:
    value = entry.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RegistryValidationError(f"{ctx}: field {key!r} must be a number or null")
    return float(value)


def _req_number(entry: dict[str, Any], key: str, ctx: str) -> float:
    value = _opt_number(entry, key, ctx)
    if value is None:
        raise RegistryValidationError(f"{ctx}: missing required number field {key!r}")
    return value


def _opt_int(entry: dict[str, Any], key: str, ctx: str) -> int | None:
    value = entry.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise RegistryValidationError(f"{ctx}: field {key!r} must be an integer or null")
    return value


def _str_tuple(entry: dict[str, Any], key: str, ctx: str) -> tuple[str, ...]:
    value = entry.get(key, [])
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise RegistryValidationError(f"{ctx}: field {key!r} must be a list of strings")
    return tuple(value)


def _enum(value: str, allowed: frozenset[str], key: str, ctx: str) -> str:
    if value not in allowed:
        raise RegistryValidationError(f"{ctx}: invalid {key} {value!r}; allowed: {sorted(allowed)}")
    return value


def _enum_tuple(
    values: tuple[str, ...], allowed: frozenset[str], key: str, ctx: str
) -> tuple[str, ...]:
    for v in values:
        _enum(v, allowed, key, ctx)
    return values


# --- model registry ---------------------------------------------------------


def _parse_model_deepstream(raw: Any, ctx: str) -> ModelDeepStreamSpec | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise RegistryValidationError(f"{ctx}: 'deepstream' must be an object or null")
    return ModelDeepStreamSpec(
        config_template=_req_str(raw, "config_template", ctx),
        parser=_req_str(raw, "parser", ctx),
        labels_file=_req_str(raw, "labels_file", ctx),
        engine_strategy=_req_str(raw, "engine_strategy", ctx),
        supports_int8=_req_bool(raw, "supports_int8", ctx),
        supports_fp16=_req_bool(raw, "supports_fp16", ctx),
        requires_custom_parser=_req_bool(raw, "requires_custom_parser", ctx),
        model_artifact_path=_req_str(raw, "model_artifact_path", ctx),
        known_working_image=_opt_str(raw, "known_working_image", ctx),
        blackwell_supported=_opt_bool(raw, "blackwell_supported", ctx),
        artifact_subdir=_opt_str(raw, "artifact_subdir", ctx),
    )


def parse_model_registry(data: dict[str, Any]) -> dict[str, ModelSpec]:
    """Pure parser/validator for model-registry JSON. Raises
    ``RegistryValidationError`` on duplicate ids / missing fields / bad
    enums."""
    registry: dict[str, ModelSpec] = {}
    seen_aliases: dict[str, str] = {}
    for entry in data.get("models", []):
        if not isinstance(entry, dict):
            raise RegistryValidationError("model registry: every entry must be an object")
        model_id = _req_str(entry, "model_id", "model registry")
        ctx = f"model {model_id!r}"
        if model_id in registry:
            raise RegistryValidationError(f"{ctx}: duplicate model_id")
        class_id_map_raw = entry.get("class_id_map", {})
        if not isinstance(class_id_map_raw, dict) or not all(
            isinstance(k, str) and isinstance(v, int) and not isinstance(v, bool)
            for k, v in class_id_map_raw.items()
        ):
            raise RegistryValidationError(f"{ctx}: class_id_map must map class name -> int id")
        input_raw = entry.get("input", {})
        if not isinstance(input_raw, dict) or not all(
            isinstance(k, str) and isinstance(v, int) and not isinstance(v, bool)
            for k, v in input_raw.items()
        ):
            raise RegistryValidationError(f"{ctx}: input must map dimension name -> int")
        spec = ModelSpec(
            model_id=model_id,
            display_name=_req_str(entry, "display_name", ctx),
            family=_req_str(entry, "family", ctx),
            task=_enum(_req_str(entry, "task", ctx), MODEL_TASKS, "task", ctx),
            version=_req_str(entry, "version", ctx),
            status=_enum(_req_str(entry, "status", ctx), COMPONENT_STATUSES, "status", ctx),
            license=_req_str(entry, "license", ctx),
            license_notes=_req_str(entry, "license_notes", ctx),
            source=_req_str(entry, "source", ctx),
            supported_classes=_str_tuple(entry, "supported_classes", ctx),
            class_id_map=dict(class_id_map_raw),
            runtime_support=_enum_tuple(
                _str_tuple(entry, "runtime_support", ctx),
                MODEL_RUNTIME_SUPPORT,
                "runtime_support",
                ctx,
            ),
            aliases=_str_tuple(entry, "aliases", ctx),
            deepstream=_parse_model_deepstream(entry.get("deepstream"), ctx),
            input=dict(input_raw),
            default_confidence_threshold=_opt_number(entry, "default_confidence_threshold", ctx),
            recommended_confidence_threshold=_opt_number(
                entry, "recommended_confidence_threshold", ctx
            ),
            compute_tier=_enum(
                _req_str(entry, "compute_tier", ctx), COMPUTE_TIERS, "compute_tier", ctx
            ),
            pricing_tier=_enum(
                _req_str(entry, "pricing_tier", ctx), PRICING_TIERS, "pricing_tier", ctx
            ),
            default_target_fps=_opt_int(entry, "default_target_fps", ctx),
            recommended_resolution=_opt_str(entry, "recommended_resolution", ctx),
            gpu_required=_req_bool(entry, "gpu_required", ctx),
            estimated_cost_weight=_req_number(entry, "estimated_cost_weight", ctx),
            observation_volume_risk=_enum(
                _req_str(entry, "observation_volume_risk", ctx),
                OBSERVATION_VOLUME_RISKS,
                "observation_volume_risk",
                ctx,
            ),
            notes=_req_str(entry, "notes", ctx),
        )
        for alias in spec.aliases:
            key = alias.strip().lower()
            if key in seen_aliases:
                raise RegistryValidationError(
                    f"{ctx}: alias {alias!r} already claimed by model {seen_aliases[key]!r}"
                )
            seen_aliases[key] = model_id
        registry[model_id] = spec
    return registry


@lru_cache(maxsize=1)
def load_model_registry() -> dict[str, ModelSpec]:
    raw = (
        resources.files("app.packages").joinpath("model_registry.json").read_text(encoding="utf-8")
    )
    return parse_model_registry(json.loads(raw))


def find_model_by_alias(name: str) -> ModelSpec | None:
    """Match an operator-facing model name (``deepstream_inference_model``)
    to its registry entry, regardless of status. Callers gate on status."""
    key = name.strip().lower()
    for spec in load_model_registry().values():
        if key == spec.model_id or key in (a.strip().lower() for a in spec.aliases):
            return spec
    return None


# --- tracker registry --------------------------------------------------------


def parse_tracker_registry(data: dict[str, Any]) -> dict[str, TrackerProfileSpec]:
    registry: dict[str, TrackerProfileSpec] = {}
    for entry in data.get("trackers", []):
        if not isinstance(entry, dict):
            raise RegistryValidationError("tracker registry: every entry must be an object")
        profile_id = _req_str(entry, "tracker_profile_id", "tracker registry")
        ctx = f"tracker profile {profile_id!r}"
        if profile_id in registry:
            raise RegistryValidationError(f"{ctx}: duplicate tracker_profile_id")
        deepstream_raw = entry.get("deepstream", {})
        if not isinstance(deepstream_raw, dict):
            raise RegistryValidationError(f"{ctx}: 'deepstream' must be an object")
        settings_raw = entry.get("settings", {})
        if not isinstance(settings_raw, dict) or not all(
            isinstance(section, str) and isinstance(values, dict)
            for section, values in settings_raw.items()
        ):
            raise RegistryValidationError(
                f"{ctx}: settings must map section -> {{key: scalar}} objects"
            )
        registry[profile_id] = TrackerProfileSpec(
            tracker_profile_id=profile_id,
            display_name=_req_str(entry, "display_name", ctx),
            runtime=_enum(_req_str(entry, "runtime", ctx), TRACKER_RUNTIMES, "runtime", ctx),
            family=_req_str(entry, "family", ctx),
            status=_enum(_req_str(entry, "status", ctx), COMPONENT_STATUSES, "status", ctx),
            algorithm=_req_str(entry, "algorithm", ctx),
            requires_reid=_req_bool(entry, "requires_reid", ctx),
            requires_model=_opt_str(entry, "requires_model", ctx),
            deepstream=dict(deepstream_raw),
            best_for=_str_tuple(entry, "best_for", ctx),
            not_recommended_for=_str_tuple(entry, "not_recommended_for", ctx),
            compute_tier=_enum(
                _req_str(entry, "compute_tier", ctx), COMPUTE_TIERS, "compute_tier", ctx
            ),
            estimated_cost_weight=_req_number(entry, "estimated_cost_weight", ctx),
            settings={section: dict(values) for section, values in settings_raw.items()},
            notes=_req_str(entry, "notes", ctx),
        )
    return registry


@lru_cache(maxsize=1)
def load_tracker_registry() -> dict[str, TrackerProfileSpec]:
    raw = (
        resources.files("app.packages")
        .joinpath("tracker_registry.json")
        .read_text(encoding="utf-8")
    )
    return parse_tracker_registry(json.loads(raw))


def find_runnable_deepstream_tracker_by_algorithm(algorithm: str) -> TrackerProfileSpec | None:
    """The runnable DeepStream tracker profile for a low-level algorithm
    name (the ``deepstream_tracker_algorithm`` setting). Planned/deprecated
    profiles and profiles without authored ``settings`` never match."""
    key = algorithm.strip().lower()
    for spec in load_tracker_registry().values():
        if (
            spec.runtime == "deepstream"
            and spec.algorithm == key
            and spec.status in RUNNABLE_COMPONENT_STATUSES
            and spec.settings
        ):
            return spec
    return None


def runnable_deepstream_tracker_algorithms() -> list[str]:
    return sorted(
        spec.algorithm
        for spec in load_tracker_registry().values()
        if spec.runtime == "deepstream"
        and spec.status in RUNNABLE_COMPONENT_STATUSES
        and spec.settings
    )


# --- analytics profile registry ----------------------------------------------


def parse_analytics_registry(data: dict[str, Any]) -> dict[str, AnalyticsProfileSpec]:
    registry: dict[str, AnalyticsProfileSpec] = {}
    for entry in data.get("analytics_profiles", []):
        if not isinstance(entry, dict):
            raise RegistryValidationError("analytics registry: every entry must be an object")
        profile_id = _req_str(entry, "analytics_profile_id", "analytics registry")
        ctx = f"analytics profile {profile_id!r}"
        if profile_id in registry:
            raise RegistryValidationError(f"{ctx}: duplicate analytics_profile_id")
        registry[profile_id] = AnalyticsProfileSpec(
            analytics_profile_id=profile_id,
            display_name=_req_str(entry, "display_name", ctx),
            runtime=_enum(_req_str(entry, "runtime", ctx), ANALYTICS_RUNTIMES, "runtime", ctx),
            status=_enum(_req_str(entry, "status", ctx), COMPONENT_STATUSES, "status", ctx),
            supported_outputs=_str_tuple(entry, "supported_outputs", ctx),
            supported_shapes=_str_tuple(entry, "supported_shapes", ctx),
            requires_tracker=_req_bool(entry, "requires_tracker", ctx),
            supports_line_crossing=_req_bool(entry, "supports_line_crossing", ctx),
            supports_roi_occupancy=_req_bool(entry, "supports_roi_occupancy", ctx),
            supports_zone_transitions=_req_bool(entry, "supports_zone_transitions", ctx),
            supports_dwell=_req_bool(entry, "supports_dwell", ctx),
            supports_direction=_req_bool(entry, "supports_direction", ctx),
            config_mode=_enum(
                _req_str(entry, "config_mode", ctx),
                ANALYTICS_CONFIG_MODES,
                "config_mode",
                ctx,
            ),
            compute_tier=_enum(
                _req_str(entry, "compute_tier", ctx), COMPUTE_TIERS, "compute_tier", ctx
            ),
            estimated_cost_weight=_req_number(entry, "estimated_cost_weight", ctx),
            observation_volume_risk=_enum(
                _req_str(entry, "observation_volume_risk", ctx),
                OBSERVATION_VOLUME_RISKS,
                "observation_volume_risk",
                ctx,
            ),
            notes=_req_str(entry, "notes", ctx),
        )
    return registry


@lru_cache(maxsize=1)
def load_analytics_registry() -> dict[str, AnalyticsProfileSpec]:
    raw = (
        resources.files("app.packages")
        .joinpath("analytics_registry.json")
        .read_text(encoding="utf-8")
    )
    return parse_analytics_registry(json.loads(raw))
