"""Resolve a ``package_id`` to a package spec.

The package registry entry declares the runtime; the worker manager
dispatches on ``spec.runtime``. Two runtimes are supported:

- ``mock``       — M8.1 mock handoff (``mock.detect.v0``)
- ``deepstream`` — the DeepStream probe packages (decode / inference /
  metadata / tracker / analytics)

Processing P5: packages now also reference the model / tracker / analytics
registries (``app.packages.registries``) via optional ``model_id`` /
``tracker_profile_id`` / ``analytics_profile_id`` fields, and ``resolve``
runs ``validate_package_compatibility`` so an invalid combination is
rejected at resolution time.

Strict: an unknown ``package_id``, an unsupported runtime, or an
incompatible registry combination is rejected clearly
(``PackageResolutionError``) so the worker posts ``failed`` rather than
silently mis-running.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from importlib import resources

from app.errors import PackageResolutionError
from app.packages.compatibility import validate_package_compatibility

SUPPORTED_RUNTIMES: frozenset[str] = frozenset({"mock", "deepstream"})


@dataclass(frozen=True)
class PackageSpec:
    package_id: str
    display_name: str
    runtime: str
    description: str
    outputs: tuple[str, ...]
    supports: dict[str, bool] = field(default_factory=dict)
    # --- Processing P5: registry references + product metadata -----------
    # status gates which component statuses the package may reference
    # (see app.packages.compatibility). All current packages are probes.
    status: str = "probe"
    model_id: str | None = None
    tracker_profile_id: str | None = None
    analytics_profile_id: str | None = None
    # Classes this package exposes; must be a subset of the model's
    # supported_classes. () = whatever the model detects (no narrowing).
    supported_classes: tuple[str, ...] = ()
    # Placeholder usage/cost metadata — NOT billed, NOT final pricing.
    compute_tier: str | None = None
    pricing_tier: str | None = None
    # --- P5-D: package-readiness metadata --------------------------------
    # Internal product-facing candidate fields for the future package
    # assignment API. NOT a package catalog, NOT exposed through api-server
    # yet. ``production_ready`` is an honesty gate: it may only be true on
    # ``supported`` packages (enforced in compatibility validation).
    category: str | None = None
    best_for: tuple[str, ...] = ()
    not_recommended_for: tuple[str, ...] = ()
    setup_complexity: str | None = None
    supported_shapes: tuple[str, ...] = ()
    supported_triggers: tuple[str, ...] = ()
    default_target_fps: int | None = None
    recommended_confidence_threshold: float | None = None
    observation_policy: str | None = None
    test_mode_safe: bool = True
    production_ready: bool = False


@lru_cache(maxsize=1)
def load_registry() -> dict[str, PackageSpec]:
    raw = (
        resources.files("app.packages")
        .joinpath("package_registry.json")
        .read_text(encoding="utf-8")
    )
    data = json.loads(raw)
    registry: dict[str, PackageSpec] = {}
    for entry in data["packages"]:
        spec = PackageSpec(
            package_id=entry["package_id"],
            display_name=entry["display_name"],
            runtime=entry["runtime"],
            description=entry["description"],
            outputs=tuple(entry.get("outputs", [])),
            supports=dict(entry.get("supports", {})),
            status=entry.get("status", "probe"),
            model_id=entry.get("model_id"),
            tracker_profile_id=entry.get("tracker_profile_id"),
            analytics_profile_id=entry.get("analytics_profile_id"),
            supported_classes=tuple(entry.get("supported_classes", [])),
            compute_tier=entry.get("compute_tier"),
            pricing_tier=entry.get("pricing_tier"),
            category=entry.get("category"),
            best_for=tuple(entry.get("best_for", [])),
            not_recommended_for=tuple(entry.get("not_recommended_for", [])),
            setup_complexity=entry.get("setup_complexity"),
            supported_shapes=tuple(entry.get("supported_shapes", [])),
            supported_triggers=tuple(entry.get("supported_triggers", [])),
            default_target_fps=entry.get("default_target_fps"),
            recommended_confidence_threshold=entry.get("recommended_confidence_threshold"),
            observation_policy=entry.get("observation_policy"),
            test_mode_safe=entry.get("test_mode_safe", True),
            production_ready=entry.get("production_ready", False),
        )
        registry[spec.package_id] = spec
    return registry


def resolve(package_id: str) -> PackageSpec:
    """Return the spec for ``package_id``.

    Raises ``PackageResolutionError`` if the package is unknown, if its
    declared runtime is not in ``SUPPORTED_RUNTIMES``, or if its registry
    references are unknown/incompatible (Processing P5).
    """
    registry = load_registry()
    spec = registry.get(package_id)
    if spec is None:
        raise PackageResolutionError(
            f"unknown package_id {package_id!r}; known: {sorted(registry)}"
        )
    if spec.runtime not in SUPPORTED_RUNTIMES:
        raise PackageResolutionError(
            f"package {package_id!r} declares unsupported runtime "
            f"{spec.runtime!r}; supported: {sorted(SUPPORTED_RUNTIMES)}"
        )
    validate_package_compatibility(spec)
    return spec
