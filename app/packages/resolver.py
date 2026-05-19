"""Resolve a ``package_id`` to a package spec.

The package registry entry declares the runtime; the worker manager
dispatches on ``spec.runtime``. M8.2B supports two runtimes:

- ``mock``       — M8.1 mock handoff (``mock.detect.v0``)
- ``deepstream`` — M8.2B decode-only probe (``deepstream.decode_probe.v0``)

Strict: an unknown ``package_id`` or a package whose runtime is not in the
supported set is rejected clearly (``PackageResolutionError``) so the worker
posts ``failed`` rather than silently mis-running.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from importlib import resources

from app.errors import PackageResolutionError

SUPPORTED_RUNTIMES: frozenset[str] = frozenset({"mock", "deepstream"})


@dataclass(frozen=True)
class PackageSpec:
    package_id: str
    display_name: str
    runtime: str
    description: str
    outputs: tuple[str, ...]
    supports: dict[str, bool] = field(default_factory=dict)


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
        )
        registry[spec.package_id] = spec
    return registry


def resolve(package_id: str) -> PackageSpec:
    """Return the spec for ``package_id``.

    Raises ``PackageResolutionError`` if the package is unknown or if its
    declared runtime is not in ``SUPPORTED_RUNTIMES``.
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
    return spec
