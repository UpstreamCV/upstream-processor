"""Resolve an eligible session's ``package_id`` to a package spec.

M8.1 ships exactly one package (``mock.detect.v0``, runtime ``mock``). Per
the brief this resolver is strict: an unknown ``package_id`` or a non-``mock``
runtime is rejected clearly (``PackageResolutionError``) so the worker posts
``failed`` rather than silently mis-running.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources

from app.errors import PackageResolutionError

_SUPPORTED_RUNTIME = "mock"


@dataclass(frozen=True)
class PackageSpec:
    package_id: str
    display_name: str
    runtime: str
    description: str
    outputs: tuple[str, ...]


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
            outputs=tuple(entry["outputs"]),
        )
        registry[spec.package_id] = spec
    return registry


def resolve(package_id: str, runtime: str) -> PackageSpec:
    """Return the spec for ``package_id``.

    Raises ``PackageResolutionError`` if the package is unknown, if the
    session's ``runtime`` is not ``mock``, or if the catalogued package's
    runtime is not ``mock`` (defence-in-depth for M8.1).
    """
    if runtime != _SUPPORTED_RUNTIME:
        raise PackageResolutionError(
            f"unsupported runtime {runtime!r}: M8.1 only supports {_SUPPORTED_RUNTIME!r}"
        )
    registry = load_registry()
    spec = registry.get(package_id)
    if spec is None:
        raise PackageResolutionError(
            f"unknown package_id {package_id!r}; known: {sorted(registry)}"
        )
    if spec.runtime != _SUPPORTED_RUNTIME:
        raise PackageResolutionError(
            f"package {package_id!r} requires runtime {spec.runtime!r}, not supported in M8.1"
        )
    return spec
