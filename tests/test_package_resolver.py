from __future__ import annotations

import pytest
from app.errors import PackageResolutionError
from app.packages import resolver
from app.packages.resolver import PackageSpec, resolve


def test_resolves_known_mock_package() -> None:
    spec = resolve("mock.detect.v0")
    assert spec.package_id == "mock.detect.v0"
    assert spec.runtime == "mock"
    assert "processor.mock_detection" in spec.outputs


def test_resolves_deepstream_decode_probe_package() -> None:
    spec = resolve("deepstream.decode_probe.v0")
    assert spec.package_id == "deepstream.decode_probe.v0"
    assert spec.runtime == "deepstream"
    assert spec.outputs == ()  # decode only — no observations
    assert spec.supports.get("decode_only") is True


def test_resolves_deepstream_inference_probe_package() -> None:
    spec = resolve("deepstream.inference_probe.v0")
    assert spec.package_id == "deepstream.inference_probe.v0"
    assert spec.runtime == "deepstream"
    assert spec.outputs == ()  # inference only — no observations
    assert spec.supports.get("inference_probe") is True
    assert spec.supports.get("decode_only") is None  # distinct from decode probe


def test_resolves_deepstream_metadata_probe_package() -> None:
    spec = resolve("deepstream.metadata_probe.v0")
    assert spec.package_id == "deepstream.metadata_probe.v0"
    assert spec.runtime == "deepstream"
    assert spec.outputs == ()  # status/debug evidence only — no observations
    assert spec.supports.get("metadata_probe") is True
    # distinct from the other deepstream packages
    assert spec.supports.get("inference_probe") is None
    assert spec.supports.get("decode_only") is None


def test_resolves_deepstream_tracker_probe_package() -> None:
    spec = resolve("deepstream.tracker_probe.v0")
    assert spec.package_id == "deepstream.tracker_probe.v0"
    assert spec.runtime == "deepstream"
    assert spec.outputs == ()  # status/debug evidence only — no observations
    assert spec.supports.get("tracker_probe") is True
    # distinct from the other deepstream packages (most-specific-wins)
    assert spec.supports.get("metadata_probe") is None
    assert spec.supports.get("inference_probe") is None
    assert spec.supports.get("decode_only") is None


def test_resolves_deepstream_analytics_probe_package() -> None:
    spec = resolve("deepstream.analytics_probe.v0")
    assert spec.package_id == "deepstream.analytics_probe.v0"
    assert spec.runtime == "deepstream"
    assert spec.outputs == ()  # status/debug evidence only — no observations
    assert spec.supports.get("analytics_probe") is True
    # distinct from the other deepstream packages (most-specific-wins)
    assert spec.supports.get("tracker_probe") is None
    assert spec.supports.get("metadata_probe") is None
    assert spec.supports.get("inference_probe") is None
    assert spec.supports.get("decode_only") is None


def test_rejects_unknown_package() -> None:
    with pytest.raises(PackageResolutionError, match="unknown package_id"):
        resolve("does.not.exist")


def test_rejects_unsupported_runtime(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    bad = PackageSpec(
        package_id="triton.thing.v0",
        display_name="Triton Thing",
        runtime="triton",  # not in SUPPORTED_RUNTIMES
        description="x",
        outputs=(),
    )
    monkeypatch.setattr(resolver, "load_registry", lambda: {bad.package_id: bad})
    with pytest.raises(PackageResolutionError, match="unsupported runtime"):
        resolve("triton.thing.v0")
