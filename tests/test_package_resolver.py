from __future__ import annotations

import pytest
from app.errors import PackageResolutionError
from app.packages.resolver import resolve


def test_resolves_known_mock_package() -> None:
    spec = resolve("mock.detect.v0", "mock")
    assert spec.package_id == "mock.detect.v0"
    assert spec.runtime == "mock"
    assert "processor.mock_detection" in spec.outputs


def test_rejects_unknown_package() -> None:
    with pytest.raises(PackageResolutionError, match="unknown package_id"):
        resolve("does.not.exist", "mock")


def test_rejects_non_mock_runtime() -> None:
    with pytest.raises(PackageResolutionError, match="unsupported runtime"):
        resolve("mock.detect.v0", "deepstream")
