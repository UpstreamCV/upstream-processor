"""Processing P5-E — YOLOX + NvDCF analytics package + package-driven tracker.

No Docker / GPU / pyds. Mirrors tests/test_yolox_analytics_package.py for the
NvDCF variant, and proves the P5-E runtime seam: the package's
``tracker_profile_id`` (not the global ``deepstream_tracker_algorithm``
setting) now selects the running tracker. Covers:

- ``deepstream.yolox_analytics_nvdcf_probe.v0`` resolving with the full
  model+tracker+analytics reference set, ``experimental`` status, and the same
  honest outputs as the IOU variant;
- the analytics process manager rendering the NvDCF tracker YAML (byte-equal to
  ``build_tracker_config("nvdcf")``, and distinct from the IOU body);
- the worker deriving ``tracker_algorithm="nvdcf"`` + lib/width/height from the
  package profile and threading them into the process manager, while the IOU
  package still drives ``"iou"``;
- the applied lineage + status payload reporting the REAL tracker (nvdcf);
- fail-closed behavior: a package that references a non-runnable tracker
  profile keeps that tracker (NO silent IOU fallback) and fails at process
  start.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from app.observations.schemas import OBSERVATION_TYPES, EligibleSession
from app.packages.resolver import PackageSpec, resolve
from app.runtimes.deepstream_runtime.inference import build_tracker_config
from app.runtimes.deepstream_runtime.metadata_probe import (
    DeepStreamMetadataProcessManager,
)
from app.runtimes.deepstream_runtime.worker import DeepStreamWorker, _MetadataCounters

from tests._fakes import FakeApi
from tests.conftest import eligible_session_dict
from tests.test_deepstream_metadata_probe import FakeMetadataPM

_URL = "rtsp://127.0.0.1:18554/sess_abcdef0123456789"
_NVDCF_PKG = "deepstream.yolox_analytics_nvdcf_probe.v0"
_IOU_PKG = "deepstream.yolox_analytics_probe.v0"
_NVDCF_LIB = "/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so"


def _session(**ov: object) -> EligibleSession:
    return EligibleSession.model_validate(eligible_session_dict(**ov))


# --- package resolution -------------------------------------------------------


def test_nvdcf_package_resolves_with_full_reference_set() -> None:
    spec = resolve(_NVDCF_PKG)
    iou = resolve(_IOU_PKG)
    assert spec.runtime == "deepstream"
    # Experimental until the live A/B promotes it — NOT probe, NOT production.
    assert spec.status == "experimental"
    assert spec.production_ready is False
    assert spec.model_id == "yolox_s_v1"
    assert spec.tracker_profile_id == "deepstream_nvdcf_balanced_v1"
    assert spec.analytics_profile_id == "deepstream_nvdsanalytics_probe_v1"
    assert spec.supported_classes == ("person",)
    # Sits BESIDE the IOU package: same detector/analytics/outputs, only the
    # tracker differs.
    assert spec.outputs == iou.outputs
    assert spec.analytics_profile_id == iou.analytics_profile_id
    assert spec.tracker_profile_id != iou.tracker_profile_id
    # Honest outputs: existing API-supported types only.
    assert set(spec.outputs) <= set(OBSERVATION_TYPES)


# --- DeepStream config: NvDCF tracker YAML ------------------------------------


def test_nvdcf_pm_renders_nvdcf_tracker_config(tmp_path: Path) -> None:
    store = tmp_path / "store"
    model_dir = store / "yolox_s_v1"
    model_dir.mkdir(parents=True)
    (model_dir / "yolox_s.onnx").write_bytes(b"onnx")
    (model_dir / "libnvdsparsebbox_yolox.so").write_bytes(b"so")
    pm = DeepStreamMetadataProcessManager(
        rtsp_url=_URL,
        image="img:test",
        network="host",
        rtsp_transport="tcp",
        probe_timeout_seconds=5,
        stop_timeout_seconds=1,
        model="yolox_s_v1",
        output_base_dir=str(tmp_path / "work"),
        keep_temp_files=True,
        sample_limit=3,
        stream_session_id="s",
        width=1280,
        height=720,
        tracker_enabled=True,
        tracker_algorithm="nvdcf",
        analytics_enabled=True,
        model_store_dir=str(store),
    )
    work = pm._prepare_work_dir()
    pm._work_dir = work
    try:
        tracker_yml = (work / "tracker_config.yml").read_text()
        # Byte-equal to the registry-backed NvDCF builder, and DISTINCT from IOU.
        assert tracker_yml == build_tracker_config("nvdcf")
        assert tracker_yml != build_tracker_config("iou")
        assert "VisualTracker:" in tracker_yml  # NvDCF-specific section
        assert "rtsp://" not in tracker_yml
    finally:
        pm._keep_temp_files = False
        pm._cleanup_work_dir()


# --- package-driven tracker selection (the P5-E seam) -------------------------


def test_worker_drives_nvdcf_tracker_into_pm(fast_settings) -> None:  # type: ignore[no-untyped-def]
    worker = DeepStreamWorker(
        session=_session(internal_rtsp_url=_URL),
        settings=fast_settings,
        api=FakeApi(),
        package=resolve(_NVDCF_PKG),
    )
    # The package profile — not the global setting — selects the tracker.
    assert worker._effective_tracker_algorithm == "nvdcf"
    assert worker._effective_tracker_lib_path == _NVDCF_LIB
    assert worker._effective_tracker_width == 960
    assert worker._effective_tracker_height == 544
    pm = worker._default_pm_factory()
    assert isinstance(pm, DeepStreamMetadataProcessManager)
    assert pm._tracker_enabled is True
    assert pm._tracker_algorithm == "nvdcf"
    assert pm._tracker_lib_path == _NVDCF_LIB


def test_iou_package_still_drives_iou(fast_settings) -> None:  # type: ignore[no-untyped-def]
    """Regression: the IOU YOLOX package is unchanged — it still derives 'iou'
    (now from its profile, which equals the old global default)."""
    worker = DeepStreamWorker(
        session=_session(internal_rtsp_url=_URL),
        settings=fast_settings,
        api=FakeApi(),
        package=resolve(_IOU_PKG),
    )
    assert worker._effective_tracker_algorithm == "iou"
    pm = worker._default_pm_factory()
    assert pm._tracker_algorithm == "iou"


def test_nvdcf_lineage_and_status_report_the_real_tracker(fast_settings) -> None:  # type: ignore[no-untyped-def]
    worker = DeepStreamWorker(
        session=_session(internal_rtsp_url=_URL),
        settings=fast_settings,
        api=FakeApi(),
        package=resolve(_NVDCF_PKG),
        pm_factory=lambda: FakeMetadataPM([]),
    )
    # Applied lineage reflects what actually runs.
    assert worker._resolved_processing_config.applied.tracker == "nvdcf"
    payload = worker._metadata_payload(
        counters=_MetadataCounters(), queue_stats={}, last_app_summary=None
    )
    assert payload["tracker_enabled"] is True
    assert payload["tracker_algorithm"] == "nvdcf"
    assert payload["tracker_profile_id"] == "deepstream_nvdcf_balanced_v1"
    assert payload["analytics_profile_id"] == "deepstream_nvdsanalytics_probe_v1"


# --- fail-closed: no silent IOU fallback --------------------------------------


def _nvsort_package() -> PackageSpec:
    """A (synthetic) analytics package referencing the still-planned NvSORT
    profile. The real resolver would reject it; we build it directly to prove
    the worker/PM fail CLOSED rather than silently running IOU."""
    return PackageSpec(
        package_id="test.yolox_analytics_nvsort.v0",
        display_name="t",
        runtime="deepstream",
        description="t",
        outputs=("processor.zone_entered", "processor.zone_exited"),
        supports={"analytics_probe": True},
        status="experimental",
        model_id="yolox_s_v1",
        tracker_profile_id="deepstream_nvsort_fast_v1",  # planned, no settings
        analytics_profile_id="deepstream_nvdsanalytics_probe_v1",
        supported_classes=("person",),
    )


def test_unrunnable_tracker_profile_fails_closed(fast_settings) -> None:  # type: ignore[no-untyped-def]
    worker = DeepStreamWorker(
        session=_session(internal_rtsp_url=_URL),
        settings=fast_settings,
        api=FakeApi(),
        package=_nvsort_package(),
    )
    # Defensive branch keeps the DECLARED algorithm — it never falls back to iou.
    assert worker._effective_tracker_algorithm == "nvsort"
    pm = worker._default_pm_factory()
    assert pm._tracker_algorithm == "nvsort"
    # Process start fails closed: build_tracker_config rejects the non-runnable
    # profile (the worker then posts ``failed``) instead of running IOU.
    with pytest.raises(ValueError, match="unknown deepstream tracker algorithm"):
        pm._prepare_work_dir()
