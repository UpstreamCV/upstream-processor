"""Processing P5-D — YOLOX analytics package + lineage cleanup tests.

No Docker / GPU / pyds. Covers:

- ``deepstream.yolox_analytics_probe.v0`` resolving with the full
  model+tracker+analytics reference set, honest outputs, and the new
  package-readiness metadata (incl. the production_ready honesty gate);
- the analytics process manager rendering YOLOX PGIE + IOU tracker YAML +
  nvdsanalytics INI together (with the host model dir mounted);
- the registry-derived lineage map (``model_key_runtime_map``) replacing
  the static ``MODEL_KEY_RUNTIME`` — old aliases included, mock supplement
  preserved — and the worker stamping ``applied.model_key`` from the
  package's effective model (YOLOX stamps ``yolox_s_v1``; TrafficCamNet
  lineage unchanged);
- the analytics status payload carrying the component registry ids while
  staying bounded;
- the full emit path with the YOLOX package: existing observation types
  only, same debounce semantics, writer-optional behavior intact.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from app.errors import RegistryValidationError
from app.observations.schemas import OBSERVATION_TYPES, EligibleSession
from app.observations.writer import ObservationWriter
from app.packages.compatibility import validate_package_compatibility
from app.packages.registries import load_analytics_registry, load_model_registry
from app.packages.resolver import PackageSpec, resolve
from app.processing.resolver import model_key_runtime_map
from app.runtimes.base import WorkerState
from app.runtimes.deepstream_runtime.inference import build_tracker_config
from app.runtimes.deepstream_runtime.metadata_probe import (
    DeepStreamMetadataProcessManager,
)
from app.runtimes.deepstream_runtime.worker import DeepStreamWorker, _MetadataCounters

from tests._fakes import FakeApi
from tests.conftest import eligible_session_dict
from tests.test_deepstream_metadata_probe import FakeMetadataPM

_URL = "rtsp://127.0.0.1:18554/sess_abcdef0123456789"
_PKG = "deepstream.yolox_analytics_probe.v0"


def _session(**ov: object) -> EligibleSession:
    return EligibleSession.model_validate(eligible_session_dict(**ov))


def _processing_envelope(model_key: str) -> dict[str, object]:
    return {"revision": 3, "config": {"version": 1, "model_key": model_key}}


# --- package resolution -------------------------------------------------------


def test_yolox_analytics_package_resolves_with_full_reference_set() -> None:
    spec = resolve(_PKG)
    assert spec.runtime == "deepstream"
    assert spec.status == "probe"
    assert spec.model_id == "yolox_s_v1"
    assert spec.tracker_profile_id == "deepstream_iou_fast_v1"
    assert spec.analytics_profile_id == "deepstream_nvdsanalytics_probe_v1"
    assert spec.supports.get("analytics_probe") is True
    assert spec.supported_classes == ("person",)
    assert spec.compute_tier == "medium"
    assert spec.pricing_tier == "free_probe"


def test_yolox_analytics_outputs_are_existing_supported_types_only() -> None:
    spec = resolve(_PKG)
    profile = load_analytics_registry()["deepstream_nvdsanalytics_probe_v1"]
    assert spec.outputs  # honest, non-empty output declaration
    # Every declared output is implemented/API-supported AND within the
    # analytics profile — no new observation taxonomy in P5-D.
    assert set(spec.outputs) <= set(OBSERVATION_TYPES)
    assert set(spec.outputs) <= set(profile.supported_outputs)
    assert "processor.zone_entered" in spec.outputs
    assert "processor.zone_exited" in spec.outputs
    assert "processor.line_crossing_object" in spec.outputs
    assert "processor.dwell_exceeded" in spec.outputs  # pre-existing (P4-E)


def test_yolox_analytics_classes_subset_of_model() -> None:
    spec = resolve(_PKG)
    model = load_model_registry()["yolox_s_v1"]
    assert set(spec.supported_classes) <= set(model.supported_classes)


def test_yolox_analytics_readiness_metadata() -> None:
    spec = resolve(_PKG)
    assert spec.category == "people_analytics"
    assert spec.setup_complexity == "medium"
    assert spec.supported_shapes == ("polygon_roi", "line_segment")
    assert "zone_entered" in spec.supported_triggers
    assert spec.default_target_fps == 30
    assert spec.recommended_confidence_threshold == 0.4
    assert spec.observation_policy == "debounced_events_plus_interval_heartbeats"
    assert spec.test_mode_safe is True
    # Honesty: a probe package is never production_ready.
    assert spec.production_ready is False


def test_production_ready_requires_supported_status() -> None:
    base = resolve(_PKG)
    dishonest = PackageSpec(
        package_id="test.pkg.v0",
        display_name="t",
        runtime="deepstream",
        description="t",
        outputs=(),
        status="probe",
        model_id=base.model_id,
        production_ready=True,
    )
    with pytest.raises(RegistryValidationError, match="production_ready"):
        validate_package_compatibility(dishonest)


def test_invalid_setup_complexity_rejected() -> None:
    bad = PackageSpec(
        package_id="test.pkg.v0",
        display_name="t",
        runtime="deepstream",
        description="t",
        outputs=(),
        status="probe",
        setup_complexity="trivial",
    )
    with pytest.raises(RegistryValidationError, match="setup_complexity"):
        validate_package_compatibility(bad)


def test_all_packages_still_resolve() -> None:
    for package_id in (
        "mock.detect.v0",
        "deepstream.decode_probe.v0",
        "deepstream.inference_probe.v0",
        "deepstream.metadata_probe.v0",
        "deepstream.tracker_probe.v0",
        "deepstream.analytics_probe.v0",
        "deepstream.yolox_probe.v0",
        _PKG,
        "deepstream.yolox_analytics_nvdcf_probe.v0",  # P5-E
    ):
        assert resolve(package_id).package_id == package_id


# --- DeepStream config: YOLOX + tracker + analytics together ------------------


def test_yolox_analytics_pm_renders_all_three_configs(tmp_path: Path) -> None:
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
        analytics_enabled=True,
        model_store_dir=str(store),
    )
    work = pm._prepare_work_dir()
    pm._work_dir = work
    try:
        pgie = (work / "pgie.txt").read_text()
        assert f"onnx-file={model_dir}/yolox_s.onnx" in pgie  # YOLOX model config
        assert "NvDsInferParseYolox" in pgie
        # IOU tracker config byte-identical to the registry-backed builder.
        assert (work / "tracker_config.yml").read_text() == build_tracker_config("iou")
        analytics = (work / "analytics_config.txt").read_text()
        assert "[roi-filtering-stream-0]" in analytics
        assert "[line-crossing-stream-0]" in analytics
        labels = (work / "labels.txt").read_text().strip().split("\n")
        assert labels[0] == "person" and len(labels) == 80
        # No secrets / raw URLs in any rendered config.
        blob = pgie + analytics + (work / "tracker_config.yml").read_text()
        assert "rtsp://" not in blob
        argv = pm.command()
        assert f"{model_dir}:{model_dir}" in argv  # host artifacts mounted
        assert "--tracker-config-path" in " ".join(argv)
        assert "--analytics-config-path" in " ".join(argv)
    finally:
        pm._keep_temp_files = False
        pm._cleanup_work_dir()


def test_worker_builds_analytics_pm_for_yolox_package(fast_settings) -> None:  # type: ignore[no-untyped-def]
    worker = DeepStreamWorker(
        session=_session(internal_rtsp_url=_URL),
        settings=fast_settings,
        api=FakeApi(),
        package=resolve(_PKG),
    )
    assert worker._mode == "analytics_probe"
    pm = worker._default_pm_factory()
    assert isinstance(pm, DeepStreamMetadataProcessManager)
    assert pm._model == "yolox_s_v1"
    assert pm._tracker_enabled is True
    assert pm._analytics_enabled is True
    assert pm._model_store_dir == fast_settings.deepstream_model_store_dir


# --- lineage -------------------------------------------------------------------


def test_model_key_runtime_map_derived_from_registry() -> None:
    mapping = model_key_runtime_map()
    # Registry ids AND aliases, all statuses.
    assert mapping["trafficcamnet_v1"] == "deepstream"
    assert mapping["trafficcamnet"] == "deepstream"  # old alias still maps
    assert mapping["sample_trafficcamnet"] == "deepstream"
    assert mapping["yolox_s_v1"] == "deepstream"
    assert mapping["yolox_s"] == "deepstream"
    assert mapping["yolov8n_v1"] == "deepstream"  # planned still known
    # Static supplement: the mock model has no registry entry.
    assert mapping["mock_detect_v0"] == "mock"
    assert "definitely_unknown_v9" not in mapping


def test_yolox_worker_lineage_stamps_yolox_model_id(fast_settings) -> None:  # type: ignore[no-untyped-def]
    worker = DeepStreamWorker(
        session=_session(
            internal_rtsp_url=_URL,
            processing_config=_processing_envelope("trafficcamnet_v1"),
        ),
        settings=fast_settings,
        api=FakeApi(),
        package=resolve(_PKG),
        pm_factory=lambda: FakeMetadataPM([]),
    )
    resolved = worker._resolved_processing_config
    assert resolved.applied.model_key == "yolox_s_v1"
    # The operator asked for TrafficCamNet; the package runs YOLOX — that
    # divergence is honestly stamped, with a known runtime mapping.
    assert "model_switch_requested_not_applied" in resolved.warnings
    assert "model_key_no_runtime_mapping" not in resolved.warnings


def test_yolox_worker_lineage_clean_when_request_matches(fast_settings) -> None:  # type: ignore[no-untyped-def]
    worker = DeepStreamWorker(
        session=_session(
            internal_rtsp_url=_URL,
            processing_config=_processing_envelope("yolox_s_v1"),
        ),
        settings=fast_settings,
        api=FakeApi(),
        package=resolve(_PKG),
        pm_factory=lambda: FakeMetadataPM([]),
    )
    resolved = worker._resolved_processing_config
    assert resolved.applied.model_key == "yolox_s_v1"
    assert "model_switch_requested_not_applied" not in resolved.warnings


def test_trafficcamnet_lineage_unchanged(fast_settings) -> None:  # type: ignore[no-untyped-def]
    """Stability: the TrafficCamNet analytics package stamps exactly what it
    did before P5-D (applied trafficcamnet_v1, no model warning on a matching
    request, requested/applied tracker behavior untouched)."""
    worker = DeepStreamWorker(
        session=_session(
            internal_rtsp_url=_URL,
            processing_config=_processing_envelope("trafficcamnet_v1"),
        ),
        settings=fast_settings,
        api=FakeApi(),
        package=resolve("deepstream.analytics_probe.v0"),
        pm_factory=lambda: FakeMetadataPM([]),
    )
    resolved = worker._resolved_processing_config
    assert resolved.applied.model_key == "trafficcamnet_v1"
    assert resolved.applied.tracker == "iou"
    assert "model_switch_requested_not_applied" not in resolved.warnings


# --- status payload -------------------------------------------------------------


def test_status_payload_carries_component_ids_and_stays_bounded(fast_settings) -> None:  # type: ignore[no-untyped-def]
    worker = DeepStreamWorker(
        session=_session(internal_rtsp_url=_URL),
        settings=fast_settings,
        api=FakeApi(),
        package=resolve(_PKG),
        pm_factory=lambda: FakeMetadataPM([]),
    )
    payload = worker._metadata_payload(
        counters=_MetadataCounters(), queue_stats={}, last_app_summary=None
    )
    assert payload["model_id"] == "yolox_s_v1"
    assert payload["tracker_profile_id"] == "deepstream_iou_fast_v1"
    assert payload["analytics_profile_id"] == "deepstream_nvdsanalytics_probe_v1"
    # Bounded: scalars/small dicts only — no per-frame arrays, no URL/secret.
    blob = json.dumps(payload)
    assert "rtsp://" not in blob
    assert "test-secret-value-xyz" not in blob
    for value in payload.values():
        if isinstance(value, list):
            assert len(value) <= 20  # bounded samples only


# --- emit path with the YOLOX package --------------------------------------------


def _yolox_analytics_frame(
    *,
    frame_num: int,
    tracker_id: str | None,
    roi_labels: list[str] | None = None,
    roi_counts: dict[str, int] | None = None,
    line_crossing_cumulative: dict[str, int] | None = None,
) -> str:
    obj: dict[str, object] = {
        # YOLOX class ids: person == 0 (COCO), unlike TrafficCamNet's 2.
        "class_id": 0,
        "label": "person",
        "confidence": 0.91,
        "bbox_pixels": {"left": 100.0, "top": 100.0, "width": 200.0, "height": 400.0},
        "bbox_normalized": {"x": 0.078, "y": 0.139, "w": 0.156, "h": 0.556},
        "tracker_id": tracker_id,
    }
    if roi_labels is not None:
        obj["debug"] = {"analytics": {"roi_labels": roi_labels}}
    payload: dict[str, object] = {
        "runtime": "deepstream",
        "metadata_source": "pyds_pad_probe",
        "stream_session_id": None,
        "frame": {"frame_num": frame_num, "width": 1280, "height": 720},
        "objects": [obj],
    }
    analytics: dict[str, object] = {}
    if roi_counts is not None:
        analytics["roi_counts"] = roi_counts
    if line_crossing_cumulative is not None:
        analytics["line_crossing_cumulative"] = line_crossing_cumulative
    if analytics:
        payload["source"] = {"analytics": analytics}
    return "UPSTREAM_METADATA=" + json.dumps(payload)


async def test_yolox_analytics_emits_existing_types_only(fast_settings) -> None:  # type: ignore[no-untyped-def]
    """Full in-process run with the YOLOX analytics package: a tracked person
    enters the ROI (stability frames satisfied) and a line counter
    increments. Expect zone_entered + line_crossing using EXISTING types
    only, with the package id attached and lineage stamping yolox_s_v1."""
    api = FakeApi()
    frames = [
        # 3 stable in-ROI frames (fast_settings stability=1 commits fast);
        # the zone set comes from frame-level roi_counts, exactly as
        # nvdsanalytics provides live; the line counter increments once on
        # the second frame.
        _yolox_analytics_frame(
            frame_num=1,
            tracker_id="7",
            roi_labels=["entry-roi"],
            roi_counts={"entry-roi": 1},
        ),
        _yolox_analytics_frame(
            frame_num=2,
            tracker_id="7",
            roi_labels=["entry-roi"],
            roi_counts={"entry-roi": 1},
            line_crossing_cumulative={"entry-line": 1},
        ),
        _yolox_analytics_frame(
            frame_num=3,
            tracker_id="7",
            roi_labels=["entry-roi"],
            roi_counts={"entry-roi": 1},
        ),
    ]
    worker = DeepStreamWorker(
        session=_session(internal_rtsp_url=_URL),
        settings=fast_settings,
        api=api,
        package=resolve(_PKG),
        writer=ObservationWriter(api, "proc_test_001"),
        pm_factory=lambda: FakeMetadataPM(frames),
    )
    await asyncio.wait_for(worker.run(), timeout=5)
    assert worker.state is WorkerState.STOPPED

    types = [o.type for o in api.observations]
    assert "processor.zone_entered" in types
    assert "processor.line_crossing" in types
    # Existing, API-supported types only — P5-D adds no taxonomy.
    assert set(types) <= set(OBSERVATION_TYPES)
    assert set(types) <= set(resolve(_PKG).outputs)
    # Bounded: a 3-frame run emits a handful of events, never per-frame.
    assert len(api.observations) <= 10
    for obs in api.observations:
        assert obs.package_id == _PKG
        blob = json.dumps(obs.model_dump(mode="json"))
        assert _URL not in blob
        assert "test-secret-value-xyz" not in blob
    # Lineage rides every observation's source block.
    zone = next(o for o in api.observations if o.type == "processor.zone_entered")
    applied = zone.payload["source"]["applied_processing_config"]["applied"]
    assert applied["model_key"] == "yolox_s_v1"


async def test_yolox_analytics_without_writer_emits_nothing(fast_settings) -> None:  # type: ignore[no-untyped-def]
    api = FakeApi()
    worker = DeepStreamWorker(
        session=_session(internal_rtsp_url=_URL),
        settings=fast_settings,
        api=api,
        package=resolve(_PKG),
        pm_factory=lambda: FakeMetadataPM(
            [_yolox_analytics_frame(frame_num=1, tracker_id="7", roi_labels=["entry-roi"])]
        ),
    )
    await asyncio.wait_for(worker.run(), timeout=5)
    assert api.observations == []


# --- P5-D live-found fix: package-derived class scoping ------------------------


def test_package_classes_become_default_object_filter(fast_settings) -> None:  # type: ignore[no-untyped-def]
    """Live-found (P5-D run 1): without this, COCO-80 background objects
    (books/tv/chairs) flood the zone/dwell paths. The package's
    supported_classes are the default per-object filter; the analytics
    aggregate filter narrows to the model's class id for single-class
    packages."""
    worker = DeepStreamWorker(
        session=_session(internal_rtsp_url=_URL),
        settings=fast_settings,
        api=FakeApi(),
        package=resolve(_PKG),
        pm_factory=lambda: FakeMetadataPM([]),
    )
    assert worker._selected_classes == ("person",)
    assert worker._analytics_class_ids == "0"  # YOLOX person class id


def test_operator_selected_classes_still_win(fast_settings) -> None:  # type: ignore[no-untyped-def]
    worker = DeepStreamWorker(
        session=_session(
            internal_rtsp_url=_URL,
            processing_config={
                "revision": 4,
                "config": {
                    "version": 1,
                    "model_key": "yolox_s_v1",
                    "selected_classes": ["car"],
                },
            },
        ),
        settings=fast_settings,
        api=FakeApi(),
        package=resolve(_PKG),
        pm_factory=lambda: FakeMetadataPM([]),
    )
    assert worker._selected_classes == ("car",)  # operator narrows; package default yields


def test_trafficcamnet_class_scoping_unchanged(fast_settings) -> None:  # type: ignore[no-untyped-def]
    worker = DeepStreamWorker(
        session=_session(internal_rtsp_url=_URL),
        settings=fast_settings,
        api=FakeApi(),
        package=resolve("deepstream.analytics_probe.v0"),
        pm_factory=lambda: FakeMetadataPM([]),
    )
    # No supported_classes on the TrafficCamNet package -> no filter, and the
    # aggregate class filter stays at the env setting. Byte-identical configs.
    assert worker._selected_classes == ()
    assert worker._analytics_class_ids == fast_settings.deepstream_analytics_class_ids


def test_worker_pm_gets_package_scoped_analytics_class_id(fast_settings) -> None:  # type: ignore[no-untyped-def]
    worker = DeepStreamWorker(
        session=_session(internal_rtsp_url=_URL),
        settings=fast_settings,
        api=FakeApi(),
        package=resolve(_PKG),
    )
    pm = worker._default_pm_factory()
    assert isinstance(pm, DeepStreamMetadataProcessManager)
    assert pm._analytics_class_ids == "0"


async def test_background_objects_do_not_emit_zone_or_dwell(fast_settings) -> None:  # type: ignore[no-untyped-def]
    """A tracked 'book' inside the ROI emits NOTHING on the per-object
    paths with the YOLOX package; the person still does."""
    api = FakeApi()
    book = {
        "class_id": 73,
        "label": "book",
        "confidence": 0.31,
        "bbox_pixels": {"left": 900.0, "top": 500.0, "width": 60.0, "height": 80.0},
        "bbox_normalized": {"x": 0.70, "y": 0.69, "w": 0.05, "h": 0.11},
        "tracker_id": "42",
        "debug": {"analytics": {"roi_labels": ["entry-roi"]}},
    }
    person = {
        "class_id": 0,
        "label": "person",
        "confidence": 0.91,
        "bbox_pixels": {"left": 100.0, "top": 100.0, "width": 200.0, "height": 400.0},
        "bbox_normalized": {"x": 0.078, "y": 0.139, "w": 0.156, "h": 0.556},
        "tracker_id": "7",
        "debug": {"analytics": {"roi_labels": ["entry-roi"]}},
    }
    frames = [
        "UPSTREAM_METADATA="
        + json.dumps(
            {
                "runtime": "deepstream",
                "metadata_source": "pyds_pad_probe",
                "stream_session_id": None,
                "frame": {"frame_num": n, "width": 1280, "height": 720},
                "objects": [book, person],
                "source": {"analytics": {"roi_counts": {"entry-roi": 2}}},
            }
        )
        for n in (1, 2)
    ]
    worker = DeepStreamWorker(
        session=_session(internal_rtsp_url=_URL),
        settings=fast_settings,
        api=api,
        package=resolve(_PKG),
        writer=ObservationWriter(api, "proc_test_001"),
        pm_factory=lambda: FakeMetadataPM(frames),
    )
    await asyncio.wait_for(worker.run(), timeout=5)
    zone_tracks = [
        o.payload["track_id"]
        for o in api.observations
        if o.type in ("processor.zone_entered", "processor.zone_exited")
    ]
    dwell_tracks = [
        o.payload["track_id"] for o in api.observations if o.type.startswith("processor.dwell_")
    ]
    assert "7" in zone_tracks  # the person enters
    assert "42" not in zone_tracks  # the book never does
    assert "42" not in dwell_tracks
