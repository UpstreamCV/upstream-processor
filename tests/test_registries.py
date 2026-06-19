"""Processing P5 — model / tracker / analytics registry + compatibility tests.

Three groups:

1. Registry loading/validation (pure ``parse_*`` functions on crafted dicts:
   duplicates, missing fields, bad enums).
2. Package ↔ registry compatibility (every shipped package resolves; invalid
   combinations are rejected with ``RegistryValidationError``).
3. Config-hook equivalence goldens: the registry-resolved TrafficCamNet model
   dir, IOU tracker YAML, and analytics capability flags are byte/value-
   identical to the pre-P5 hardcoded behavior.
"""

from __future__ import annotations

import dataclasses
import json
from typing import Any

import pytest
from app.errors import PackageResolutionError, RegistryValidationError
from app.observations.schemas import OBSERVATION_TYPES
from app.packages.compatibility import (
    OUTPUTS_REQUIRING_TRACKER,
    validate_package_compatibility,
)
from app.packages.registries import (
    find_runnable_deepstream_tracker_by_profile_id,
    load_analytics_registry,
    load_model_registry,
    load_tracker_registry,
    parse_analytics_registry,
    parse_model_registry,
    parse_tracker_registry,
    runnable_deepstream_tracker_algorithms,
)
from app.packages.resolver import PackageSpec, resolve
from app.runtimes.deepstream_runtime.inference import (
    build_tracker_config,
    render_tracker_ll_config,
    resolve_model_dir,
)

# --- fixtures: minimal valid registry entries ------------------------------


def _model_entry(**overrides: Any) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "model_id": "test_model_v1",
        "display_name": "Test Model",
        "family": "test",
        "task": "object_detection",
        "version": "v1",
        "status": "probe",
        "license": "Apache-2.0",
        "license_notes": "test",
        "license_class": "permissive",
        "first_party": True,
        "source": "test",
        "supported_classes": ["person", "car"],
        "class_id_map": {"person": 0, "car": 1},
        "runtime_support": ["deepstream_nvinfer"],
        "aliases": [],
        "deepstream": None,
        "input": {"width": 640, "height": 640, "channels": 3},
        "default_confidence_threshold": 0.25,
        "recommended_confidence_threshold": 0.4,
        "compute_tier": "low",
        "pricing_tier": "free_probe",
        "default_target_fps": 30,
        "recommended_resolution": "1280x720",
        "gpu_required": True,
        "estimated_cost_weight": 1.0,
        "observation_volume_risk": "low",
        "notes": "test",
    }
    entry.update(overrides)
    return entry


def _tracker_entry(**overrides: Any) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "tracker_profile_id": "test_tracker_v1",
        "display_name": "Test Tracker",
        "runtime": "deepstream",
        "family": "test",
        "status": "probe",
        "algorithm": "iou",
        "requires_reid": False,
        "requires_model": None,
        "deepstream": {},
        "best_for": [],
        "not_recommended_for": [],
        "compute_tier": "low",
        "estimated_cost_weight": 0.2,
        "settings": {"BaseConfig": {"minDetectorConfidence": 0}},
        "notes": "test",
    }
    entry.update(overrides)
    return entry


def _analytics_entry(**overrides: Any) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "analytics_profile_id": "test_analytics_v1",
        "display_name": "Test Analytics",
        "runtime": "deepstream",
        "status": "probe",
        "supported_outputs": ["processor.line_crossing"],
        "supported_shapes": ["polygon_roi", "line_segment"],
        "requires_tracker": True,
        "supports_line_crossing": True,
        "supports_roi_occupancy": True,
        "supports_zone_transitions": True,
        "supports_dwell": True,
        "supports_direction": False,
        "config_mode": "generated_ini",
        "compute_tier": "low",
        "estimated_cost_weight": 0.3,
        "observation_volume_risk": "medium",
        "notes": "test",
    }
    entry.update(overrides)
    return entry


# --- registry loading -------------------------------------------------------


def test_model_registry_loads_with_expected_entries() -> None:
    registry = load_model_registry()
    assert "trafficcamnet_v1" in registry
    tcn = registry["trafficcamnet_v1"]
    assert tcn.status == "probe"  # the only live-verified model is a probe
    assert tcn.task == "object_detection"
    assert "deepstream_nvinfer" in tcn.runtime_support
    assert set(tcn.aliases) == {"trafficcamnet", "sample_trafficcamnet"}
    assert tcn.supported_classes == ("car", "bicycle", "person", "roadsign")
    # P5-B: YOLOX-S is wired (runnable) but not yet 'supported'; the AGPL
    # candidate stays a planned placeholder with its license-review flag.
    yolox = registry["yolox_s_v1"]
    assert yolox.status in ("experimental", "probe")
    assert yolox.status != "supported"  # never promoted prematurely
    assert "Apache-2.0" in yolox.license_notes
    assert yolox.deepstream is not None
    assert yolox.deepstream.requires_custom_parser is True
    assert yolox.deepstream.artifact_subdir == "yolox_s_v1"
    assert len(yolox.supported_classes) == 80  # full COCO-80, in id order
    assert yolox.class_id_map["person"] == 0
    # P5-K licensing/provenance axis: YOLOX is permissive + first-party.
    assert yolox.license_class == "permissive"
    assert yolox.first_party is True
    assert tcn.license_class == "commercial_agreement" and tcn.first_party is True
    # P5-K: YOLOv8n flipped planned -> experimental now that the Ultralytics
    # license is held; it stays restrictive / BYOM-only (first_party=false).
    yolov8n = registry["yolov8n_v1"]
    assert yolov8n.status == "experimental"
    assert yolov8n.license_class == "restrictive"
    assert yolov8n.first_party is False
    assert "AGPL" in yolov8n.license_notes
    assert len(yolov8n.supported_classes) == 80  # completed to full COCO-80
    # P5-K new model families (in-repo prep, all experimental, none probe yet).
    for mid in ("yolox_nano_v1", "yolox_tiny_v1", "yolox_m_v1"):
        assert registry[mid].status == "experimental"
        assert registry[mid].license_class == "permissive"
        assert registry[mid].first_party is True
    for mid in ("yolo11n_v1", "yolo11s_v1", "yolo11m_v1"):
        assert registry[mid].license_class == "restrictive"
        assert registry[mid].first_party is False  # AGPL -> BYOM-only
    assert registry["peoplenet_v1"].license_class == "commercial_agreement"
    assert registry["peoplenet_v1"].first_party is True


def test_p5l_model_metrics_are_published_reference() -> None:
    reg = load_model_registry()
    # Every detector carries curated published-reference metrics; none claim to
    # be Upstream-measured (no GPU benchmarking).
    for mid, m in reg.items():
        assert m.metrics is not None, mid
        assert m.metrics.upstream_measured is False, mid
        assert m.metrics.metric_source  # provenance is required
    # COCO-trained YOLO detectors carry COCO mAP + params.
    yolox_s = reg["yolox_s_v1"].metrics
    assert yolox_s is not None
    assert yolox_s.coco_map50_95 == 40.5
    assert yolox_s.params_m == 9.0
    assert reg["yolo11s_v1"].metrics.coco_map50_95 == 47.0  # type: ignore[union-attr]
    # NVIDIA TAO models have NO COCO mAP; their vendor KPI is separate + labelled.
    pn = reg["peoplenet_v1"].metrics
    assert pn is not None
    assert pn.coco_map50_95 is None
    assert pn.vendor_kpi_value == 94.19
    assert pn.vendor_kpi_label and "not comparable" in (pn.vendor_kpi_note or "")
    assert reg["trafficcamnet_v1"].metrics.coco_map50_95 is None  # type: ignore[union-attr]


def test_model_metrics_optional_and_strict() -> None:
    # metrics is optional (None when absent) but, when present, requires
    # metric_source + upstream_measured.
    assert parse_model_registry({"models": [_model_entry()]})["test_model_v1"].metrics is None
    with_metrics = _model_entry(metrics={"metric_source": "x", "upstream_measured": False})
    spec = parse_model_registry({"models": [with_metrics]})["test_model_v1"]
    assert spec.metrics is not None and spec.metrics.coco_map50_95 is None
    bad = _model_entry(metrics={"coco_map50_95": 40.0})  # missing required metric_source
    with pytest.raises(RegistryValidationError, match="metric_source"):
        parse_model_registry({"models": [bad]})


def test_tracker_registry_loads_with_expected_entries() -> None:
    registry = load_tracker_registry()
    iou = registry["deepstream_iou_fast_v1"]
    assert iou.status == "probe"  # the only LIVE-verified tracker
    assert iou.algorithm == "iou"
    assert iou.settings  # authored, runnable config
    # P5-E: NvDCF is now authored + runnable (experimental until live A/B), but
    # NOT yet probe — it has authored ll-config settings and a runnable status.
    nvdcf = registry["deepstream_nvdcf_balanced_v1"]
    assert nvdcf.status == "experimental"
    assert nvdcf.algorithm == "nvdcf"
    assert nvdcf.settings  # authored ll-config
    assert nvdcf.requires_reid is False  # ReID-free perf profile
    # NvSORT is still planned-only: no authored settings, never runnable.
    nvsort = registry["deepstream_nvsort_fast_v1"]
    assert nvsort.status == "planned"
    assert nvsort.settings == {}


def test_analytics_registry_loads_and_outputs_are_known_types() -> None:
    registry = load_analytics_registry()
    profile = registry["deepstream_nvdsanalytics_probe_v1"]
    assert profile.status == "probe"
    assert profile.requires_tracker is True
    # Every advertised output is a real, already-whitelisted observation
    # type — P5 emits no new observation types.
    assert set(profile.supported_outputs) <= set(OBSERVATION_TYPES)


def test_duplicate_ids_rejected() -> None:
    with pytest.raises(RegistryValidationError, match="duplicate model_id"):
        parse_model_registry({"models": [_model_entry(), _model_entry()]})
    with pytest.raises(RegistryValidationError, match="duplicate tracker_profile_id"):
        parse_tracker_registry({"trackers": [_tracker_entry(), _tracker_entry()]})
    with pytest.raises(RegistryValidationError, match="duplicate analytics_profile_id"):
        parse_analytics_registry({"analytics_profiles": [_analytics_entry(), _analytics_entry()]})


def test_duplicate_model_alias_rejected() -> None:
    data = {
        "models": [
            _model_entry(aliases=["shared"]),
            _model_entry(model_id="other_model_v1", aliases=["shared"]),
        ]
    }
    with pytest.raises(RegistryValidationError, match="alias 'shared' already claimed"):
        parse_model_registry(data)


def test_missing_required_fields_rejected() -> None:
    bad_model = _model_entry()
    del bad_model["display_name"]
    with pytest.raises(RegistryValidationError, match="display_name"):
        parse_model_registry({"models": [bad_model]})
    bad_tracker = _tracker_entry()
    del bad_tracker["algorithm"]
    with pytest.raises(RegistryValidationError, match="algorithm"):
        parse_tracker_registry({"trackers": [bad_tracker]})
    bad_analytics = _analytics_entry()
    del bad_analytics["requires_tracker"]
    with pytest.raises(RegistryValidationError, match="requires_tracker"):
        parse_analytics_registry({"analytics_profiles": [bad_analytics]})


def test_invalid_enum_values_rejected() -> None:
    with pytest.raises(RegistryValidationError, match="invalid status"):
        parse_model_registry({"models": [_model_entry(status="shipped")]})
    with pytest.raises(RegistryValidationError, match="invalid task"):
        parse_model_registry({"models": [_model_entry(task="segmentation")]})
    with pytest.raises(RegistryValidationError, match="invalid runtime_support"):
        parse_model_registry({"models": [_model_entry(runtime_support=["tensorflow"])]})
    with pytest.raises(RegistryValidationError, match="invalid compute_tier"):
        parse_model_registry({"models": [_model_entry(compute_tier="huge")]})
    with pytest.raises(RegistryValidationError, match="invalid pricing_tier"):
        parse_model_registry({"models": [_model_entry(pricing_tier="gold")]})
    with pytest.raises(RegistryValidationError, match="invalid runtime"):
        parse_tracker_registry({"trackers": [_tracker_entry(runtime="cpp")]})
    with pytest.raises(RegistryValidationError, match="invalid config_mode"):
        parse_analytics_registry(
            {"analytics_profiles": [_analytics_entry(config_mode="hand_authored")]}
        )


def test_shipped_registry_files_are_valid_json_and_parse() -> None:
    # The cached loaders parse the real packaged files without raising.
    assert load_model_registry()
    assert load_tracker_registry()
    assert load_analytics_registry()


# --- package ↔ registry compatibility ---------------------------------------

_ALL_PACKAGES = (
    "mock.detect.v0",
    "deepstream.decode_probe.v0",
    "deepstream.inference_probe.v0",
    "deepstream.metadata_probe.v0",
    "deepstream.tracker_probe.v0",
    "deepstream.analytics_probe.v0",
    "deepstream.yolox_probe.v0",
    "deepstream.yolox_analytics_probe.v0",
    # P5-K: YOLOX-S + IOU people variants reuse the live-verified pipeline.
    "deepstream.yolox_people_fast_probe.v0",
    "deepstream.yolox_people_high_recall_probe.v0",
    "deepstream.yolox_people_occupancy_probe.v0",
    "deepstream.yolox_people_dwell_probe.v0",
)

# P5-E/P5-K: experimental packages resolve too, but carry a non-probe status
# (runnable-but-unverified). P5-K adds the line-crossing variant (line crossing
# not live-confirmed), the TrafficCamNet vehicle package, and one
# people-analytics package per new model (all experimental — no GPU verify).
_EXPERIMENTAL_PACKAGES = (
    "deepstream.yolox_analytics_nvdcf_probe.v0",
    "deepstream.yolox_people_line_crossing_probe.v0",
    "deepstream.trafficcamnet_vehicle_probe.v0",
    "deepstream.yolox_nano_people_probe.v0",
    "deepstream.yolox_tiny_people_probe.v0",
    "deepstream.yolox_m_people_probe.v0",
    "deepstream.yolo11n_people_probe.v0",
    "deepstream.yolo11s_people_probe.v0",
    "deepstream.yolo11m_people_probe.v0",
    "deepstream.yolov8n_people_probe.v0",
    "deepstream.peoplenet_people_probe.v0",
)


@pytest.mark.parametrize("package_id", _ALL_PACKAGES)
def test_every_shipped_package_resolves(package_id: str) -> None:
    spec = resolve(package_id)
    assert spec.package_id == package_id
    assert spec.status == "probe"
    assert spec.pricing_tier == "free_probe"


@pytest.mark.parametrize("package_id", _EXPERIMENTAL_PACKAGES)
def test_experimental_packages_resolve_but_are_not_probe(package_id: str) -> None:
    spec = resolve(package_id)
    assert spec.package_id == package_id
    assert spec.status == "experimental"
    assert spec.pricing_tier == "free_probe"
    assert spec.production_ready is False


def test_shipped_package_registry_references() -> None:
    assert resolve("mock.detect.v0").model_id is None
    assert resolve("deepstream.decode_probe.v0").model_id is None
    assert resolve("deepstream.inference_probe.v0").model_id == "trafficcamnet_v1"
    tracker_pkg = resolve("deepstream.tracker_probe.v0")
    assert tracker_pkg.model_id == "trafficcamnet_v1"
    assert tracker_pkg.tracker_profile_id == "deepstream_iou_fast_v1"
    analytics_pkg = resolve("deepstream.analytics_probe.v0")
    assert analytics_pkg.model_id == "trafficcamnet_v1"
    assert analytics_pkg.tracker_profile_id == "deepstream_iou_fast_v1"
    assert analytics_pkg.analytics_profile_id == "deepstream_nvdsanalytics_probe_v1"
    # P5-E: the NvDCF variant sits beside the IOU YOLOX analytics package and
    # differs ONLY in the tracker profile.
    iou_yolox = resolve("deepstream.yolox_analytics_probe.v0")
    nvdcf_yolox = resolve("deepstream.yolox_analytics_nvdcf_probe.v0")
    assert iou_yolox.tracker_profile_id == "deepstream_iou_fast_v1"
    assert nvdcf_yolox.tracker_profile_id == "deepstream_nvdcf_balanced_v1"
    assert nvdcf_yolox.model_id == iou_yolox.model_id == "yolox_s_v1"
    assert nvdcf_yolox.analytics_profile_id == iou_yolox.analytics_profile_id
    assert nvdcf_yolox.outputs == iou_yolox.outputs
    assert nvdcf_yolox.supported_classes == ("person",)


def test_p5k_variant_packages_reference_yolox_s_and_narrow_honestly() -> None:
    profile = load_analytics_registry()["deepstream_nvdsanalytics_probe_v1"]
    # The 5 use-case variants all run the verified YOLOX-S + IOU pipeline.
    for pid in (
        "deepstream.yolox_people_fast_probe.v0",
        "deepstream.yolox_people_high_recall_probe.v0",
        "deepstream.yolox_people_line_crossing_probe.v0",
        "deepstream.yolox_people_occupancy_probe.v0",
        "deepstream.yolox_people_dwell_probe.v0",
    ):
        spec = resolve(pid)
        assert spec.model_id == "yolox_s_v1"
        assert spec.tracker_profile_id == "deepstream_iou_fast_v1"
        # Narrowed outputs stay a subset of the analytics profile.
        assert set(spec.outputs) <= set(profile.supported_outputs)
    # Line crossing is the one variant whose primary trigger is unverified.
    assert resolve("deepstream.yolox_people_line_crossing_probe.v0").status == "experimental"
    assert resolve("deepstream.yolox_people_line_crossing_probe.v0").supported_shapes == (
        "line_segment",
    )
    assert resolve("deepstream.yolox_people_occupancy_probe.v0").status == "probe"


def test_p5k_per_model_packages_are_experimental_and_match_provenance() -> None:
    models = load_model_registry()
    # package_id -> (model_id, expected first_party of the backing model)
    cases = {
        "deepstream.yolox_nano_people_probe.v0": ("yolox_nano_v1", True),
        "deepstream.yolo11s_people_probe.v0": ("yolo11s_v1", False),
        "deepstream.yolov8n_people_probe.v0": ("yolov8n_v1", False),
        "deepstream.peoplenet_people_probe.v0": ("peoplenet_v1", True),
        "deepstream.trafficcamnet_vehicle_probe.v0": ("trafficcamnet_v1", True),
    }
    for pid, (model_id, first_party) in cases.items():
        spec = resolve(pid)
        assert spec.status == "experimental"  # no GPU verification this milestone
        assert spec.model_id == model_id
        assert models[model_id].first_party is first_party


def _pkg(**overrides: Any) -> PackageSpec:
    base: dict[str, Any] = {
        "package_id": "test.pkg.v0",
        "display_name": "Test",
        "runtime": "deepstream",
        "description": "test",
        "outputs": (),
        "supports": {},
        "status": "probe",
    }
    base.update(overrides)
    return PackageSpec(**base)


def test_unknown_model_id_fails() -> None:
    with pytest.raises(RegistryValidationError, match="unknown model_id"):
        validate_package_compatibility(_pkg(model_id="does_not_exist_v1"))


def test_unknown_tracker_profile_id_fails() -> None:
    with pytest.raises(RegistryValidationError, match="unknown tracker_profile_id"):
        validate_package_compatibility(
            _pkg(model_id="trafficcamnet_v1", tracker_profile_id="does_not_exist_v1")
        )


def test_unknown_analytics_profile_id_fails() -> None:
    with pytest.raises(RegistryValidationError, match="unknown analytics_profile_id"):
        validate_package_compatibility(
            _pkg(
                model_id="trafficcamnet_v1",
                tracker_profile_id="deepstream_iou_fast_v1",
                analytics_profile_id="does_not_exist_v1",
            )
        )


def test_mock_runtime_cannot_reference_a_model() -> None:
    with pytest.raises(RegistryValidationError, match="cannot reference a model"):
        validate_package_compatibility(_pkg(runtime="mock", model_id="trafficcamnet_v1"))


def test_model_without_required_runtime_support_fails() -> None:
    models = dict(load_model_registry())
    tcn = models["trafficcamnet_v1"]
    models["trafficcamnet_v1"] = dataclasses.replace(tcn, runtime_support=("python",))
    with pytest.raises(RegistryValidationError, match="requires model runtime support"):
        validate_package_compatibility(_pkg(model_id="trafficcamnet_v1"), models=models)


def test_tracker_for_wrong_runtime_fails() -> None:
    trackers = dict(load_tracker_registry())
    iou = trackers["deepstream_iou_fast_v1"]
    trackers["deepstream_iou_fast_v1"] = dataclasses.replace(iou, runtime="python")
    with pytest.raises(RegistryValidationError, match="for runtime 'python'"):
        validate_package_compatibility(
            _pkg(model_id="trafficcamnet_v1", tracker_profile_id="deepstream_iou_fast_v1"),
            trackers=trackers,
        )


def test_package_class_not_supported_by_model_fails() -> None:
    with pytest.raises(RegistryValidationError, match="not supported by"):
        validate_package_compatibility(
            _pkg(model_id="trafficcamnet_v1", supported_classes=("person", "forklift"))
        )


def test_supported_classes_without_model_fails() -> None:
    with pytest.raises(RegistryValidationError, match="references no model_id"):
        validate_package_compatibility(_pkg(supported_classes=("person",)))


def test_output_requiring_tracker_fails_without_tracker() -> None:
    assert "processor.zone_entered" in OUTPUTS_REQUIRING_TRACKER
    with pytest.raises(RegistryValidationError, match="per-track outputs"):
        validate_package_compatibility(
            _pkg(model_id="trafficcamnet_v1", outputs=("processor.zone_entered",))
        )


def test_analytics_requires_tracker_fails_without_tracker() -> None:
    with pytest.raises(RegistryValidationError, match="requires a tracker"):
        validate_package_compatibility(
            _pkg(
                model_id="trafficcamnet_v1",
                analytics_profile_id="deepstream_nvdsanalytics_probe_v1",
            )
        )


def test_output_not_supported_by_analytics_profile_fails() -> None:
    with pytest.raises(RegistryValidationError, match="not supported by"):
        validate_package_compatibility(
            _pkg(
                model_id="trafficcamnet_v1",
                tracker_profile_id="deepstream_iou_fast_v1",
                analytics_profile_id="deepstream_nvdsanalytics_probe_v1",
                outputs=("processor.mock_detection",),
            )
        )


def test_supported_package_cannot_use_probe_components() -> None:
    with pytest.raises(RegistryValidationError, match="status 'supported'"):
        validate_package_compatibility(_pkg(status="supported", model_id="trafficcamnet_v1"))


def test_probe_package_may_use_probe_components() -> None:
    validate_package_compatibility(
        _pkg(
            model_id="trafficcamnet_v1",
            tracker_profile_id="deepstream_iou_fast_v1",
            analytics_profile_id="deepstream_nvdsanalytics_probe_v1",
        )
    )  # does not raise


def test_planned_components_are_never_referencable() -> None:
    # Even a probe package cannot bundle a planned (artifact-less) model. P5-K
    # flipped the last shipped 'planned' model (yolov8n) to experimental, so
    # inject a planned model to exercise the gate.
    planned_models = dict(load_model_registry())
    planned_models["planned_only_v1"] = dataclasses.replace(
        planned_models["yolox_s_v1"], model_id="planned_only_v1", status="planned"
    )
    with pytest.raises(RegistryValidationError, match="status 'planned'"):
        validate_package_compatibility(
            _pkg(model_id="planned_only_v1"), models=planned_models
        )
    # P5-E: NvDCF is no longer planned (it is referencable); NvSORT is still
    # the planned, artifact-less tracker that must never be bundled.
    with pytest.raises(RegistryValidationError, match="status 'planned'"):
        validate_package_compatibility(
            _pkg(
                model_id="trafficcamnet_v1",
                tracker_profile_id="deepstream_nvsort_fast_v1",
            )
        )


def test_invalid_package_tier_metadata_fails() -> None:
    with pytest.raises(RegistryValidationError, match="invalid compute_tier"):
        validate_package_compatibility(_pkg(compute_tier="enormous"))
    with pytest.raises(RegistryValidationError, match="invalid pricing_tier"):
        validate_package_compatibility(_pkg(pricing_tier="platinum"))
    with pytest.raises(RegistryValidationError, match="invalid status"):
        validate_package_compatibility(_pkg(status="ga"))


def test_registry_validation_error_is_a_package_resolution_error() -> None:
    # The supervisor's existing resolution-failure path (post `failed` +
    # quarantine) must catch compatibility failures unchanged.
    assert issubclass(RegistryValidationError, PackageResolutionError)


# --- config-hook equivalence goldens -----------------------------------------

# The exact YAML body the pre-P5 hardcoded ``build_tracker_config("iou")``
# produced (P3, live-verified). The registry-backed builder must reproduce
# it byte-for-byte.
_PRE_P5_IOU_YAML = (
    "%YAML:1.0\n"
    "BaseConfig:\n"
    "  minDetectorConfidence: 0\n"
    "TargetManagement:\n"
    "  preserveStreamUpdateOrder: 0\n"
    "  maxTargetsPerStream: 150\n"
    "  minIouDiff4NewTarget: 0.5\n"
    "  probationAge: 4\n"
    "  maxShadowTrackingAge: 38\n"
    "  earlyTerminationAge: 1\n"
    "TrajectoryManagement:\n"
    "  useUniqueID: 0\n"
    "DataAssociator:\n"
    "  dataAssociatorType: 0\n"
    "  associationMatcherType: 0\n"
    "  checkClassMatch: 1\n"
    "  minMatchingScore4Overall: 0.0\n"
    "  minMatchingScore4SizeSimilarity: 0.0\n"
    "  minMatchingScore4Iou: 0.0\n"
    "  matchingScoreWeight4SizeSimilarity: 0.4\n"
    "  matchingScoreWeight4Iou: 0.6\n"
)

_PRE_P5_MODEL_DIR = "/opt/nvidia/deepstream/deepstream/samples/models/Primary_Detector"


def test_iou_tracker_config_is_byte_identical_to_pre_p5() -> None:
    assert build_tracker_config("iou") == _PRE_P5_IOU_YAML
    assert build_tracker_config() == _PRE_P5_IOU_YAML  # default unchanged


def test_planned_tracker_algorithms_rejected_like_unknown() -> None:
    # P5-E: nvdcf is now runnable, so only nvsort (still planned) and unknown
    # names are rejected like unknown algorithms.
    for algo in ("nvsort", "definitely-not-real"):
        with pytest.raises(ValueError, match="unknown deepstream tracker algorithm"):
            build_tracker_config(algo)


def test_nvdcf_is_runnable_and_renders_required_sections() -> None:
    # P5-E: nvdcf is now a runnable algorithm alongside iou.
    assert runnable_deepstream_tracker_algorithms() == ["iou", "nvdcf"]
    body = build_tracker_config("nvdcf")
    # nvtracker requires the %YAML:1.0 marker; the NvDCF perf profile adds the
    # StateEstimator + VisualTracker sections on top of the IOU-style base.
    assert body.startswith("%YAML:1.0\n")
    # P5-F: reconciled to the DS 8.0 config_tracker_NvDCF_perf.yml section set.
    for section in (
        "BaseConfig:",
        "TargetManagement:",
        "TrajectoryManagement:",
        "DataAssociator:",
        "StateEstimator:",
        "VisualTracker:",
    ):
        assert section in body
    assert "visualTrackerType: 2" in body  # NvDCF_VPI, per the DS 8.0 perf sample
    # ReID-free: the perf sample has no ReID section / model dependency.
    assert "ReID:" not in body
    assert "reidType" not in body
    assert "\t" not in body  # YAML is whitespace-sensitive


def test_nvdcf_tracker_config_renders_registry_settings_verbatim() -> None:
    # The builder is generic/data-driven: build_tracker_config("nvdcf") must be
    # exactly render_tracker_ll_config of the registry profile's settings (no
    # per-algorithm branching). This pins the rendering without committing to a
    # fragile hand-authored golden before live verification.
    nvdcf = load_tracker_registry()["deepstream_nvdcf_balanced_v1"]
    assert build_tracker_config("nvdcf") == render_tracker_ll_config(nvdcf.settings)


def test_nvdcf_resolvable_by_profile_id() -> None:
    spec = find_runnable_deepstream_tracker_by_profile_id("deepstream_nvdcf_balanced_v1")
    assert spec is not None
    assert spec.algorithm == "nvdcf"
    # NvSORT is planned (no settings) → not runnable by profile id either.
    assert find_runnable_deepstream_tracker_by_profile_id("deepstream_nvsort_fast_v1") is None
    assert find_runnable_deepstream_tracker_by_profile_id("nope_v1") is None


def test_model_dir_resolution_identical_to_pre_p5() -> None:
    assert resolve_model_dir("trafficcamnet") == _PRE_P5_MODEL_DIR
    assert resolve_model_dir("sample_trafficcamnet") == _PRE_P5_MODEL_DIR
    assert resolve_model_dir("TrafficCamNet") == _PRE_P5_MODEL_DIR  # case-insensitive, as before
    assert resolve_model_dir("trafficcamnet_v1") == _PRE_P5_MODEL_DIR  # registry id also maps


def test_flipped_yolov8n_is_runnable_unknown_still_rejected() -> None:
    # P5-K: yolov8n flipped planned -> experimental (Ultralytics license held),
    # so it now resolves to its host-mounted artifact dir; unknown models still
    # raise. (No shipped model is 'planned' after P5-K.)
    assert resolve_model_dir("yolov8n_v1") == "/opt/upstream/models/yolov8n_v1"
    with pytest.raises(ValueError, match="unknown deepstream inference model"):
        resolve_model_dir("yolo-custom")


def test_analytics_profile_flags_match_live_worker_capabilities() -> None:
    """The registry entry must describe exactly what the analytics worker
    does today: line crossing + ROI occupancy + zone transitions + dwell
    (all live-verified or smoke-tested), NO strict direction support, and a
    hard tracker requirement (nvdsanalytics operates on tracked objects —
    the process manager enforces it)."""
    profile = load_analytics_registry()["deepstream_nvdsanalytics_probe_v1"]
    assert profile.supports_line_crossing is True
    assert profile.supports_roi_occupancy is True
    assert profile.supports_zone_transitions is True
    assert profile.supports_dwell is True
    assert profile.supports_direction is False
    assert profile.requires_tracker is True
    assert set(profile.supported_outputs) == {
        "processor.line_crossing",
        "processor.line_crossing_object",
        "processor.roi_occupancy",
        "processor.zone_entered",
        "processor.zone_exited",
        "processor.dwell_started",
        "processor.dwell_updated",
        "processor.dwell_exceeded",
    }


def test_iou_registry_settings_round_trip_through_json() -> None:
    """Belt-and-braces: the registry file's JSON scalars must render with
    the exact formatting nvtracker was verified with (ints without decimal
    points, floats with them)."""
    iou = load_tracker_registry()["deepstream_iou_fast_v1"]
    rendered = json.dumps(iou.settings)
    assert '"minMatchingScore4Overall": 0.0' in rendered
    assert '"maxTargetsPerStream": 150' in rendered
