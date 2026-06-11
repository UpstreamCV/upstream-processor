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
    load_analytics_registry,
    load_model_registry,
    load_tracker_registry,
    parse_analytics_registry,
    parse_model_registry,
    parse_tracker_registry,
)
from app.packages.resolver import PackageSpec, resolve
from app.runtimes.deepstream_runtime.inference import (
    build_tracker_config,
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
    # YOLO candidates exist but are placeholders only.
    for candidate in ("yolox_s_v1", "yolov8n_v1"):
        assert registry[candidate].status == "planned"
    # The AGPL candidate carries an explicit license-review flag.
    assert "REVIEW REQUIRED" in registry["yolov8n_v1"].license_notes


def test_tracker_registry_loads_with_expected_entries() -> None:
    registry = load_tracker_registry()
    iou = registry["deepstream_iou_fast_v1"]
    assert iou.status == "probe"  # the only live-verified tracker
    assert iou.algorithm == "iou"
    assert iou.settings  # authored, runnable config
    # NvDCF / NvSORT are planned-only: no authored settings, never runnable.
    for planned in ("deepstream_nvdcf_balanced_v1", "deepstream_nvsort_fast_v1"):
        assert registry[planned].status == "planned"
        assert registry[planned].settings == {}


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
)


@pytest.mark.parametrize("package_id", _ALL_PACKAGES)
def test_every_shipped_package_resolves(package_id: str) -> None:
    spec = resolve(package_id)
    assert spec.package_id == package_id
    assert spec.status == "probe"
    assert spec.pricing_tier == "free_probe"


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
    # Even a probe package cannot bundle a planned (artifact-less) model
    # or a planned tracker profile.
    with pytest.raises(RegistryValidationError, match="status 'planned'"):
        validate_package_compatibility(_pkg(model_id="yolox_s_v1"))
    with pytest.raises(RegistryValidationError, match="status 'planned'"):
        validate_package_compatibility(
            _pkg(
                model_id="trafficcamnet_v1",
                tracker_profile_id="deepstream_nvdcf_balanced_v1",
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
    for algo in ("nvdcf", "nvsort", "definitely-not-real"):
        with pytest.raises(ValueError, match="unknown deepstream tracker algorithm"):
            build_tracker_config(algo)


def test_model_dir_resolution_identical_to_pre_p5() -> None:
    assert resolve_model_dir("trafficcamnet") == _PRE_P5_MODEL_DIR
    assert resolve_model_dir("sample_trafficcamnet") == _PRE_P5_MODEL_DIR
    assert resolve_model_dir("TrafficCamNet") == _PRE_P5_MODEL_DIR  # case-insensitive, as before
    assert resolve_model_dir("trafficcamnet_v1") == _PRE_P5_MODEL_DIR  # registry id also maps


def test_planned_model_is_not_runnable() -> None:
    with pytest.raises(ValueError, match="not runnable"):
        resolve_model_dir("yolox_s_v1")
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
