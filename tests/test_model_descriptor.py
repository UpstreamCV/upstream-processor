"""BYOM Phase 0 — Model Descriptor contract tests.

Covers: every registered runnable DeepStream model normalizes into a valid
descriptor; no package points at a model whose descriptor decoder is unknown;
descriptor validation catches missing/invalid required fields; and the
descriptor's *declared* preprocessing/output agrees with what the decoder
actually emits (so the declarative view and the byte-template can't silently
diverge). No GPU / Docker — pure functions.
"""

from __future__ import annotations

import pytest
from app.packages.model_descriptor import (
    DescriptorArtifact,
    DescriptorInput,
    DescriptorOutput,
    DescriptorProvenance,
    DescriptorValidationError,
    ModelDescriptor,
    descriptor_from_spec,
)
from app.packages.registries import (
    RUNNABLE_COMPONENT_STATUSES,
    ModelSpec,
    find_model_by_alias,
    load_model_registry,
)
from app.packages.resolver import load_registry
from app.runtimes.deepstream_runtime.decoders import DECODER_REGISTRY
from app.runtimes.deepstream_runtime.inference import build_model_runtime_plan


def _runnable_deepstream_specs() -> list[ModelSpec]:
    return [
        s
        for s in load_model_registry().values()
        if s.status in RUNNABLE_COMPONENT_STATUSES and s.deepstream is not None
    ]


def _spec(name: str) -> ModelSpec:
    spec = find_model_by_alias(name)
    assert spec is not None, f"model {name!r} not in registry"
    return spec


# --- coverage: every model / every package resolves ---------------------


def test_every_runnable_deepstream_model_builds_a_valid_descriptor() -> None:
    specs = _runnable_deepstream_specs()
    assert specs, "registry has no runnable deepstream models"
    for spec in specs:
        assert spec.deepstream is not None
        d = descriptor_from_spec(spec)
        assert d.model_id == spec.model_id
        assert d.task == "detect"
        assert d.decoder == spec.deepstream.config_template
        assert d.decoder in DECODER_REGISTRY
        assert d.class_map == spec.class_id_map
        assert d.input.width == spec.input["width"]
        assert d.input.height == spec.input["height"]
        assert d.provenance.license_class == spec.license_class
        assert d.provenance.first_party == spec.first_party
        assert d.artifact.checksum is None  # not hashed yet in Phase 0


def test_no_package_points_at_an_unresolved_model_descriptor() -> None:
    packages = [p for p in load_registry().values() if p.model_id is not None]
    assert packages, "registry has no model-backed packages"
    for pkg in packages:
        assert pkg.model_id is not None
        spec = find_model_by_alias(pkg.model_id)
        assert spec is not None, (
            f"package {pkg.package_id!r} references unknown model {pkg.model_id!r}"
        )
        d = descriptor_from_spec(spec)
        assert d.decoder in DECODER_REGISTRY, (
            f"package {pkg.package_id!r} resolves to unknown decoder {d.decoder!r}"
        )


# --- specific descriptor values for representative families -------------


def test_yolox_descriptor_values() -> None:
    d = descriptor_from_spec(_spec("yolox_s_v1"))
    assert d.family == "yolo_anchorfree"
    assert d.decoder == "yolox_onnx_fp16_v1"
    assert d.precision == "fp16"
    assert d.input.color == "BGR"
    assert d.input.color_format == 1
    assert d.input.net_scale_factor == 1.0
    assert d.input.letterbox is True
    assert d.output.parse_bbox_func == "NvDsInferParseYolox"
    assert d.output.topk == 300
    assert d.output.nms_iou == 0.45
    assert d.output.default_conf == 0.25
    assert d.output.class0_threshold is None
    assert d.artifact.onnx_filename == "yolox_s.onnx"  # family default (onnx_file unset)
    assert d.artifact.artifact_layout == "host_model_store"
    assert d.artifact.custom_parser_lib == "libnvdsparsebbox_yolox.so"
    assert d.artifact.labels_source == "registry_generated"


def test_trafficcamnet_descriptor_values() -> None:
    d = descriptor_from_spec(_spec("trafficcamnet_v1"))
    assert d.family == "detectnet_v2"
    assert d.precision == "int8"
    assert d.input.color_format == 0
    assert d.output.parse_bbox_func is None
    assert d.output.topk == 20
    assert d.output.default_conf == 0.2
    assert d.output.class0_threshold == 0.4  # the second [class-attrs-0] block
    assert d.artifact.artifact_layout == "image_builtin"
    assert d.artifact.labels_source == "image"
    assert d.artifact.custom_parser_lib is None


def test_peoplenet_descriptor_values() -> None:
    d = descriptor_from_spec(_spec("peoplenet_v1"))
    assert d.family == "detectnet_v2"
    assert d.precision == "fp16"
    assert d.output.parse_bbox_func is None
    assert d.artifact.onnx_filename == "resnet34_peoplenet.onnx"
    assert d.artifact.artifact_layout == "host_model_store"
    assert d.artifact.custom_parser_lib is None


# --- validation catches missing / invalid required fields ---------------


def _valid_input() -> DescriptorInput:
    return DescriptorInput(
        width=640,
        height=640,
        channels=3,
        layout="NCHW",
        color="RGB",
        color_format=0,
        net_scale_factor=1.0,
        letterbox=True,
        pad_value=0,
    )


def _valid_output() -> DescriptorOutput:
    return DescriptorOutput(
        nms_mode="cluster",
        decoder="yolox_onnx_fp16_v1",
        parse_bbox_func="NvDsInferParseYolox",
        topk=300,
        nms_iou=0.45,
        default_conf=0.25,
    )


def _valid_artifact() -> DescriptorArtifact:
    return DescriptorArtifact(
        onnx_filename="m.onnx",
        artifact_layout="host_model_store",
        model_artifact_path="/m/m",
        artifact_subdir="m",
        labels_source="registry_generated",
    )


def _valid_provenance() -> DescriptorProvenance:
    return DescriptorProvenance(license_class="permissive", first_party=True, source="x")


def _descriptor(**over: object) -> ModelDescriptor:
    base: dict[str, object] = {
        "model_id": "m",
        "task": "detect",
        "family": "yolo_anchorfree",
        "decoder": "yolox_onnx_fp16_v1",
        "input": _valid_input(),
        "output": _valid_output(),
        "class_map": {"person": 0},
        "precision": "fp16",
        "artifact": _valid_artifact(),
        "provenance": _valid_provenance(),
    }
    base.update(over)
    return ModelDescriptor(**base)  # type: ignore[arg-type]


def test_valid_descriptor_constructs() -> None:
    d = _descriptor()
    assert d.model_id == "m"


def test_input_rejects_nonpositive_dims() -> None:
    with pytest.raises(DescriptorValidationError):
        DescriptorInput(
            width=0,
            height=640,
            channels=3,
            layout="NCHW",
            color="RGB",
            color_format=0,
            net_scale_factor=1.0,
            letterbox=True,
            pad_value=0,
        )


def test_input_rejects_unknown_color() -> None:
    with pytest.raises(DescriptorValidationError):
        DescriptorInput(
            width=640,
            height=640,
            channels=3,
            layout="NCHW",
            color="YUV",
            color_format=0,
            net_scale_factor=1.0,
            letterbox=True,
            pad_value=0,
        )


def test_output_rejects_empty_decoder() -> None:
    with pytest.raises(DescriptorValidationError):
        DescriptorOutput(
            nms_mode="cluster",
            decoder="",
            parse_bbox_func=None,
            topk=300,
            nms_iou=0.45,
            default_conf=0.25,
        )


def test_artifact_rejects_empty_onnx() -> None:
    with pytest.raises(DescriptorValidationError):
        DescriptorArtifact(
            onnx_filename="",
            artifact_layout="host_model_store",
            model_artifact_path="/m/m",
            artifact_subdir="m",
            labels_source="registry_generated",
        )


def test_descriptor_rejects_non_detect_task() -> None:
    with pytest.raises(DescriptorValidationError):
        _descriptor(task="segment")


def test_descriptor_rejects_unknown_family() -> None:
    with pytest.raises(DescriptorValidationError):
        _descriptor(family="made_up")


def test_descriptor_rejects_empty_class_map() -> None:
    with pytest.raises(DescriptorValidationError):
        _descriptor(class_map={})


def test_descriptor_rejects_output_decoder_mismatch() -> None:
    # output.decoder must equal the descriptor decoder
    with pytest.raises(DescriptorValidationError):
        _descriptor(decoder="ultralytics_onnx_fp16_v1")  # output still says yolox


# --- consistency: declared values match the rendered config -------------


def test_descriptor_declarations_match_rendered_config() -> None:
    for spec in _runnable_deepstream_specs():
        d = descriptor_from_spec(spec)
        cfg = build_model_runtime_plan(
            model=spec.model_id, work_dir="/w", model_store_dir="/m"
        ).pgie_config
        assert f"model-color-format={d.input.color_format}\n" in cfg
        assert f"num-detected-classes={len(d.class_map)}\n" in cfg
        assert f"topk={d.output.topk}\n" in cfg
        assert f"nms-iou-threshold={d.output.nms_iou}\n" in cfg
        assert f"pre-cluster-threshold={d.output.default_conf}\n" in cfg
        if d.output.parse_bbox_func is not None:
            assert f"parse-bbox-func-name={d.output.parse_bbox_func}\n" in cfg
        else:
            assert "parse-bbox-func-name" not in cfg
        if d.output.class0_threshold is not None:
            assert f"pre-cluster-threshold={d.output.class0_threshold}\n" in cfg
