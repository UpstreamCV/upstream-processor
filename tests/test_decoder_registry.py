"""BYOM Phase 0 — Decoder Registry tests.

The finite registry that ``build_model_runtime_plan`` dispatches through. Covers
the exact key set, the clear error for an unknown key (and that it stays a
``ValueError`` so the worker's plan-failure path keeps handling it), that each
decoder renders a plausible config, the per-decoder metadata, and the engine
filename helper. No GPU / Docker.
"""

from __future__ import annotations

import pytest
from app.runtimes.deepstream_runtime.decoders import (
    DECODER_REGISTRY,
    DecoderRegistryError,
    PgieRenderContext,
    engine_filename_for,
    get_decoder,
)

EXPECTED_KEYS = {
    "builtin_trafficcamnet_int8_v1",
    "yolox_onnx_fp16_v1",
    "ultralytics_onnx_fp16_v1",
    "builtin_detectnet_peoplenet_v1",
}


def test_registry_has_exactly_the_expected_decoders() -> None:
    assert set(DECODER_REGISTRY) == EXPECTED_KEYS


def test_get_decoder_unknown_key_raises_clear_error() -> None:
    with pytest.raises(DecoderRegistryError) as exc:
        get_decoder("does_not_exist")
    msg = str(exc.value)
    assert "does_not_exist" in msg
    for key in EXPECTED_KEYS:
        assert key in msg  # the error lists the known decoders


def test_decoder_registry_error_is_value_error() -> None:
    # back-compat: the worker's plan-failure path catches ValueError, exactly
    # as it did for the pre-Phase-0 unsupported-config_template raise.
    assert issubclass(DecoderRegistryError, ValueError)
    with pytest.raises(ValueError):
        get_decoder("nope")


@pytest.mark.parametrize("key", sorted(EXPECTED_KEYS))
def test_get_decoder_returns_matching_spec_and_renders(key: str) -> None:
    dec = get_decoder(key)
    assert dec.decoder_key == key
    ctx = PgieRenderContext(
        model_dir="/m/x",
        engine_path="/m/x/e.engine",
        labels_path="/w/labels.txt",
        num_classes=80,
        onnx_file="x.onnx",
        confidence_threshold=None,
    )
    cfg = dec.render(ctx)
    assert cfg.startswith("[property]\n")
    assert "cluster-mode=2\n" in cfg
    assert "[class-attrs-all]\n" in cfg


def test_trafficcamnet_decoder_metadata() -> None:
    dec = get_decoder("builtin_trafficcamnet_int8_v1")
    assert dec.inference_backend == "tensorrt_int8"
    assert dec.artifact_layout == "image_builtin"
    assert dec.engine_location == "work_dir"
    assert dec.engine_filename == "trafficcamnet_b1_gpu0_int8.engine"
    assert dec.parse_bbox_func is None
    assert dec.custom_parser_lib is None
    assert dec.labels_source == "image"


def test_yolox_decoder_metadata() -> None:
    dec = get_decoder("yolox_onnx_fp16_v1")
    assert dec.inference_backend == "tensorrt_fp16"
    assert dec.artifact_layout == "host_model_store"
    assert dec.engine_location == "model_dir"
    assert dec.engine_filename is None
    assert dec.parse_bbox_func == "NvDsInferParseYolox"
    assert dec.custom_parser_lib == "libnvdsparsebbox_yolox.so"
    assert dec.labels_source == "registry_generated"


def test_ultralytics_decoder_metadata() -> None:
    dec = get_decoder("ultralytics_onnx_fp16_v1")
    assert dec.inference_backend == "tensorrt_fp16"
    assert dec.artifact_layout == "host_model_store"
    assert dec.parse_bbox_func == "NvDsInferParseUltralytics"
    assert dec.custom_parser_lib == "libnvdsparsebbox_ultralytics.so"
    assert dec.labels_source == "registry_generated"


def test_peoplenet_decoder_metadata() -> None:
    dec = get_decoder("builtin_detectnet_peoplenet_v1")
    assert dec.inference_backend == "tensorrt_fp16"
    assert dec.artifact_layout == "host_model_store"
    assert dec.engine_location == "model_dir"
    assert dec.parse_bbox_func is None  # built-in DetectNet parser
    assert dec.custom_parser_lib is None
    assert dec.labels_source == "registry_generated"


def test_engine_filename_for() -> None:
    tcn = get_decoder("builtin_trafficcamnet_int8_v1")
    # work_dir decoder: fixed engine name regardless of onnx
    assert engine_filename_for(tcn, "anything.onnx") == "trafficcamnet_b1_gpu0_int8.engine"
    yolox = get_decoder("yolox_onnx_fp16_v1")
    # model_dir decoder: derived from the onnx stem
    assert engine_filename_for(yolox, "yolox_s.onnx") == "yolox_s_b1_gpu0_fp16.engine"
    assert engine_filename_for(yolox, "yolo11m.onnx") == "yolo11m_b1_gpu0_fp16.engine"
