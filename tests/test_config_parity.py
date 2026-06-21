"""BYOM Phase 0 — golden parity tests for the descriptor + decoder refactor.

``build_model_runtime_plan`` was refactored from four hard-coded
``config_template`` branches to a data-driven path: each model is normalized
into a :class:`ModelDescriptor` whose ``decoder`` selects a ``DecoderSpec``
from a finite registry. This file PROVES the refactor changed no behaviour:
for every current model the generated PGIE config and the whole
``ModelRuntimePlan`` are byte-identical to the pre-refactor output.

The expected PGIE bytes are PINNED LITERALS here — per-family golden templates
with per-model substitutions — written down independently of the production
builders. Any drift in production (key order, values, paths, assembly) fails
against these fixed literals. No GPU / Docker: ``build_model_runtime_plan`` is
a pure function.
"""

from __future__ import annotations

import functools
from collections.abc import Callable

import pytest
from app.runtimes.deepstream_runtime.inference import (
    ModelRuntimePlan,
    build_model_runtime_plan,
)

WORK = "/w"
STORE = "/m"
_TCN_DIR = "/opt/nvidia/deepstream/deepstream/samples/models/Primary_Detector"

# Parser identifiers carried straight from the registry ``deepstream.parser``.
_PARSER_BUILTIN = "nvinfer_builtin_detectnet_v2"
_PARSER_YOLOX = (
    "NvDsInferParseYolox (custom grid-decode parser, app/runtimes/deepstream_runtime/yolox_parser)"
)
_PARSER_ULTRA = (
    "NvDsInferParseUltralytics (custom decode parser, "
    "app/runtimes/deepstream_runtime/ultralytics_parser; NOT yet GPU-validated)"
)

# The COCO-80 labels body (registry class order), generated into the run dir
# for the YOLOX / Ultralytics families.
COCO80 = (
    "person\nbicycle\ncar\nmotorcycle\nairplane\nbus\ntrain\ntruck\nboat\n"
    "traffic light\nfire hydrant\nstop sign\nparking meter\nbench\nbird\ncat\n"
    "dog\nhorse\nsheep\ncow\nelephant\nbear\nzebra\ngiraffe\nbackpack\numbrella\n"
    "handbag\ntie\nsuitcase\nfrisbee\nskis\nsnowboard\nsports ball\nkite\n"
    "baseball bat\nbaseball glove\nskateboard\nsurfboard\ntennis racket\nbottle\n"
    "wine glass\ncup\nfork\nknife\nspoon\nbowl\nbanana\napple\nsandwich\norange\n"
    "broccoli\ncarrot\nhot dog\npizza\ndonut\ncake\nchair\ncouch\npotted plant\n"
    "bed\ndining table\ntoilet\ntv\nlaptop\nmouse\nremote\nkeyboard\ncell phone\n"
    "microwave\noven\ntoaster\nsink\nrefrigerator\nbook\nclock\nvase\nscissors\n"
    "teddy bear\nhair drier\ntoothbrush\n"
)


# --- golden PGIE templates (pinned literals, independent of production) ---


def _tcn_golden(conf: float | None) -> str:
    all_thr = "0.2" if conf is None else f"{conf}"
    cls0 = "0.4" if conf is None else f"{conf}"
    return (
        "[property]\n"
        "gpu-id=0\n"
        "net-scale-factor=0.00392156862745098\n"
        f"onnx-file={_TCN_DIR}/resnet18_trafficcamnet_pruned.onnx\n"
        f"model-engine-file={WORK}/trafficcamnet_b1_gpu0_int8.engine\n"
        f"labelfile-path={_TCN_DIR}/labels.txt\n"
        f"int8-calib-file={_TCN_DIR}/cal_trt.bin\n"
        "batch-size=1\n"
        "process-mode=1\n"
        "model-color-format=0\n"
        "network-mode=1\n"
        "num-detected-classes=4\n"
        "interval=0\n"
        "gie-unique-id=1\n"
        "cluster-mode=2\n"
        "[class-attrs-all]\n"
        "topk=20\n"
        "nms-iou-threshold=0.5\n"
        f"pre-cluster-threshold={all_thr}\n"
        "[class-attrs-0]\n"
        "topk=20\n"
        "nms-iou-threshold=0.5\n"
        f"pre-cluster-threshold={cls0}\n"
    )


def _yolox_golden(model_dir: str, onnx: str, engine: str, conf: float | None) -> str:
    thr = "0.25" if conf is None else f"{conf}"
    return (
        "[property]\n"
        "gpu-id=0\n"
        "net-scale-factor=1.0\n"
        "model-color-format=1\n"
        f"onnx-file={model_dir}/{onnx}\n"
        f"model-engine-file={model_dir}/{engine}\n"
        f"labelfile-path={WORK}/labels.txt\n"
        "batch-size=1\n"
        "process-mode=1\n"
        "network-mode=2\n"
        "num-detected-classes=80\n"
        "interval=0\n"
        "gie-unique-id=1\n"
        "cluster-mode=2\n"
        "maintain-aspect-ratio=1\n"
        "symmetric-padding=0\n"
        "parse-bbox-func-name=NvDsInferParseYolox\n"
        f"custom-lib-path={model_dir}/libnvdsparsebbox_yolox.so\n"
        "[class-attrs-all]\n"
        "topk=300\n"
        "nms-iou-threshold=0.45\n"
        f"pre-cluster-threshold={thr}\n"
    )


def _ultra_golden(model_dir: str, onnx: str, engine: str, conf: float | None) -> str:
    thr = "0.25" if conf is None else f"{conf}"
    return (
        "[property]\n"
        "gpu-id=0\n"
        "net-scale-factor=0.00392156862745098\n"
        "model-color-format=0\n"
        f"onnx-file={model_dir}/{onnx}\n"
        f"model-engine-file={model_dir}/{engine}\n"
        f"labelfile-path={WORK}/labels.txt\n"
        "batch-size=1\n"
        "process-mode=1\n"
        "network-mode=2\n"
        "num-detected-classes=80\n"
        "interval=0\n"
        "gie-unique-id=1\n"
        "cluster-mode=2\n"
        "maintain-aspect-ratio=1\n"
        "symmetric-padding=0\n"
        "parse-bbox-func-name=NvDsInferParseUltralytics\n"
        f"custom-lib-path={model_dir}/libnvdsparsebbox_ultralytics.so\n"
        "[class-attrs-all]\n"
        "topk=300\n"
        "nms-iou-threshold=0.45\n"
        f"pre-cluster-threshold={thr}\n"
    )


def _detectnet_golden(
    model_dir: str, onnx: str, engine: str, num_classes: int, conf: float | None
) -> str:
    thr = "0.2" if conf is None else f"{conf}"
    return (
        "[property]\n"
        "gpu-id=0\n"
        "net-scale-factor=0.00392156862745098\n"
        "model-color-format=0\n"
        f"onnx-file={model_dir}/{onnx}\n"
        f"model-engine-file={model_dir}/{engine}\n"
        f"labelfile-path={WORK}/labels.txt\n"
        "batch-size=1\n"
        "process-mode=1\n"
        "network-mode=2\n"
        f"num-detected-classes={num_classes}\n"
        "interval=0\n"
        "gie-unique-id=1\n"
        "cluster-mode=2\n"
        "[class-attrs-all]\n"
        "topk=20\n"
        "nms-iou-threshold=0.5\n"
        f"pre-cluster-threshold={thr}\n"
    )


# --- the 10-model expectation table -------------------------------------


class Golden:
    """One model's full expected runtime plan. ``pgie`` is a callable so the
    confidence-threshold-dependent thresholds are exercised at None + 0.55."""

    def __init__(
        self,
        *,
        model_id: str,
        family: str,
        backend: str,
        model_dir: str,
        parser: str,
        labels_body: str | None,
        mounts: tuple[str, ...],
        required: tuple[str, ...],
        image_builtin: bool,
        pgie: Callable[[float | None], str],
    ) -> None:
        self.model_id = model_id
        self.family = family
        self.backend = backend
        self.model_dir = model_dir
        self.parser = parser
        self.labels_body = labels_body
        self.mounts = mounts
        self.required = required
        self.image_builtin = image_builtin
        self.pgie: Callable[[float | None], str] = pgie


def _yolox(model_id: str, onnx: str) -> Golden:
    md = f"{STORE}/{model_id}"
    engine = onnx[:-5] + "_b1_gpu0_fp16.engine"
    return Golden(
        model_id=model_id,
        family="yolox",
        backend="tensorrt_fp16",
        model_dir=md,
        parser=_PARSER_YOLOX,
        labels_body=COCO80,
        mounts=(md,),
        required=(f"{md}/{onnx}", f"{md}/libnvdsparsebbox_yolox.so"),
        image_builtin=False,
        pgie=functools.partial(_yolox_golden, md, onnx, engine),
    )


def _ultra(model_id: str, family: str, onnx: str) -> Golden:
    md = f"{STORE}/{model_id}"
    engine = onnx[:-5] + "_b1_gpu0_fp16.engine"
    return Golden(
        model_id=model_id,
        family=family,
        backend="tensorrt_fp16",
        model_dir=md,
        parser=_PARSER_ULTRA,
        labels_body=COCO80,
        mounts=(md,),
        required=(f"{md}/{onnx}", f"{md}/libnvdsparsebbox_ultralytics.so"),
        image_builtin=False,
        pgie=functools.partial(_ultra_golden, md, onnx, engine),
    )


GOLDENS: list[Golden] = [
    Golden(
        model_id="trafficcamnet_v1",
        family="tao_detectnet_v2",
        backend="tensorrt_int8",
        model_dir=_TCN_DIR,
        parser=_PARSER_BUILTIN,
        labels_body=None,
        mounts=(),
        required=(),
        image_builtin=True,
        pgie=_tcn_golden,
    ),
    _yolox("yolox_s_v1", "yolox_s.onnx"),
    _yolox("yolox_nano_v1", "yolox_nano.onnx"),
    _yolox("yolox_tiny_v1", "yolox_tiny.onnx"),
    _yolox("yolox_m_v1", "yolox_m.onnx"),
    _ultra("yolov8n_v1", "ultralytics_yolov8", "yolov8n.onnx"),
    _ultra("yolo11n_v1", "ultralytics_yolo11", "yolo11n.onnx"),
    _ultra("yolo11s_v1", "ultralytics_yolo11", "yolo11s.onnx"),
    _ultra("yolo11m_v1", "ultralytics_yolo11", "yolo11m.onnx"),
    Golden(
        model_id="peoplenet_v1",
        family="tao_detectnet_v2",
        backend="tensorrt_fp16",
        model_dir=f"{STORE}/peoplenet_v1",
        parser=_PARSER_BUILTIN,
        labels_body="person\nbag\nface\n",
        mounts=(f"{STORE}/peoplenet_v1",),
        required=(f"{STORE}/peoplenet_v1/resnet34_peoplenet.onnx",),
        image_builtin=False,
        pgie=functools.partial(
            _detectnet_golden,
            f"{STORE}/peoplenet_v1",
            "resnet34_peoplenet.onnx",
            "resnet34_peoplenet_b1_gpu0_fp16.engine",
            3,
        ),
    ),
]

_IDS = [g.model_id for g in GOLDENS]


def _plan(model_id: str, conf: float | None) -> ModelRuntimePlan:
    return build_model_runtime_plan(
        model=model_id, work_dir=WORK, model_store_dir=STORE, confidence_threshold=conf
    )


@pytest.mark.parametrize("g", GOLDENS, ids=_IDS)
@pytest.mark.parametrize("conf", [None, 0.55])
def test_pgie_config_is_byte_identical(g: Golden, conf: float | None) -> None:
    """The generated PGIE config is byte-for-byte the pinned golden, for both
    the default (None) and an operator override (0.55) confidence."""
    assert _plan(g.model_id, conf).pgie_config == g.pgie(conf)


@pytest.mark.parametrize("g", GOLDENS, ids=_IDS)
def test_runtime_plan_fields_unchanged(g: Golden) -> None:
    """Every non-config ModelRuntimePlan field matches the pre-refactor plan."""
    plan = _plan(g.model_id, None)
    assert plan.model_id == g.model_id
    assert plan.model_family == g.family
    assert plan.inference_backend == g.backend
    assert plan.model_dir == g.model_dir
    assert plan.parser == g.parser
    assert plan.labels_body == g.labels_body
    assert plan.extra_mount_dirs == g.mounts
    assert plan.required_files == g.required
    assert plan.image_builtin is g.image_builtin


def test_yolox_s_explicit_golden_live_verified_path() -> None:
    """Dedicated, fully-literal golden for the LIVE-VERIFIED yolox_s path
    (P5-B, crowdcountS2 RTX 5070 Ti) — the one config proven end-to-end on a
    GPU. Pinned in full so any drift on this path is impossible to miss."""
    expected = (
        "[property]\n"
        "gpu-id=0\n"
        "net-scale-factor=1.0\n"
        "model-color-format=1\n"
        "onnx-file=/m/yolox_s_v1/yolox_s.onnx\n"
        "model-engine-file=/m/yolox_s_v1/yolox_s_b1_gpu0_fp16.engine\n"
        "labelfile-path=/w/labels.txt\n"
        "batch-size=1\n"
        "process-mode=1\n"
        "network-mode=2\n"
        "num-detected-classes=80\n"
        "interval=0\n"
        "gie-unique-id=1\n"
        "cluster-mode=2\n"
        "maintain-aspect-ratio=1\n"
        "symmetric-padding=0\n"
        "parse-bbox-func-name=NvDsInferParseYolox\n"
        "custom-lib-path=/m/yolox_s_v1/libnvdsparsebbox_yolox.so\n"
        "[class-attrs-all]\n"
        "topk=300\n"
        "nms-iou-threshold=0.45\n"
        "pre-cluster-threshold=0.25\n"
    )
    assert _plan("yolox_s_v1", None).pgie_config == expected
    # operator override flows into pre-cluster-threshold, nothing else.
    overridden = _plan("yolox_s_v1", 0.6).pgie_config
    assert overridden == expected.replace("pre-cluster-threshold=0.25", "pre-cluster-threshold=0.6")
