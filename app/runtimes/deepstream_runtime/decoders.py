"""BYOM Phase 0 — DeepStream decoder registry.

A *finite, data-driven* registry that maps a model's ``decoder`` key (which in
Phase 0 is exactly its registry ``deepstream.config_template`` value) to the
behaviour needed to turn a model into a running nvinfer plan:

- the exact PGIE byte-template (the ``build_*_pgie_config`` functions, moved
  here verbatim from ``inference.py`` so they remain byte-for-byte identical);
- the assembly facts ``build_model_runtime_plan`` needs around that template
  (inference backend, where the TensorRT engine lives, whether artifacts are
  baked into the image or host-mounted, the custom parser ``.so``, and where
  labels come from).

Why keep four hand-written byte-templates instead of one generic renderer?
Because the four current families do **not** differ only in values — they
differ in *key order and key set* (TrafficCamNet emits ``model-color-format``
at a different position, carries an ``int8-calib-file`` line and a second
``[class-attrs-0]`` block; the YOLO families add ``maintain-aspect-ratio`` /
``parse-bbox-func-name``; PeopleNet has no custom parser). A single
fixed-order renderer therefore could not reproduce every family byte-for-byte.
So in Phase 0 the **descriptor supplies the per-model values** and the
**decoder owns each family's exact template**. Collapsing the templates into a
fully descriptor-emitted renderer is deferred to a later phase (it needs a
golden corpus to land against; see ``docs/byom-descriptors.md``).

Adding a new *model* in an existing family is pure descriptor data (no code
here). Adding a new *family* is one new ``DecoderSpec`` entry (plus, if its
output layout is genuinely new, one render function) — never a new ``if`` in
``build_model_runtime_plan``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

# --- decoder keys -------------------------------------------------------
# These strings ARE the registry ``deepstream.config_template`` values. They
# are the single source of truth for decoder identity; ``model_descriptor``
# mirrors them in its per-template profile table and a test asserts the two
# never diverge.
TRAFFICCAMNET_DECODER = "builtin_trafficcamnet_int8_v1"
YOLOX_DECODER = "yolox_onnx_fp16_v1"
ULTRALYTICS_DECODER = "ultralytics_onnx_fp16_v1"  # YOLO11 / YOLOv8 (custom parser)
PEOPLENET_DECODER = "builtin_detectnet_peoplenet_v1"  # NGC PeopleNet (builtin detectnet parser)

# Filename conventions inside a host model-store dir (created by the
# operator-run prepare scripts; never committed).
_YOLOX_ONNX_FILENAME = "yolox_s.onnx"
_YOLOX_PARSER_LIB_FILENAME = "libnvdsparsebbox_yolox.so"
_YOLOX_ENGINE_FILENAME = "yolox_s_b1_gpu0_fp16.engine"
# The Ultralytics (YOLO11 / YOLOv8) family shares one custom decode parser,
# compiled by scripts/prepare_ultralytics_model.sh into the model dir.
_ULTRALYTICS_PARSER_LIB_FILENAME = "libnvdsparsebbox_ultralytics.so"


class DecoderRegistryError(ValueError):
    """Asked for a decoder key the runtime does not know how to render.

    Subclasses ``ValueError`` so the worker's existing plan-failure path
    (post ``failed`` rather than crash) keeps handling it, exactly as the
    pre-Phase-0 ``raise ValueError`` for an unsupported ``config_template`` did.
    """


def _fp16_engine_filename(onnx_file: str) -> str:
    """``<stem>_b1_gpu0_fp16.engine`` for an ONNX filename. For the original
    ``yolox_s.onnx`` this reproduces ``_YOLOX_ENGINE_FILENAME`` exactly."""
    stem = onnx_file[:-5] if onnx_file.endswith(".onnx") else onnx_file
    return f"{stem}_b1_gpu0_fp16.engine"


# --- PGIE byte-templates (moved verbatim from inference.py) --------------


def build_pgie_config(
    *, model_dir: str, engine_path: str, confidence_threshold: float | None = None
) -> str:
    """PGIE config: bundled TrafficCamNet ONNX, INT8, batch 1. The
    ``model-engine-file`` points at a not-yet-existing path so nvinfer must
    BUILD the TensorRT engine (the SM-arch test). Mirrors M8.2C1-A.

    M7C: when ``confidence_threshold`` is supplied (the resolved per-camera
    operator value), it overrides BOTH ``pre-cluster-threshold`` entries
    (``class-attrs-all`` and ``class-attrs-0``) so detection filtering is
    uniform across classes. When ``None`` (no operator override) the output
    is byte-identical to the previous hard-coded 0.2 / 0.4 defaults."""
    all_thr = "0.2" if confidence_threshold is None else f"{confidence_threshold}"
    cls0_thr = "0.4" if confidence_threshold is None else f"{confidence_threshold}"
    return (
        "[property]\n"
        "gpu-id=0\n"
        "net-scale-factor=0.00392156862745098\n"
        f"onnx-file={model_dir}/resnet18_trafficcamnet_pruned.onnx\n"
        f"model-engine-file={engine_path}\n"
        f"labelfile-path={model_dir}/labels.txt\n"
        f"int8-calib-file={model_dir}/cal_trt.bin\n"
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
        f"pre-cluster-threshold={cls0_thr}\n"
    )


def build_yolox_pgie_config(
    *,
    model_dir: str,
    engine_path: str,
    labels_path: str,
    num_classes: int,
    confidence_threshold: float | None = None,
    onnx_file: str = _YOLOX_ONNX_FILENAME,
) -> str:
    """PGIE config for the official (raw, undecoded) YOLOX ONNX export.

    Preprocessing mirrors YOLOX's own demo: NO normalization
    (``net-scale-factor=1.0``, no offsets) and BGR channel order
    (``model-color-format=1``) — YOLOX removed input normalization in
    v0.1.1+. ``maintain-aspect-ratio=1`` letterboxes like the YOLOX demo
    (which pads bottom/right; padding VALUE differs — 0 vs 114 — an accepted
    probe-level approximation). FP16 (``network-mode=2``); INT8 needs a
    calibration cache we don't have. NMS is nvinfer's per-class
    ``cluster-mode=2`` clustering; the custom parser
    (``NvDsInferParseYolox``) only decodes grids + thresholds.

    M7C parity: ``confidence_threshold`` overrides ``pre-cluster-threshold``
    when an operator value is applied; default 0.25 (YOLOX demo default).
    """
    thr = "0.25" if confidence_threshold is None else f"{confidence_threshold}"
    return (
        "[property]\n"
        "gpu-id=0\n"
        "net-scale-factor=1.0\n"
        "model-color-format=1\n"
        f"onnx-file={model_dir}/{onnx_file}\n"
        f"model-engine-file={engine_path}\n"
        f"labelfile-path={labels_path}\n"
        "batch-size=1\n"
        "process-mode=1\n"
        "network-mode=2\n"
        f"num-detected-classes={num_classes}\n"
        "interval=0\n"
        "gie-unique-id=1\n"
        "cluster-mode=2\n"
        "maintain-aspect-ratio=1\n"
        "symmetric-padding=0\n"
        "parse-bbox-func-name=NvDsInferParseYolox\n"
        f"custom-lib-path={model_dir}/{_YOLOX_PARSER_LIB_FILENAME}\n"
        "[class-attrs-all]\n"
        "topk=300\n"
        "nms-iou-threshold=0.45\n"
        f"pre-cluster-threshold={thr}\n"
    )


def build_ultralytics_pgie_config(
    *,
    model_dir: str,
    engine_path: str,
    labels_path: str,
    num_classes: int,
    onnx_file: str,
    confidence_threshold: float | None = None,
) -> str:
    """PGIE config for an Ultralytics (YOLO11 / YOLOv8) ONNX export.

    P5-K, IN-REPO PREP — NOT yet GPU-validated. Preprocessing mirrors
    Ultralytics: RGB (``model-color-format=0``), 1/255 normalization
    (``net-scale-factor``), letterbox (``maintain-aspect-ratio=1``). FP16
    (``network-mode=2``). The custom parser ``NvDsInferParseUltralytics``
    (app/runtimes/deepstream_runtime/ultralytics_parser) decodes the
    ``[1, 4+num_classes, num_anchors]`` transposed, objectness-free output;
    per-class NMS is nvinfer's ``cluster-mode=2`` clustering.

    The parser is authored from the published Ultralytics output contract but
    has NOT been compiled/run on a GPU in this milestone — the package that
    references this model is ``experimental`` and warns accordingly.
    """
    thr = "0.25" if confidence_threshold is None else f"{confidence_threshold}"
    return (
        "[property]\n"
        "gpu-id=0\n"
        "net-scale-factor=0.00392156862745098\n"
        "model-color-format=0\n"
        f"onnx-file={model_dir}/{onnx_file}\n"
        f"model-engine-file={engine_path}\n"
        f"labelfile-path={labels_path}\n"
        "batch-size=1\n"
        "process-mode=1\n"
        "network-mode=2\n"
        f"num-detected-classes={num_classes}\n"
        "interval=0\n"
        "gie-unique-id=1\n"
        "cluster-mode=2\n"
        "maintain-aspect-ratio=1\n"
        "symmetric-padding=0\n"
        "parse-bbox-func-name=NvDsInferParseUltralytics\n"
        f"custom-lib-path={model_dir}/{_ULTRALYTICS_PARSER_LIB_FILENAME}\n"
        "[class-attrs-all]\n"
        "topk=300\n"
        "nms-iou-threshold=0.45\n"
        f"pre-cluster-threshold={thr}\n"
    )


def build_detectnet_pgie_config(
    *,
    model_dir: str,
    engine_path: str,
    labels_path: str,
    num_classes: int,
    onnx_file: str,
    confidence_threshold: float | None = None,
) -> str:
    """PGIE config for a host-mounted DetectNet_v2 ONNX (NGC PeopleNet).

    P5-K, IN-REPO PREP — NOT yet GPU-validated. Uses the BUILT-IN
    DetectNet_v2 bbox parser (``requires_custom_parser=false``, same family as
    the bundled TrafficCamNet) — no custom-lib-path. RGB, 1/255 scale, FP16
    (``network-mode=2``; PeopleNet's INT8 calibration cache is optional and
    not assumed here). Labels are registry-generated (person/bag/face)."""
    thr = "0.2" if confidence_threshold is None else f"{confidence_threshold}"
    return (
        "[property]\n"
        "gpu-id=0\n"
        "net-scale-factor=0.00392156862745098\n"
        "model-color-format=0\n"
        f"onnx-file={model_dir}/{onnx_file}\n"
        f"model-engine-file={engine_path}\n"
        f"labelfile-path={labels_path}\n"
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


# --- render context + decoder spec --------------------------------------


@dataclass(frozen=True)
class PgieRenderContext:
    """The per-run, descriptor-derived inputs a decoder's render function
    needs. Built by ``build_model_runtime_plan`` from the Model Descriptor;
    every field is populated even when a given decoder ignores it (the
    built-in DetectNet template ignores ``onnx_file`` / ``labels_path`` /
    ``num_classes``), so the render signature is uniform across families."""

    model_dir: str
    engine_path: str
    labels_path: str
    num_classes: int
    onnx_file: str
    confidence_threshold: float | None = None


def _render_trafficcamnet(ctx: PgieRenderContext) -> str:
    return build_pgie_config(
        model_dir=ctx.model_dir,
        engine_path=ctx.engine_path,
        confidence_threshold=ctx.confidence_threshold,
    )


def _render_yolox(ctx: PgieRenderContext) -> str:
    return build_yolox_pgie_config(
        model_dir=ctx.model_dir,
        engine_path=ctx.engine_path,
        labels_path=ctx.labels_path,
        num_classes=ctx.num_classes,
        confidence_threshold=ctx.confidence_threshold,
        onnx_file=ctx.onnx_file,
    )


def _render_ultralytics(ctx: PgieRenderContext) -> str:
    return build_ultralytics_pgie_config(
        model_dir=ctx.model_dir,
        engine_path=ctx.engine_path,
        labels_path=ctx.labels_path,
        num_classes=ctx.num_classes,
        onnx_file=ctx.onnx_file,
        confidence_threshold=ctx.confidence_threshold,
    )


def _render_detectnet(ctx: PgieRenderContext) -> str:
    return build_detectnet_pgie_config(
        model_dir=ctx.model_dir,
        engine_path=ctx.engine_path,
        labels_path=ctx.labels_path,
        num_classes=ctx.num_classes,
        onnx_file=ctx.onnx_file,
        confidence_threshold=ctx.confidence_threshold,
    )


@dataclass(frozen=True)
class DecoderSpec:
    """One model family's render template + plan-assembly facts. Parameterised,
    never per-model — a new model in this family reuses the same spec."""

    decoder_key: str
    render: Callable[[PgieRenderContext], str]
    # "tensorrt_int8" | "tensorrt_fp16"
    inference_backend: str
    # "image_builtin" (artifacts baked into the DeepStream image, no mount, no
    # host requirements) | "host_model_store" (mounted from the host store).
    artifact_layout: str
    # "work_dir" (engine built per-run into the temp dir) | "model_dir"
    # (engine cached in the model store across runs).
    engine_location: str
    # Fixed engine filename (used when ``engine_location == 'work_dir'``); when
    # ``None`` the engine name is derived from the ONNX via
    # ``_fp16_engine_filename`` (the cached host-store case).
    engine_filename: str | None
    # Informational: the custom bbox parser entry point, or ``None`` for the
    # built-in DetectNet_v2 parser.
    parse_bbox_func: str | None
    # Custom parser ``.so`` that must exist on the host (adds a required file),
    # or ``None`` for the built-in parser.
    custom_parser_lib: str | None
    # "image" (labels baked into the image) | "registry_generated" (labels file
    # written from the model's ``supported_classes``).
    labels_source: str


DECODER_REGISTRY: dict[str, DecoderSpec] = {
    TRAFFICCAMNET_DECODER: DecoderSpec(
        decoder_key=TRAFFICCAMNET_DECODER,
        render=_render_trafficcamnet,
        inference_backend="tensorrt_int8",
        artifact_layout="image_builtin",
        engine_location="work_dir",
        engine_filename="trafficcamnet_b1_gpu0_int8.engine",
        parse_bbox_func=None,
        custom_parser_lib=None,
        labels_source="image",
    ),
    YOLOX_DECODER: DecoderSpec(
        decoder_key=YOLOX_DECODER,
        render=_render_yolox,
        inference_backend="tensorrt_fp16",
        artifact_layout="host_model_store",
        engine_location="model_dir",
        engine_filename=None,
        parse_bbox_func="NvDsInferParseYolox",
        custom_parser_lib=_YOLOX_PARSER_LIB_FILENAME,
        labels_source="registry_generated",
    ),
    ULTRALYTICS_DECODER: DecoderSpec(
        decoder_key=ULTRALYTICS_DECODER,
        render=_render_ultralytics,
        inference_backend="tensorrt_fp16",
        artifact_layout="host_model_store",
        engine_location="model_dir",
        engine_filename=None,
        parse_bbox_func="NvDsInferParseUltralytics",
        custom_parser_lib=_ULTRALYTICS_PARSER_LIB_FILENAME,
        labels_source="registry_generated",
    ),
    PEOPLENET_DECODER: DecoderSpec(
        decoder_key=PEOPLENET_DECODER,
        render=_render_detectnet,
        inference_backend="tensorrt_fp16",
        artifact_layout="host_model_store",
        engine_location="model_dir",
        engine_filename=None,
        parse_bbox_func=None,  # built-in DetectNet_v2 parser
        custom_parser_lib=None,
        labels_source="registry_generated",
    ),
}


def engine_filename_for(decoder: DecoderSpec, onnx_file: str) -> str:
    """The TensorRT engine filename for this decoder: the fixed name when one
    is pinned (the per-run TrafficCamNet INT8 engine), else derived from the
    ONNX (the cached FP16 host-store case)."""
    if decoder.engine_filename is not None:
        return decoder.engine_filename
    return _fp16_engine_filename(onnx_file)


def get_decoder(decoder_key: str) -> DecoderSpec:
    """Look up a decoder by key, raising a clear ``DecoderRegistryError``
    (subclass of ``ValueError``) listing the known keys when it is unknown.
    Replaces the pre-Phase-0 trailing ``raise ValueError`` for an unsupported
    ``config_template`` in ``build_model_runtime_plan``."""
    try:
        return DECODER_REGISTRY[decoder_key]
    except KeyError:
        raise DecoderRegistryError(
            f"unknown decoder {decoder_key!r}; known decoders: {sorted(DECODER_REGISTRY)}"
        ) from None
