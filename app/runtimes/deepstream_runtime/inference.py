"""M8.2C1-B DeepStream nvinfer inference subprocess manager.

Runs the *exact* pipeline proven in M8.2C1-A
(``scripts/probe_deepstream_inference.sh``) as a managed asyncio subprocess:
a bounded ``deepstream-app`` running RTSP → ``nvstreammux`` → ``nvinfer``
(bundled TrafficCamNet sample model) → ``fakesink``, with no display, the
TensorRT engine built fresh from the sample ONNX, and detection evidence
written to a KITTI dump (``gie-kitti-output-dir``).

    docker run --rm --name <uniq> --gpus all --network <net> \
      -v <workdir>:<workdir> <ds8-image> \
      bash -lc "timeout -s INT <t> deepstream-app -c <workdir>/ds_app.txt"

Inference only — **no pyds, no NvDsBatchMeta, no metadata adapter, no
observations**. The RTSP URL is written ONLY into the per-run temp
``ds_app.txt`` under the system tempdir (never committed, cleaned up by
default) and into the subprocess argv; it is never logged (logs use the
redacted form). KITTI counts are status/debug evidence only — *not* metadata
extraction.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import math
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

from app.packages.registries import (
    RUNNABLE_COMPONENT_STATUSES,
    ModelSpec,
    find_model_by_alias,
    find_runnable_deepstream_tracker_by_algorithm,
    load_model_registry,
    runnable_deepstream_tracker_algorithms,
)
from app.redact import redact_rtsp

_log = logging.getLogger("upstream_processor.deepstream.infer")


def _runnable_deepstream_model_names() -> list[str]:
    """All operator-facing names (aliases + ids) that resolve to a runnable
    DeepStream model — for error messages."""
    names: list[str] = []
    for spec in load_model_registry().values():
        if spec.status in RUNNABLE_COMPONENT_STATUSES and spec.deepstream is not None:
            names.append(spec.model_id)
            names.extend(spec.aliases)
    return sorted(names)


def _resolve_runnable_model_spec(model: str) -> ModelSpec:
    """Registry lookup shared by ``resolve_model_dir`` and
    ``build_model_runtime_plan``. Raises ``ValueError`` for an unknown or
    not-yet-runnable (planned / deprecated) model so the worker posts
    ``failed`` with a clear message rather than launching a misconfigured
    pipeline."""
    spec = find_model_by_alias(model)
    if spec is None:
        raise ValueError(
            f"unknown deepstream inference model {model!r}; "
            f"known runnable models: {_runnable_deepstream_model_names()}"
        )
    if spec.status not in RUNNABLE_COMPONENT_STATUSES:
        raise ValueError(
            f"unknown deepstream inference model {model!r}: registry entry "
            f"{spec.model_id!r} is status {spec.status!r} and not runnable; "
            f"known runnable models: {_runnable_deepstream_model_names()}"
        )
    if spec.deepstream is None:
        raise ValueError(
            f"unknown deepstream inference model {model!r}: registry entry "
            f"{spec.model_id!r} has no DeepStream deployment spec"
        )
    return spec


def resolve_model_dir(model: str) -> str:
    """Map the configured model name to its in-container model dir, via the
    Processing P5 model registry (``model_registry.json``). For the proven
    TrafficCamNet aliases this returns exactly the path the pre-registry code
    hardcoded (the symlinked DeepStream sample dir, valid in both 7.1 and 8.0
    images — the M8.2C1-A probe built the engine from exactly this path).

    Note: for host-mounted models (``artifact_subdir`` set) this returns the
    DEFAULT store location; ``build_model_runtime_plan`` is the settings-aware
    resolution used by the metadata-probe runtime."""
    spec = _resolve_runnable_model_spec(model)
    assert spec.deepstream is not None  # guaranteed by the resolver
    return spec.deepstream.model_artifact_path


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


# --- P5-B YOLOX model runtime plan --------------------------------------

# Filename conventions inside a YOLOX model store dir (created by the
# operator-run scripts/prepare_yolox_model.sh; never committed).
_YOLOX_ONNX_FILENAME = "yolox_s.onnx"
_YOLOX_PARSER_LIB_FILENAME = "libnvdsparsebbox_yolox.so"
_YOLOX_ENGINE_FILENAME = "yolox_s_b1_gpu0_fp16.engine"
# P5-K: the Ultralytics (YOLO11 / YOLOv8) family shares one custom decode
# parser, compiled by scripts/prepare_ultralytics_model.sh into the model dir.
_ULTRALYTICS_PARSER_LIB_FILENAME = "libnvdsparsebbox_ultralytics.so"

# Registry ``deepstream.config_template`` values this runtime knows how to
# render. Anything else fails loudly at plan time.
_TRAFFICCAMNET_TEMPLATE = "builtin_trafficcamnet_int8_v1"
_YOLOX_TEMPLATE = "yolox_onnx_fp16_v1"
# P5-K templates (in-repo prep; not yet GPU-verified):
_ULTRALYTICS_TEMPLATE = "ultralytics_onnx_fp16_v1"  # YOLO11 / YOLOv8 (custom parser)
_PEOPLENET_TEMPLATE = "builtin_detectnet_peoplenet_v1"  # NGC PeopleNet (builtin detectnet parser)


def _fp16_engine_filename(onnx_file: str) -> str:
    """``<stem>_b1_gpu0_fp16.engine`` for an ONNX filename. For the original
    ``yolox_s.onnx`` this reproduces ``_YOLOX_ENGINE_FILENAME`` exactly."""
    stem = onnx_file[:-5] if onnx_file.endswith(".onnx") else onnx_file
    return f"{stem}_b1_gpu0_fp16.engine"


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


@dataclass(frozen=True)
class ModelRuntimePlan:
    """Everything the metadata process manager needs to run one registry
    model: the rendered PGIE config, the labels file body (when the model's
    labels are registry-generated rather than baked into the image), extra
    container bind mounts, and the host files that must exist before start.
    Built by ``build_model_runtime_plan``; pure data."""

    model_id: str
    model_family: str
    model_dir: str
    pgie_config: str
    parser: str
    inference_backend: str
    labels_body: str | None
    extra_mount_dirs: tuple[str, ...]
    required_files: tuple[str, ...]
    image_builtin: bool


def build_model_runtime_plan(
    *,
    model: str,
    work_dir: str,
    model_store_dir: str,
    confidence_threshold: float | None = None,
) -> ModelRuntimePlan:
    """Resolve a registry model into a concrete runtime plan for the pyds
    metadata pipeline. Dispatches on the registry entry's
    ``deepstream.config_template``:

    - ``builtin_trafficcamnet_int8_v1`` — byte-identical to the pre-P5-B
      behavior: image-builtin sample model, INT8 engine built per-run into
      the temp work dir, no extra mounts, labels from the image.
    - ``yolox_onnx_fp16_v1`` — host-mounted artifacts under
      ``<model_store_dir>/<artifact_subdir>`` (official raw ONNX + the
      compiled custom parser, both prepared by the operator script); the
      TensorRT FP16 engine is CACHED in that dir across runs; labels are
      generated from the registry's ``supported_classes`` (COCO order) into
      the work dir.

    Raises ``ValueError`` (worker posts ``failed``) for unknown/not-runnable
    models or unsupported templates. Missing-artifact checks are left to the
    process manager (host I/O) using ``required_files``.
    """
    spec = _resolve_runnable_model_spec(model)
    ds = spec.deepstream
    assert ds is not None  # guaranteed by the resolver

    if ds.config_template == _TRAFFICCAMNET_TEMPLATE:
        model_dir = ds.model_artifact_path
        engine_path = f"{work_dir}/trafficcamnet_b1_gpu0_int8.engine"
        return ModelRuntimePlan(
            model_id=spec.model_id,
            model_family=spec.family,
            model_dir=model_dir,
            pgie_config=build_pgie_config(
                model_dir=model_dir,
                engine_path=engine_path,
                confidence_threshold=confidence_threshold,
            ),
            parser=ds.parser,
            inference_backend="tensorrt_int8",
            labels_body=None,
            extra_mount_dirs=(),
            required_files=(),
            image_builtin=True,
        )

    if ds.config_template == _YOLOX_TEMPLATE:
        subdir = ds.artifact_subdir or spec.model_id
        model_dir = f"{model_store_dir.rstrip('/')}/{subdir}"
        labels_path = f"{work_dir}/labels.txt"
        # P5-K: YOLOX sizes share this template + the grid-decode parser; only
        # the ONNX/engine filename differs (``onnx_file``). For yolox_s_v1
        # (onnx_file unset) this is byte-identical to the pre-P5-K plan.
        onnx_file = ds.onnx_file or _YOLOX_ONNX_FILENAME
        engine_file = _fp16_engine_filename(onnx_file)
        return ModelRuntimePlan(
            model_id=spec.model_id,
            model_family=spec.family,
            model_dir=model_dir,
            pgie_config=build_yolox_pgie_config(
                model_dir=model_dir,
                engine_path=f"{model_dir}/{engine_file}",
                labels_path=labels_path,
                num_classes=len(spec.supported_classes),
                confidence_threshold=confidence_threshold,
                onnx_file=onnx_file,
            ),
            parser=ds.parser,
            inference_backend="tensorrt_fp16",
            labels_body="\n".join(spec.supported_classes) + "\n",
            extra_mount_dirs=(model_dir,),
            required_files=(
                f"{model_dir}/{onnx_file}",
                f"{model_dir}/{_YOLOX_PARSER_LIB_FILENAME}",
            ),
            image_builtin=False,
        )

    if ds.config_template == _ULTRALYTICS_TEMPLATE:
        # P5-K (in-repo prep; parser NOT yet GPU-validated). Host-mounted
        # Ultralytics ONNX + the shared custom decode parser.
        subdir = ds.artifact_subdir or spec.model_id
        model_dir = f"{model_store_dir.rstrip('/')}/{subdir}"
        labels_path = f"{work_dir}/labels.txt"
        onnx_file = ds.onnx_file or f"{spec.model_id}.onnx"
        engine_file = _fp16_engine_filename(onnx_file)
        return ModelRuntimePlan(
            model_id=spec.model_id,
            model_family=spec.family,
            model_dir=model_dir,
            pgie_config=build_ultralytics_pgie_config(
                model_dir=model_dir,
                engine_path=f"{model_dir}/{engine_file}",
                labels_path=labels_path,
                num_classes=len(spec.supported_classes),
                onnx_file=onnx_file,
                confidence_threshold=confidence_threshold,
            ),
            parser=ds.parser,
            inference_backend="tensorrt_fp16",
            labels_body="\n".join(spec.supported_classes) + "\n",
            extra_mount_dirs=(model_dir,),
            required_files=(
                f"{model_dir}/{onnx_file}",
                f"{model_dir}/{_ULTRALYTICS_PARSER_LIB_FILENAME}",
            ),
            image_builtin=False,
        )

    if ds.config_template == _PEOPLENET_TEMPLATE:
        # P5-K (in-repo prep; NOT yet GPU-verified). Host-mounted NGC PeopleNet
        # ONNX through the built-in DetectNet_v2 parser (no custom lib).
        subdir = ds.artifact_subdir or spec.model_id
        model_dir = f"{model_store_dir.rstrip('/')}/{subdir}"
        labels_path = f"{work_dir}/labels.txt"
        onnx_file = ds.onnx_file or f"{spec.model_id}.onnx"
        engine_file = _fp16_engine_filename(onnx_file)
        return ModelRuntimePlan(
            model_id=spec.model_id,
            model_family=spec.family,
            model_dir=model_dir,
            pgie_config=build_detectnet_pgie_config(
                model_dir=model_dir,
                engine_path=f"{model_dir}/{engine_file}",
                labels_path=labels_path,
                num_classes=len(spec.supported_classes),
                onnx_file=onnx_file,
                confidence_threshold=confidence_threshold,
            ),
            parser=ds.parser,
            inference_backend="tensorrt_fp16",
            labels_body="\n".join(spec.supported_classes) + "\n",
            extra_mount_dirs=(model_dir,),
            required_files=(f"{model_dir}/{onnx_file}",),
            image_builtin=False,
        )

    raise ValueError(
        f"deepstream inference model {model!r}: registry config_template "
        f"{ds.config_template!r} is not supported by this runtime"
    )


def build_app_config(
    *,
    rtsp_url: str,
    pgie_path: str,
    kitti_dir: str,
    width: int,
    height: int,
    rtsp_transport: str,
) -> str:
    """deepstream-app config: RTSP source → nvinfer → fakesink. No
    tiled-display, no OSD, no tracker (no X11). KITTI dump = detection
    evidence. The raw RTSP URL appears only in this temp file. Mirrors
    M8.2C1-A."""
    # deepstream-app source `select-rtp-protocol`: 4 = TCP, 0 = UDP/any.
    rtp_proto = 4 if rtsp_transport.strip().lower() == "tcp" else 0
    return (
        "[application]\n"
        "enable-perf-measurement=1\n"
        "perf-measurement-interval-sec=5\n"
        f"gie-kitti-output-dir={kitti_dir}\n"
        "[tiled-display]\n"
        "enable=0\n"
        "[source0]\n"
        "enable=1\n"
        "type=4\n"
        f"uri={rtsp_url}\n"
        "num-sources=1\n"
        "gpu-id=0\n"
        "cudadec-memtype=0\n"
        f"select-rtp-protocol={rtp_proto}\n"
        "latency=200\n"
        "[streammux]\n"
        "gpu-id=0\n"
        "batch-size=1\n"
        "batched-push-timeout=40000\n"
        f"width={width}\n"
        f"height={height}\n"
        "live-source=1\n"
        "nvbuf-memory-type=0\n"
        "[primary-gie]\n"
        "enable=1\n"
        "gpu-id=0\n"
        "gie-unique-id=1\n"
        "batch-size=1\n"
        f"config-file={pgie_path}\n"
        "[sink0]\n"
        "enable=1\n"
        "type=1\n"
        "sync=0\n"
        "[osd]\n"
        "enable=0\n"
        "[tracker]\n"
        "enable=0\n"
        "[tests]\n"
        "file-loop=0\n"
    )


# --- P3 tracker-probe builders -----------------------------------------


def render_tracker_ll_config(settings: dict[str, dict[str, Any]]) -> str:
    """Render a tracker profile's ``settings`` (section -> key -> scalar,
    from ``tracker_registry.json``) into the nvtracker low-level YAML body.

    DS 8.0 nvtracker parses %YAML:1.0 (note the colon, not a space) — keep
    the prefix exactly. Section/key order follows the registry JSON order,
    which mirrors the SDK sample. Pure / side-effect free."""
    lines = ["%YAML:1.0\n"]
    for section, values in settings.items():
        lines.append(f"{section}:\n")
        for key, value in values.items():
            lines.append(f"  {key}: {value}\n")
    return "".join(lines)


def build_tracker_config(algorithm: str = "iou") -> str:
    """Low-level tracker YAML config for ``nvtracker``, resolved from the
    Processing P5 tracker registry (``tracker_registry.json``).

    The DS 8.0 sample IOU config (``config_tracker_IOU.yml``) is the upstream
    template; the registry's ``deepstream_iou_fast_v1`` profile ships our own
    minimal copy so the worker is self-contained and tunables live in this
    repo (rendered byte-identically to the pre-registry hardcoded YAML).
    Required by ``nvtracker`` via the ``ll-config-file`` property. Planned
    profiles (NvDCF / NvSORT — no authored, live-verified settings yet) are
    rejected exactly like unknown algorithms. Pure / side-effect free."""
    profile = find_runnable_deepstream_tracker_by_algorithm(algorithm)
    if profile is None:
        raise ValueError(
            f"unknown deepstream tracker algorithm {algorithm!r}; "
            f"runnable profiles support: {runnable_deepstream_tracker_algorithms()}"
        )
    return render_tracker_ll_config(profile.settings)


# --- P3.5 analytics-probe config builder -------------------------------

# Stream id (per-stream sections key off this). v0 only batches a single
# RTSP source so 0 is the only stream index ever used; making it a named
# constant keeps the section names tidy and lets future multi-stream
# work parameterize it without breaking tests.
_ANALYTICS_STREAM_ID = 0

# v0 ships one ROI ("entry-roi") and one line ("entry-line"). Both are
# parameterized but the public knob surface stays tiny: width / height of
# the frame the geometry was authored against. The dashboard-driven /
# operator-authored config generator is a later milestone.
_DEFAULT_ROI_LABEL = "entry-roi"
_DEFAULT_LINE_LABEL = "entry-line"


def normalize_analytics_class_ids(class_ids: str) -> str:
    """Validate + normalize the operator-facing
    ``UPSTREAM_DEEPSTREAM_ANALYTICS_CLASS_IDS`` value into the
    integer string nvdsanalytics' ``class-id=`` line expects.

    Accepted inputs:
      - ``"all"`` / ``""`` / ``"-1"`` → ``"-1"`` (any class counted)
      - Single integer like ``"0"`` / ``"2"`` → unchanged
      - Anything else → ``ValueError`` (multi-class lists are NOT
        supported — nvdsanalytics' parser takes a single int per
        section, so a comma-list would require multiple sections,
        which is a later milestone)."""
    raw = (class_ids or "").strip().lower()
    if raw in ("", "all", "-1"):
        return "-1"
    # Reject comma-lists explicitly so a typo produces a clear error
    # rather than a silently-mistuned filter.
    if "," in raw:
        raise ValueError(
            f"deepstream_analytics_class_ids: comma-lists are not supported "
            f"by nvdsanalytics' single-section class-id key; got {class_ids!r}. "
            "Use 'all' for every class or a single integer for one class."
        )
    try:
        _ = int(raw)
    except ValueError as exc:
        raise ValueError(
            f"deepstream_analytics_class_ids must be 'all', '-1', or a single integer; "
            f"got {class_ids!r}"
        ) from exc
    return raw


@dataclass(frozen=True)
class GeometryPoint:
    """One pixel-coordinate vertex (origin top-left)."""

    x: int
    y: int

    def to_json(self) -> dict[str, int]:
        return {"x": self.x, "y": self.y}


@dataclass(frozen=True)
class RoiGeometry:
    """One analytics ROI as an ordered polygon (clockwise from top-left)."""

    id: str
    label: str
    points: list[GeometryPoint]

    def to_json(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "points": [p.to_json() for p in self.points],
        }


@dataclass(frozen=True)
class LineGeometry:
    """One crossing line as its two segment endpoints. The nvdsanalytics
    direction-reference points are an LC implementation detail and are
    deliberately excluded — a dashboard draws only the segment."""

    id: str
    label: str
    points: list[GeometryPoint]

    def to_json(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "points": [p.to_json() for p in self.points],
        }


@dataclass(frozen=True)
class AnalyticsGeometry:
    """The ROI/line geometry the analytics probe is configured with — the
    single source shared by ``build_analytics_config`` (the nvdsanalytics
    INI) and the P5-C perception-snapshot builder."""

    rois: list[RoiGeometry]
    lines: list[LineGeometry]

    def to_json(self) -> dict[str, Any]:
        return {
            "rois": [r.to_json() for r in self.rois],
            "lines": [line.to_json() for line in self.lines],
        }


def compute_analytics_geometry(
    *,
    frame_width: int = 1280,
    frame_height: int = 720,
    roi_margin_pct: float = 0.05,
    roi_label: str = _DEFAULT_ROI_LABEL,
    line_label: str = _DEFAULT_LINE_LABEL,
) -> AnalyticsGeometry:
    """The v0 analytics geometry: one near-full-frame ROI + one horizontal
    crossing line. Pure / side-effect free. Single source of the geometry
    numbers — ``build_analytics_config`` renders them into the nvdsanalytics
    INI, and the P5-C perception-snapshot builder serializes them as JSON for
    the dashboard overlay."""
    if frame_width <= 0 or frame_height <= 0:
        raise ValueError(
            f"frame_width and frame_height must be > 0; got {frame_width}x{frame_height}"
        )
    if not roi_label or not line_label:
        raise ValueError("roi_label and line_label must be non-empty")
    if not 0.0 <= roi_margin_pct < 0.5:
        raise ValueError(f"roi_margin_pct must be in [0.0, 0.5); got {roi_margin_pct}")
    # Near-full-frame ROI: default ``margin=0.05`` → 5%..95% on both axes.
    # Polygon points clockwise from top-left as nvdsanalytics expects.
    rx0 = int(frame_width * roi_margin_pct)
    rx1 = int(frame_width * (1.0 - roi_margin_pct))
    ry0 = int(frame_height * roi_margin_pct)
    ry1 = int(frame_height * (1.0 - roi_margin_pct))
    roi = RoiGeometry(
        id=roi_label,
        label=roi_label,
        points=[
            GeometryPoint(rx0, ry0),
            GeometryPoint(rx1, ry0),
            GeometryPoint(rx1, ry1),
            GeometryPoint(rx0, ry1),
        ],
    )
    # Horizontal line across the middle of the ROI — the two segment
    # endpoints only (the LC direction-reference points stay INI-side).
    line_y = frame_height // 2
    lx0 = int(frame_width * 0.25)
    lx1 = int(frame_width * 0.75)
    line = LineGeometry(
        id=line_label,
        label=line_label,
        points=[GeometryPoint(lx0, line_y), GeometryPoint(lx1, line_y)],
    )
    return AnalyticsGeometry(rois=[roi], lines=[line])


def _line_perpendicular_reference_points(
    p1: GeometryPoint,
    p2: GeometryPoint,
    *,
    frame_width: int,
    frame_height: int,
    offset: int = 80,
) -> tuple[GeometryPoint, GeometryPoint]:
    """The two direction-reference points nvdsanalytics needs for a line.

    Perpendicular to the line segment, on opposite sides of its midpoint, at
    ``offset`` pixels (clamped to the frame). For the horizontal probe-default
    line this reduces to ``(midpoint - (0, 80), midpoint + (0, 80))`` —
    byte-identical to the prior hardcoded math, which the P5-C golden test
    pins. For diagonal operator-drawn lines the reference points now align
    with the line's normal instead of being stuck on the frame's vertical
    axis, which is required for any future ``mode=strict`` direction work.
    """
    mid_x = (p1.x + p2.x) // 2
    mid_y = (p1.y + p2.y) // 2
    dx = p2.x - p1.x
    dy = p2.y - p1.y
    length = math.sqrt(dx * dx + dy * dy) or 1.0
    # Rotate (dx, dy) by 90° CCW → (-dy, dx); scale to ``offset`` px length.
    norm_x = -dy / length * offset
    norm_y = dx / length * offset
    ref0_x = max(0, min(frame_width - 1, int(mid_x - norm_x)))
    ref0_y = max(0, min(frame_height - 1, int(mid_y - norm_y)))
    ref1_x = max(0, min(frame_width - 1, int(mid_x + norm_x)))
    ref1_y = max(0, min(frame_height - 1, int(mid_y + norm_y)))
    return GeometryPoint(ref0_x, ref0_y), GeometryPoint(ref1_x, ref1_y)


def build_analytics_config(
    *,
    frame_width: int = 1280,
    frame_height: int = 720,
    geometry: AnalyticsGeometry | None = None,
    roi_label: str = _DEFAULT_ROI_LABEL,
    line_label: str = _DEFAULT_LINE_LABEL,
    roi_margin_pct: float = 0.05,
    class_ids: str = "all",
) -> str:
    """``nvdsanalytics`` config (INI-style) for the supplied ROIs + line
    crossings. Pure / side-effect free so it is unit-testable without
    DeepStream.

    Two call modes:

    - ``geometry=None`` (legacy): build a probe-default geometry from
      ``roi_margin_pct`` / ``roi_label`` / ``line_label`` via
      ``compute_analytics_geometry``. Preserves the pre-P6-C behavior the
      probe tests still exercise.
    - ``geometry=...`` (P6-C): use the supplied ``AnalyticsGeometry``
      directly, emitting one ``roi-<id>=`` per ROI and one
      ``line-crossing-<id>=`` per line under the single
      ``[roi-filtering-stream-0]`` / ``[line-crossing-stream-0]`` sections
      that nvdsanalytics expects. ``roi_margin_pct`` / ``roi_label`` /
      ``line_label`` are ignored in this mode.

    The ROI is a near-full-frame rectangle (covers ~90% x 90% by
    default after the 5% margin on each side) — wide enough that an
    operator's bbox bottom-center (nvdsanalytics' reference point
    for ROI containment) is inside even when the operator stands at
    the bottom edge of the frame. The line is horizontal across the
    middle of the ROI, so walking through the ROI also crosses the
    line. Coordinates are in pixels of the ``config-width`` /
    ``config-height`` (set to the nvstreammux output).

    ``class_ids`` controls the per-section class filter (see
    ``normalize_analytics_class_ids``).
    """
    if geometry is None:
        geometry = compute_analytics_geometry(
            frame_width=frame_width,
            frame_height=frame_height,
            roi_margin_pct=roi_margin_pct,
            roi_label=roi_label,
            line_label=line_label,
        )
    class_id_value = normalize_analytics_class_ids(class_ids)

    roi_lines: list[str] = []
    for roi in geometry.rois:
        # ROI polygon — clockwise from top-left (see compute_analytics_geometry).
        # The DS 8.0 nvdsanalytics property parser keys ROIs off "roi-"-prefixed
        # properties: the label is the suffix after "roi-" in the KEY name
        # (NOT in the value); the value is the polygon point list.
        # See sources/gst-plugins/gst-nvdsanalytics/nvdsanalytics_property_parser.cpp
        # (DSANALYTICS_PROPERTY_ROI = "roi-"). Verified against the DS 8.0
        # sample at sample_apps/deepstream-nvdsanalytics-test/config_nvdsanalytics.txt
        # which uses ``roi-RF=`` with "RF" as the label.
        points = ";".join(f"{p.x};{p.y}" for p in roi.points)
        roi_lines.append(f"roi-{roi.id}={points}\n")

    lc_lines: list[str] = []
    for line in geometry.lines:
        # nvdsanalytics LC requires the 2 line-segment endpoints PLUS 2
        # direction-reference point pairs:
        #   line_x0;line_y0;line_x1;line_y1;dir_x0;dir_y0;dir_x1;dir_y1
        # The reference points are perpendicular to the segment so the math
        # works for any line orientation (not just the horizontal probe-default).
        endpoints = line.points
        ref0, ref1 = _line_perpendicular_reference_points(
            endpoints[0],
            endpoints[1],
            frame_width=frame_width,
            frame_height=frame_height,
        )
        lc_points = (
            ";".join(f"{p.x};{p.y}" for p in endpoints) + f";{ref0.x};{ref0.y};{ref1.x};{ref1.y}"
        )
        # line-crossing-<Label>=<line-coords>;<direction-coords> — label is
        # part of the key (parser strips the "line-crossing-" prefix).
        lc_lines.append(f"line-crossing-{line.id}={lc_points}\n")

    return (
        "[property]\n"
        "enable=1\n"
        f"config-width={int(frame_width)}\n"
        f"config-height={int(frame_height)}\n"
        # osd-mode=0: don't draw anything (we render to fakesink anyway).
        "osd-mode=0\n"
        "display-font-size=12\n"
        f"[roi-filtering-stream-{_ANALYTICS_STREAM_ID}]\n"
        "enable=1\n"
        + "".join(roi_lines)
        # inverse-roi=0: count objects INSIDE the polygon (vs outside).
        + "inverse-roi=0\n"
        # P4-D follow-up: class filter is env-driven via
        # ``UPSTREAM_DEEPSTREAM_ANALYTICS_CLASS_IDS``. Default ``-1``
        # (all classes) — the live run showed TrafficCamNet sometimes
        # misclassifies people as cars, so probe-default-all is the
        # safe choice. Production scenes can narrow per-deployment.
        + f"class-id={class_id_value}\n"
        + f"[line-crossing-stream-{_ANALYTICS_STREAM_ID}]\n"
        + "enable=1\n"
        + "".join(lc_lines)
        # Same class filter as the ROI section.
        + f"class-id={class_id_value}\n"
        + "extended=0\n"
        # mode=loose: count any crossing without strict direction
        # adherence; we want to prove events flow, not enforce direction.
        # P6-C preserves operator ``direction: "in"|"out"`` in observation
        # lineage only — strict-direction enforcement is a follow-up.
        + "mode=loose\n"
    )


def build_inference_docker_command(
    *,
    work_dir: str,
    image: str,
    network: str,
    probe_timeout_seconds: float,
    container_name: str,
    gpus: str = "all",
) -> list[str]:
    """The docker argv for the bounded inference run. Pure/side-effect free
    so it is unit-testable without Docker. Mirrors the proven M8.2C1-A
    invocation (``--network``, work dir bind-mounted at the same path so the
    configs' absolute paths resolve inside the container). ``gpus`` is the
    literal ``docker run --gpus`` value ("all" or e.g. "device=1")."""
    inner = (
        f"timeout -s INT {int(probe_timeout_seconds)} deepstream-app -c {work_dir}/ds_app.txt 2>&1"
    )
    return [
        "docker", "run", "--rm", "--name", container_name,
        "--gpus", gpus, "--network", network,
        "-v", f"{work_dir}:{work_dir}", image,
        "bash", "-lc", inner,
    ]  # fmt: skip


class DeepStreamInferenceProcessManager:
    """Real Docker-backed ``ProcessManager`` for the inference probe.

    Owns a per-run temp work dir (deepstream-app + PGIE config + KITTI output
    dir), bind-mounted into the DeepStream 8.0 container. Streams stdout,
    supports bounded graceful stop + SIGKILL-after-timeout, exposes KITTI
    detection counts, and removes the temp dir on stop unless
    ``keep_temp_files`` is set. The RTSP URL is only in the temp config +
    subprocess argv; logs use the redacted form."""

    def __init__(
        self,
        *,
        rtsp_url: str,
        image: str,
        network: str,
        rtsp_transport: str,
        probe_timeout_seconds: float,
        stop_timeout_seconds: float,
        model: str,
        output_base_dir: str,
        keep_temp_files: bool,
        width: int,
        height: int,
        gpus: str = "all",
    ) -> None:
        self._rtsp_url = rtsp_url
        self._image = image
        self._network = network
        self._rtsp_transport = rtsp_transport
        self._probe_timeout = probe_timeout_seconds
        self._stop_timeout = stop_timeout_seconds
        self._model = model
        self._output_base_dir = output_base_dir
        self._keep_temp_files = keep_temp_files
        self._width = width
        self._height = height
        self._gpus = gpus
        self._container = f"upstream-dsi-{uuid4().hex[:12]}"
        self._work_dir: Path | None = None
        self._proc: asyncio.subprocess.Process | None = None
        self._returncode: int | None = None

    # --- temp work dir ----------------------------------------------------

    def _prepare_work_dir(self) -> Path:
        model_dir = resolve_model_dir(self._model)  # may raise ValueError
        base = self._output_base_dir.strip() or None
        if base is not None:
            Path(base).mkdir(parents=True, exist_ok=True)
        work = Path(tempfile.mkdtemp(prefix="upstream-m82c1b.", dir=base))
        kitti = work / "kitti"
        kitti.mkdir(parents=True, exist_ok=True)
        pgie_path = work / "pgie.txt"
        app_path = work / "ds_app.txt"
        engine_path = work / "trafficcamnet_b1_gpu0_int8.engine"
        pgie_path.write_text(
            build_pgie_config(model_dir=model_dir, engine_path=str(engine_path)),
            encoding="utf-8",
        )
        # The raw RTSP URL goes ONLY into this temp file (never logged, never
        # committed; removed on stop unless keep_temp_files).
        app_path.write_text(
            build_app_config(
                rtsp_url=self._rtsp_url,
                pgie_path=str(pgie_path),
                kitti_dir=str(kitti),
                width=self._width,
                height=self._height,
                rtsp_transport=self._rtsp_transport,
            ),
            encoding="utf-8",
        )
        return work

    def _cleanup_work_dir(self) -> None:
        if self._work_dir is None or self._keep_temp_files:
            return
        with contextlib.suppress(OSError):
            shutil.rmtree(self._work_dir, ignore_errors=True)

    def kitti_stats(self) -> tuple[int, int]:
        """(frame_file_count, total_detection_lines) from the KITTI dump.

        Status/debug evidence only — NOT metadata extraction. Best-effort:
        returns ``(0, 0)`` if the dir is gone (e.g. already cleaned)."""
        if self._work_dir is None:
            return (0, 0)
        kitti = self._work_dir / "kitti"
        if not kitti.is_dir():
            return (0, 0)
        frames = 0
        detections = 0
        try:
            for f in kitti.glob("*.txt"):
                frames += 1
                with contextlib.suppress(OSError):
                    text = f.read_text(encoding="utf-8", errors="replace")
                    detections += sum(1 for line in text.splitlines() if line.strip())
        except OSError:
            return (frames, detections)
        return (frames, detections)

    # --- ProcessManager protocol -----------------------------------------

    def command(self) -> list[str]:
        assert self._work_dir is not None, "command() called before start()"
        return build_inference_docker_command(
            work_dir=str(self._work_dir),
            image=self._image,
            network=self._network,
            probe_timeout_seconds=self._probe_timeout,
            container_name=self._container,
            gpus=self._gpus,
        )

    async def start(self) -> None:
        self._work_dir = self._prepare_work_dir()
        argv = self.command()
        _log.info(
            "deepstream inference start: image=%s network=%s model=%s url=%s",
            self._image,
            self._network,
            self._model,
            redact_rtsp(self._rtsp_url),  # never log the real URL
        )
        self._proc = await asyncio.create_subprocess_exec(
            *argv,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )

    async def readline(self) -> str | None:
        if self._proc is None or self._proc.stdout is None:
            return None
        data = await self._proc.stdout.readline()
        if not data:
            self._returncode = await self._proc.wait()
            return None
        return data.decode("utf-8", "replace").rstrip("\n")

    async def stop(self) -> None:
        try:
            if self._proc is not None:
                try:
                    stopper = await asyncio.create_subprocess_exec(
                        "docker", "stop", "--time", str(int(self._stop_timeout)),
                        self._container,
                        stdout=asyncio.subprocess.DEVNULL,
                        stderr=asyncio.subprocess.DEVNULL,
                    )  # fmt: skip
                    await asyncio.wait_for(stopper.wait(), timeout=self._stop_timeout + 5)
                except (OSError, TimeoutError) as exc:
                    _log.warning("docker stop best-effort failed: %s", exc)

                if self._proc.returncode is None:
                    with contextlib.suppress(ProcessLookupError):
                        self._proc.terminate()
                    try:
                        await asyncio.wait_for(self._proc.wait(), timeout=self._stop_timeout)
                    except TimeoutError:
                        _log.warning(
                            "inference subprocess did not exit in %.0fs; killing",
                            self._stop_timeout,
                        )
                        with contextlib.suppress(ProcessLookupError):
                            self._proc.kill()
                        await self._proc.wait()
                self._returncode = self._proc.returncode
        finally:
            self._cleanup_work_dir()

    @property
    def returncode(self) -> int | None:
        return self._returncode
