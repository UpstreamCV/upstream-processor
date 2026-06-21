"""BYOM Phase 0 — the Model Descriptor contract.

A declarative, normalized description of one detection model: its input
geometry + preprocessing, its output/NMS decoding, its class map, precision,
the artifacts it needs, and its licensing provenance. This is the stable shape
later BYOM phases build on — automated prepare targets the ``artifact`` block,
the validation gate checks the ``input``/``output`` contract, and a future
per-account registry persists descriptors (DB-backed) instead of deriving them.

Phase 0 stance (no DB, no upload, no new JSON): descriptors are **generated
from the existing registry** via :func:`descriptor_from_spec`. Nothing about
the model registry, package IDs, or runtime bytes changes — the descriptor is
an additional, normalized *view* over ``ModelSpec`` that
``build_model_runtime_plan`` now routes through.

Important Phase-0 boundary: the descriptor *declares* preprocessing values
(scale, colour, topk, NMS, default confidence) but is NOT yet the thing that
emits config bytes — each family's exact byte-template still lives in its
decoder (see ``app/runtimes/deepstream_runtime/decoders.py``), because the four
families differ in nvinfer key *order and set*, not just values. A consistency
test asserts the declared values agree with what the decoder emits, so they
cannot silently diverge. Lifting the constants into a generic
descriptor-driven renderer is deferred to a later phase.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.packages.registries import ModelSpec

# Decoder keys (== registry ``deepstream.config_template`` values). Mirrored
# from ``decoders.DECODER_REGISTRY`` keys; a test asserts the two never
# diverge. Kept as local literals so this low-level package module does not
# import the higher-level deepstream runtime.
_TRAFFICCAMNET_DECODER = "builtin_trafficcamnet_int8_v1"
_YOLOX_DECODER = "yolox_onnx_fp16_v1"
_ULTRALYTICS_DECODER = "ultralytics_onnx_fp16_v1"
_PEOPLENET_DECODER = "builtin_detectnet_peoplenet_v1"

# --- descriptor enums ---------------------------------------------------

# Phase 0 supports detection only.
DESCRIPTOR_TASKS = frozenset({"detect"})
# Coarse decode families. ``decoder`` (a registry key) is the *precise*
# selector; ``family`` is the human/forward-looking grouping. The enum is
# intentionally wider than today's two values so later phases slot in without a
# schema change.
DESCRIPTOR_FAMILIES = frozenset(
    {
        "detectnet_v2",
        "yolo_anchorfree",
        "yolo_anchor_grid",
        "embedded_nms",
        "ssd_retinanet",
        "rtdetr",
    }
)
DESCRIPTOR_PRECISIONS = frozenset({"int8", "fp16"})
DESCRIPTOR_COLORS = frozenset({"RGB", "BGR"})
DESCRIPTOR_LAYOUTS = frozenset({"NCHW"})
DESCRIPTOR_ARTIFACT_LAYOUTS = frozenset({"image_builtin", "host_model_store"})
DESCRIPTOR_LABELS_SOURCES = frozenset({"image", "registry_generated"})


class DescriptorValidationError(ValueError):
    """A model descriptor is malformed (missing/invalid required field, or a
    model whose registry entry cannot be expressed as a Phase-0 descriptor).

    Subclasses ``ValueError`` so it flows through the same plan-failure path as
    the other ``build_model_runtime_plan`` errors."""


# --- descriptor sub-records ---------------------------------------------


@dataclass(frozen=True)
class DescriptorInput:
    """Input geometry + preprocessing. ``color_format`` is the raw nvinfer
    ``model-color-format`` value (0=RGB, 1=BGR); ``net_scale_factor`` is the
    nvinfer ``net-scale-factor``. ``mean``/``std`` are reserved (no current
    model emits per-channel offsets)."""

    width: int
    height: int
    channels: int
    layout: str
    color: str
    color_format: int
    net_scale_factor: float
    letterbox: bool
    pad_value: int | None
    mean: tuple[float, ...] | None = None
    std: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        for name in ("width", "height", "channels"):
            if getattr(self, name) <= 0:
                raise DescriptorValidationError(
                    f"descriptor input {name} must be > 0, got {getattr(self, name)!r}"
                )
        if self.layout not in DESCRIPTOR_LAYOUTS:
            raise DescriptorValidationError(
                f"descriptor input layout {self.layout!r} not in {sorted(DESCRIPTOR_LAYOUTS)}"
            )
        if self.color not in DESCRIPTOR_COLORS:
            raise DescriptorValidationError(
                f"descriptor input color {self.color!r} not in {sorted(DESCRIPTOR_COLORS)}"
            )
        if self.color_format not in (0, 1):
            raise DescriptorValidationError(
                f"descriptor input color_format must be 0 or 1, got {self.color_format!r}"
            )


@dataclass(frozen=True)
class DescriptorOutput:
    """Output decoding + NMS. ``decoder`` is the registry key that selects the
    byte-template; ``parse_bbox_func`` is the custom parser entry point (None
    for the built-in DetectNet parser); the thresholds mirror what the decoder
    emits. ``class0_threshold`` records the TrafficCamNet-only second
    ``[class-attrs-0]`` override block (None for every other family)."""

    nms_mode: str
    decoder: str
    parse_bbox_func: str | None
    topk: int
    nms_iou: float
    default_conf: float
    class0_threshold: float | None = None

    def __post_init__(self) -> None:
        if not self.decoder:
            raise DescriptorValidationError("descriptor output decoder must be non-empty")
        if self.topk <= 0:
            raise DescriptorValidationError(
                f"descriptor output topk must be > 0, got {self.topk!r}"
            )


@dataclass(frozen=True)
class DescriptorArtifact:
    """The local artifacts this model needs from the store. ``checksum`` is
    optional in Phase 0 — current prepared artifacts are not yet hashed; later
    phases populate it for content-addressed distribution."""

    onnx_filename: str
    artifact_layout: str
    model_artifact_path: str
    artifact_subdir: str | None
    labels_source: str
    custom_parser_lib: str | None = None
    checksum: str | None = None

    def __post_init__(self) -> None:
        if not self.onnx_filename:
            raise DescriptorValidationError("descriptor artifact onnx_filename must be non-empty")
        if self.artifact_layout not in DESCRIPTOR_ARTIFACT_LAYOUTS:
            raise DescriptorValidationError(
                f"descriptor artifact layout {self.artifact_layout!r} not in "
                f"{sorted(DESCRIPTOR_ARTIFACT_LAYOUTS)}"
            )
        if self.labels_source not in DESCRIPTOR_LABELS_SOURCES:
            raise DescriptorValidationError(
                f"descriptor artifact labels_source {self.labels_source!r} not in "
                f"{sorted(DESCRIPTOR_LABELS_SOURCES)}"
            )


@dataclass(frozen=True)
class DescriptorProvenance:
    """Licensing/provenance axis carried straight from the registry (drives the
    first-party gate in the catalog; see ``LICENSE_CLASSES``)."""

    license_class: str
    first_party: bool
    source: str
    notes: str = ""


# --- the descriptor -----------------------------------------------------


@dataclass(frozen=True)
class ModelDescriptor:
    """Declarative, normalized description of one detection model. Built from a
    registry ``ModelSpec`` by :func:`descriptor_from_spec`; pure data."""

    model_id: str
    task: str
    family: str
    decoder: str
    input: DescriptorInput
    output: DescriptorOutput
    class_map: dict[str, int]
    precision: str
    artifact: DescriptorArtifact
    provenance: DescriptorProvenance

    def __post_init__(self) -> None:
        if not self.model_id:
            raise DescriptorValidationError("descriptor model_id must be non-empty")
        if self.task not in DESCRIPTOR_TASKS:
            raise DescriptorValidationError(
                f"descriptor {self.model_id!r}: task {self.task!r} not in "
                f"{sorted(DESCRIPTOR_TASKS)}"
            )
        if self.family not in DESCRIPTOR_FAMILIES:
            raise DescriptorValidationError(
                f"descriptor {self.model_id!r}: family {self.family!r} not in "
                f"{sorted(DESCRIPTOR_FAMILIES)}"
            )
        if not self.decoder:
            raise DescriptorValidationError(
                f"descriptor {self.model_id!r}: decoder must be non-empty"
            )
        if self.precision not in DESCRIPTOR_PRECISIONS:
            raise DescriptorValidationError(
                f"descriptor {self.model_id!r}: precision {self.precision!r} not in "
                f"{sorted(DESCRIPTOR_PRECISIONS)}"
            )
        if not self.class_map:
            raise DescriptorValidationError(
                f"descriptor {self.model_id!r}: class_map must be non-empty"
            )
        if self.output.decoder != self.decoder:
            raise DescriptorValidationError(
                f"descriptor {self.model_id!r}: output.decoder {self.output.decoder!r} "
                f"must equal decoder {self.decoder!r}"
            )


# --- per-decoder profiles (the per-family invariants) -------------------


@dataclass(frozen=True)
class _TemplateProfile:
    """The family-level facts ``descriptor_from_spec`` needs to fill an input /
    output / artifact record from a registry entry. ``default_onnx`` is the
    ONNX filename when the registry leaves ``onnx_file`` unset; ``None`` means
    derive it from the model id (``f"{model_id}.onnx"``), reproducing the
    pre-Phase-0 fallback exactly."""

    family: str
    precision: str
    color: str
    color_format: int
    net_scale_factor: float
    letterbox: bool
    pad_value: int | None
    nms_mode: str
    parse_bbox_func: str | None
    topk: int
    nms_iou: float
    default_conf: float
    class0_threshold: float | None
    artifact_layout: str
    labels_source: str
    custom_parser_lib: str | None
    default_onnx: str | None


_ONE_OVER_255 = 0.00392156862745098

_TEMPLATE_PROFILES: dict[str, _TemplateProfile] = {
    _TRAFFICCAMNET_DECODER: _TemplateProfile(
        family="detectnet_v2",
        precision="int8",
        color="RGB",
        color_format=0,
        net_scale_factor=_ONE_OVER_255,
        letterbox=False,
        pad_value=None,
        nms_mode="cluster",
        parse_bbox_func=None,
        topk=20,
        nms_iou=0.5,
        default_conf=0.2,
        class0_threshold=0.4,
        artifact_layout="image_builtin",
        labels_source="image",
        custom_parser_lib=None,
        default_onnx="resnet18_trafficcamnet_pruned.onnx",
    ),
    _YOLOX_DECODER: _TemplateProfile(
        family="yolo_anchorfree",
        precision="fp16",
        color="BGR",
        color_format=1,
        net_scale_factor=1.0,
        letterbox=True,
        pad_value=0,
        nms_mode="cluster",
        parse_bbox_func="NvDsInferParseYolox",
        topk=300,
        nms_iou=0.45,
        default_conf=0.25,
        class0_threshold=None,
        artifact_layout="host_model_store",
        labels_source="registry_generated",
        custom_parser_lib="libnvdsparsebbox_yolox.so",
        default_onnx="yolox_s.onnx",
    ),
    _ULTRALYTICS_DECODER: _TemplateProfile(
        family="yolo_anchorfree",
        precision="fp16",
        color="RGB",
        color_format=0,
        net_scale_factor=_ONE_OVER_255,
        letterbox=True,
        pad_value=0,
        nms_mode="cluster",
        parse_bbox_func="NvDsInferParseUltralytics",
        topk=300,
        nms_iou=0.45,
        default_conf=0.25,
        class0_threshold=None,
        artifact_layout="host_model_store",
        labels_source="registry_generated",
        custom_parser_lib="libnvdsparsebbox_ultralytics.so",
        default_onnx=None,  # -> f"{model_id}.onnx"
    ),
    _PEOPLENET_DECODER: _TemplateProfile(
        family="detectnet_v2",
        precision="fp16",
        color="RGB",
        color_format=0,
        net_scale_factor=_ONE_OVER_255,
        letterbox=False,
        pad_value=None,
        nms_mode="cluster",
        parse_bbox_func=None,
        topk=20,
        nms_iou=0.5,
        default_conf=0.2,
        class0_threshold=None,
        artifact_layout="host_model_store",
        labels_source="registry_generated",
        custom_parser_lib=None,
        default_onnx=None,  # -> f"{model_id}.onnx"
    ),
}

# Registry ``task`` -> descriptor ``task``. Only detection has a DeepStream
# deployment today; other tasks have no descriptor in Phase 0.
_TASK_MAP = {"object_detection": "detect"}

# Reserved for callers that want to pre-validate without constructing a spec.
DESCRIPTOR_DECODERS = frozenset(_TEMPLATE_PROFILES)


def _resolve_onnx_filename(spec: ModelSpec, profile: _TemplateProfile) -> str:
    """Reproduce the pre-Phase-0 ONNX filename resolution exactly: an explicit
    registry ``onnx_file`` wins; else the family default; else
    ``f"{model_id}.onnx"`` (the Ultralytics / PeopleNet fallback)."""
    assert spec.deepstream is not None
    if spec.deepstream.onnx_file:
        return spec.deepstream.onnx_file
    if profile.default_onnx is not None:
        return profile.default_onnx
    return f"{spec.model_id}.onnx"


def descriptor_from_spec(spec: ModelSpec) -> ModelDescriptor:
    """Normalize a registry ``ModelSpec`` into a :class:`ModelDescriptor`.

    Strict on descriptor shape, but does not reject an existing model unless
    its registry entry genuinely cannot be expressed (no DeepStream spec,
    unknown ``config_template``, or a non-detection task)."""
    ds = spec.deepstream
    if ds is None:
        raise DescriptorValidationError(
            f"model {spec.model_id!r} has no DeepStream spec; cannot build a descriptor"
        )
    profile = _TEMPLATE_PROFILES.get(ds.config_template)
    if profile is None:
        raise DescriptorValidationError(
            f"model {spec.model_id!r}: config_template {ds.config_template!r} has no "
            f"descriptor profile (not supported by this runtime); "
            f"known: {sorted(_TEMPLATE_PROFILES)}"
        )
    task = _TASK_MAP.get(spec.task)
    if task is None:
        raise DescriptorValidationError(
            f"model {spec.model_id!r}: task {spec.task!r} is not a Phase-0 descriptor task"
        )

    onnx_filename = _resolve_onnx_filename(spec, profile)
    descriptor_input = DescriptorInput(
        width=spec.input["width"],
        height=spec.input["height"],
        channels=spec.input["channels"],
        layout="NCHW",
        color=profile.color,
        color_format=profile.color_format,
        net_scale_factor=profile.net_scale_factor,
        letterbox=profile.letterbox,
        pad_value=profile.pad_value,
    )
    descriptor_output = DescriptorOutput(
        nms_mode=profile.nms_mode,
        decoder=ds.config_template,
        parse_bbox_func=profile.parse_bbox_func,
        topk=profile.topk,
        nms_iou=profile.nms_iou,
        default_conf=profile.default_conf,
        class0_threshold=profile.class0_threshold,
    )
    artifact = DescriptorArtifact(
        onnx_filename=onnx_filename,
        artifact_layout=profile.artifact_layout,
        model_artifact_path=ds.model_artifact_path,
        artifact_subdir=ds.artifact_subdir,
        labels_source=profile.labels_source,
        custom_parser_lib=profile.custom_parser_lib,
        checksum=None,
    )
    provenance = DescriptorProvenance(
        license_class=spec.license_class,
        first_party=spec.first_party,
        source=spec.source,
        notes=spec.notes,
    )
    return ModelDescriptor(
        model_id=spec.model_id,
        task=task,
        family=profile.family,
        decoder=ds.config_template,
        input=descriptor_input,
        output=descriptor_output,
        class_map=dict(spec.class_id_map),
        precision=profile.precision,
        artifact=artifact,
        provenance=provenance,
    )
