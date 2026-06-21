# Model Descriptors & Decoders — developer notes (BYOM Phase 0)

Phase 0 made the curated model runtime **data-driven**. Two new modules sit
between the registry and the DeepStream config generator:

- `app/packages/model_descriptor.py` — the **Model Descriptor**: a declarative,
  normalized description of one detection model (input geometry + preprocessing,
  output/NMS decoding, class map, precision, artifacts, provenance). Generated
  from a registry `ModelSpec` by `descriptor_from_spec(spec)`. No DB, no new
  JSON — it's a normalized *view* over the existing registry.
- `app/runtimes/deepstream_runtime/decoders.py` — the **Decoder Registry**: a
  finite `dict[decoder_key → DecoderSpec]`. Each `DecoderSpec` owns one model
  family's exact PGIE byte-template (a `build_*_pgie_config` function) plus the
  facts needed to assemble the rest of the runtime plan (inference backend,
  engine location, artifact layout, custom parser `.so`, labels source).

`build_model_runtime_plan` (in `inference.py`) is now one data-driven body:

```
spec  = _resolve_runnable_model_spec(model)   # registry lookup (unchanged)
desc  = descriptor_from_spec(spec)            # ModelSpec → ModelDescriptor
dec   = get_decoder(desc.decoder)             # finite registry lookup
plan  = render + assemble(desc, dec, …)       # PGIE + paths + mounts + required files
```

The `decoder` key is, in Phase 0, exactly the registry `deepstream.config_template`
value. The four keys are: `builtin_trafficcamnet_int8_v1`, `yolox_onnx_fp16_v1`,
`ultralytics_onnx_fp16_v1`, `builtin_detectnet_peoplenet_v1`.

## Why the decoder keeps a byte-template (not a generic renderer)

The four current families do **not** differ only in values — they differ in
nvinfer **key order and key set** (TrafficCamNet places `model-color-format`
differently, has an `int8-calib-file` line and a second `[class-attrs-0]` block;
the YOLO families add `maintain-aspect-ratio`/`parse-bbox-func-name`; PeopleNet
uses the built-in parser). A single fixed-order renderer therefore could not
reproduce every family byte-for-byte.

So Phase 0 splits responsibility: the **descriptor supplies the values**
(onnx filename, class count, colour, scale, thresholds, …) and the **decoder
owns the exact key order/structure**. The descriptor *also declares* the same
preprocessing values, and `tests/test_model_descriptor.py` asserts the declared
values match what the decoder emits — so the two can't silently diverge.

## How to add a new model for a SUPPORTED family

Pure data — **no code here**. Add the model to `app/packages/model_registry.json`
with a `deepstream.config_template` equal to an existing decoder key, e.g. another
YOLOX size:

```jsonc
{
  "model_id": "yolox_l_v1",
  "family": "yolox",
  "task": "object_detection",
  "deepstream": {
    "config_template": "yolox_onnx_fp16_v1",   // existing decoder key
    "onnx_file": "yolox_l.onnx",               // per-model artifact name
    "artifact_subdir": "yolox_l_v1",
    "parser": "NvDsInferParseYolox (…)",
    "requires_custom_parser": true,
    …
  },
  …
}
```

`descriptor_from_spec` picks up the new entry automatically; the consistency +
descriptor tests cover it (they iterate every registered model). Verify the new
model's generated PGIE config is what you expect and add it to the golden table
in `tests/test_config_parity.py` if you want a pinned byte-level assertion.

## How to add a new DECODER family later

When a model needs a genuinely different output layout (e.g. embedded-NMS
EfficientNMS output, RT-DETR, SSD/RetinaNet), add **one** decoder — not a branch
in `build_model_runtime_plan`:

1. In `decoders.py`, write the family's `build_<family>_pgie_config(...)` (or, if
   it can reuse an existing renderer, skip this) and a `_render_<family>(ctx)`
   adapter that maps a `PgieRenderContext` onto it.
2. Add a `DecoderSpec` entry to `DECODER_REGISTRY` keyed by the new
   `config_template` string, filling in `inference_backend`, `artifact_layout`
   (`image_builtin` | `host_model_store`), `engine_location` (`work_dir` |
   `model_dir`), `engine_filename` (fixed name or `None` to derive),
   `parse_bbox_func`, `custom_parser_lib`, `labels_source`.
3. In `model_descriptor.py`, add a `_TemplateProfile` to `_TEMPLATE_PROFILES`
   keyed by the same `config_template` string (the per-family declarative
   profile: colour, scale, letterbox, topk, nms, default confidence, default
   onnx name, …) and, if the family is a new coarse grouping, add it to
   `DESCRIPTOR_FAMILIES`.
4. Add the family to the golden table in `tests/test_config_parity.py`.

`build_model_runtime_plan` does not change.

## Intentionally deferred to later phases

Phase 0 is a setup refactor, **not** the BYOM product surface. Not implemented
here (see `byom-architecture.md` for the phased plan):

- A fully descriptor-emitted **generic renderer** that lifts the per-family
  constants out of the byte-templates (needs a golden corpus to land against).
- **Artifact checksums** (`DescriptorArtifact.checksum` exists but is `None` —
  current prepared artifacts are not hashed yet) and content-addressed storage.
- **Per-account / DB-backed descriptors**, upload flows, the automated build &
  validation pipeline, artifact distribution.
- **Triton / `nvinferserver`** serving, embedded-NMS, op-allowlists.
- New model **families/tasks** beyond representing today's detectors.
