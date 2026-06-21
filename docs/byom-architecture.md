# BYOM Model Serving — Architecture & Implementation Plan

**Status:** Proposed (design). **Scope:** cross-repo (upstream-processor, upstream-api-server, upstream-dashboard).
**Owner:** _TBD_. **Last updated:** 2026-06-19.

> Goal: let a customer bring their own detection model — upload weights, we ingest /
> validate / prepare / serve it — for the **standard, popular architectures** people
> actually use (YOLO family, Ultralytics, RT-DETR, SSD/RetinaNet, NVIDIA TAO nets),
> running efficiently on shared GPUs. "Literally any model" is asymptotic; this design
> targets the 90% that "the popular ones" covers and **fails safely** (rejects at
> validation) for the rest.

---

## 1. Baseline (where we are today)

The processor runs models through a **curated, registry-driven, template-dispatched** path:

- Models live in static, hand-curated registries (`app/packages/model_registry.json`,
  api-side mirror `services/package_catalog.py` / `services/models.py`). **No per-account
  models, no DB table, no upload path.**
- `app/runtimes/deepstream_runtime/inference.py::build_model_runtime_plan` dispatches on
  **4 hardcoded `config_template` handlers** (`builtin_trafficcamnet_int8_v1`,
  `yolox_onnx_fp16_v1`, `ultralytics_onnx_fp16_v1`, `builtin_detectnet_peoplenet_v1`),
  each welded to a specific **decode parser** (`NvDsInferParseYolox`,
  `NvDsInferParseUltralytics`, built-in DetectNet_v2) and fixed preprocessing.
- Inference is **`nvinfer` per session**, `batch-size=1`, `gpu-id=0`. No cross-stream
  batching, no multi-model GPU sharing. `runtime_support` declares
  `nvinferserver`/`triton` tokens but **only `nvinfer` is implemented**.
- Artifacts reach the GPU host via **operator-run `scripts/prepare_*.sh`** into a
  host-global model store (`UPSTREAM_DEEPSTREAM_MODEL_STORE_DIR`); the api-server never
  ships bytes — the processor resolves a **package-id string** against its own local store.
- TensorRT engines build on first run, cached per model in the store. No validation of
  uploaded artifacts (no ONNX check, no shape/class verification, no smoke inference).

**Key insight:** the `prepare_*.sh` flow + the hand-written parsers *are* a manual BYOM
pipeline for curated architectures. This plan **automates, generalizes, isolates, and
secures** that — it is an evolution, not a rewrite.

---

## 2. Goals & non-goals

**Goals (v1):**
- Self-serve ingest of standard **object detectors** (weights or ONNX) per account.
- Automated prepare: normalize → validate → build (TensorRT) → smoke-test → publish.
- Efficient **multi-model serving** on shared GPUs (batching, load/unload).
- Per-account isolation, quotas, and licensing/provenance (extends the existing
  `license_class` / `first_party` / `catalog_first_party_only` gate).
- Reuse the proven DeepStream decode → track → `nvdsanalytics` pipeline + the P5-L
  browse/compare UI.
- An **honest validation gate**: a model is `ready` only after a real inference passes.

**Non-goals (v1 — explicitly deferred):**
- Arbitrary/custom-op models; non-exportable models; dynamic control-flow graphs.
- Non-detection tasks (classification / segmentation / pose) — Phase 4.
- **Customer-supplied code** (parsers, TensorRT plugins, `.so`) — never accepted.
- Cross-account model sharing / a public model marketplace.
- INT8 by default (needs calibration data — opt-in, Phase 4).
- Accuracy guarantees — we *run* the customer's model; we do not certify its accuracy.

---

## 3. Design principles (the 6 bets)

1. **ONNX is the universal IR.** Everything normalizes to ONNX at ingest; source
   framework stops mattering past the front door.
2. **Push post-processing into the graph (embedded NMS).** Prefer/auto-produce ONNX that
   outputs final detections (`EfficientNMS`/ONNX-NMS) → one universal output parser →
   eliminates per-model decode code for the common case.
3. **Declarative Model Descriptor + a small Decoder Registry** replace the hardcoded
   `config_template` handlers. A model is *data*; decoders are a *finite, parameterized*
   set keyed by head family.
4. **Serve via Triton; keep DeepStream for media/analytics.** Inference routes through a
   shared Triton Inference Server (via the `nvinferserver` gie); decode/track/analytics
   stay in DeepStream.
5. **Customers bring DATA, not CODE.** ONNX op-allowlist; conversion/build in sandboxed,
   no-network ephemeral containers; never accept compiled artifacts.
6. **`ready` requires a passing smoke inference.** Validation is a structural gate, not
   metadata — matches the project honesty contract.

---

## 4. Target architecture

Three planes, plus the existing pipeline:

```
CONTROL PLANE  (api-server / EC2)        owns intent + lifecycle state
  account_models DB (per-tenant, RLS) · Model Descriptor · quotas · licensing · catalog merge
        │ enqueue build job
PREPARE PLANE  (sandboxed build workers) the automated prepare_*.sh
  normalize→ONNX · validate(op-allowlist, signature) · embed-NMS · build TRT per GPU-arch
  · smoke-test(sample frame → sane boxes?) · publish → Model Repository
        │ manifest + onnx + engine + descriptor (content-addressed, per-account)
SERVING PLANE  (GPU host[s])
  Model Repository (object store)  ──►  Triton Inference Server
                                          dynamic batching · multi-model · load/unload(LRU)
                                          TRT/ORT backends · per-account isolation
                                            ▲ nvinferserver (gRPC)
  DeepStream worker: decode → track → nvdsanalytics  +  generic Decoder Registry
```

### 4.1 Model Descriptor (the keystone contract)

Declarative spec produced at ingest (auto-detected where possible, customer-confirmed,
validated by smoke test). Stored as JSONB on the model version.

| Field | Notes |
|---|---|
| `model_id` | account-namespaced, e.g. `acct_<id>.<slug>.v<n>` |
| `task` | `detect` (v1); later `classify`/`segment`/`pose` |
| `family` | `embedded_nms` \| `yolo_anchorfree` \| `yolo_anchor_grid` \| `detectnet_v2` \| `ssd_retinanet` \| `rtdetr` |
| `input` | `width,height,channels,layout(NCHW/NHWC),color(rgb/bgr),scale,mean[],std[],letterbox(bool),pad_value` |
| `output` | `nms: embedded \| decoder`; `decoder: <registry_key>`; `tensor_layout` (strides/anchors/transposed) |
| `class_map` | index → label list |
| `precision` | `fp16` (default) \| `int8` |
| `artifact` | `{ onnx_key, engine_keys:{gpu_arch:key}, checksum }` |
| `provenance` | `{ license_class, first_party, license_attested_by, source }` |

### 4.2 Decoder Registry (kills the per-model-parser problem)

A finite library of **generic, config-parameterized** decoders — NOT per-model C++:

- `embedded_nms` — output is `(num_dets, boxes, scores, classes)`; **the default, covers
  most popular detectors once exported with NMS.**
- `yolo_anchorfree` — YOLOv8/v9/v11/YOLOX-style transposed `[1, 4+C, A]`.
- `yolo_anchor_grid` — YOLOv5/v7 grid/anchor layout.
- `detectnet_v2` — TAO grid (built-in DeepStream parser).
- `ssd_retinanet` — multi-level SSD/RetinaNet/EfficientDet heads.
- `rtdetr` — DETR/RT-DETR set prediction (no NMS).

Adding a family = a descriptor template + (maybe) one decoder. The existing
`NvDsInferParseYolox` / `NvDsInferParseUltralytics` become the first two registry entries.

### 4.3 Data model (api-server, Postgres + RLS)

- `account_models` — `id, account_id(FK,RLS), slug, display_name, task, family,
  current_version_id, license_class, first_party, license_attested_at, created_by,
  created_at, archived_at`. `UNIQUE(account_id, slug)`.
- `account_model_versions` — `id, model_id(FK), version, descriptor(jsonb),
  onnx_storage_key, size_bytes, checksum, status, status_reason, smoke_test(jsonb),
  created_at`. Status enum: `uploaded→validating→converting→building→smoke_testing→ready|failed`.
- `account_model_engines` — `id, version_id(FK), gpu_arch, precision, engine_storage_key,
  trt_version, built_at`. (The portable engine cache index.)
- `account_processing_packages` — binds a `ready` model version to a runnable package
  (model + tracker + analytics + allowed_overrides), so it flows through the **existing**
  P5-G/P5-L request/apply + the first-party gate unchanged.

All account-scoped tables get RLS keyed on `account_id` (same posture as `cameras`).

### 4.4 API surface (new)

- `POST /v1/models` — create model + first version; returns a resumable/multipart upload
  target (signed URL) for the weights.
- `POST /v1/models/{id}/versions` — new version.
- `GET /v1/models` · `GET /v1/models/{id}` — per-account list/detail + lifecycle status.
- `DELETE /v1/models/{id}` — soft-archive.
- Internal (processor-authed): `GET /v1/internal/processor/model-manifests?account_id=`
  (what changed, with checksums) and `GET .../model-artifact/{version_id}/{gpu_arch}`
  (stream engine/onnx). The build pipeline runs off an internal job queue, not a public endpoint.

### 4.5 Serving (Triton) specifics

- **Model repository** = content-addressed object store, per-account namespaced:
  `models/<account_id>/<slug>_v<n>/{config.pbtxt, 1/model.plan|model.onnx}` (+ an ensemble
  for preprocess→infer→NMS when needed). Triton pulls from it (cloud-repo or synced local).
- **Engine cache key** = `sha256(onnx) · precision · gpu_arch · max_batch · trt_version`.
  TensorRT engines are **not portable across GPU generations** → build per arch.
- **Efficiency**: Triton dynamic batching across streams/tenants; instance groups for
  concurrency; **model load/unload (LRU warm pool)** so *registered* ≫ *resident* (trades
  cold-start latency for GPU memory). DeepStream calls Triton via `nvinferserver`.
- **Isolation**: per-tier Triton instances or **MIG** partitions for premium/dedicated
  tenants; per-account repo paths; resource limits.

### 4.6 Security model

- Conversion + engine build run in a **locked-down ephemeral container** (no network
  egress, CPU/mem/wall-time caps, read-only FS except the output dir).
- **ONNX op-allowlist** — reject custom ops, plugin ops, and unsafe dynamic control flow.
- **Never** accept `.so` / TensorRT plugins / customer code. The decode is *our* generic
  code; the customer supplies *weights only*.
- Artifact size caps + checksum + (optional) malware scan on upload.

---

## 5. Model lifecycle (state machine)

```
upload weights ─► uploaded ─► validating ─► converting(→ONNX) ─► building(TRT, per arch)
  ─► smoke_testing(sample frame → boxes sane?) ─► ready ─► assignable to a camera
                                              └► failed (+ status_reason, surfaced in UI)
```

Every transition is visible in the dashboard, so failures arrive **with a reason,
pre-deployment**.

---

## 6. Implementation phases

Each phase is independently shippable and delivers value. Phases 0–2 require **no Triton**
and reach the first end-to-end self-serve BYOM; Triton lands in Phase 3.

### Phase 0 — Descriptor + Decoder Registry refactor (no infra, no behavior change) — ✅ IMPLEMENTED
**Why first:** de-risk everything by making the runtime data-driven *before* adding infra.
- **processor:** introduce the Model Descriptor type; refactor `build_model_runtime_plan`
  from 4 hardcoded templates → descriptor-driven config building + a Decoder Registry
  (wrap the existing YOLOX/Ultralytics/DetectNet parsers as the first entries). Retrofit
  the 10 current models to descriptors.
- **Exit criteria:** every existing model runs via descriptors; **golden tests prove the
  generated PGIE config + behavior is byte-identical to today** (esp. the live-verified
  `yolox_s` path). No new endpoints, no infra.
- **Unlocks:** "add a model" becomes data, not code.

> **Status — implemented 2026-06-20 (processor only, uncommitted).** See
> [`byom-descriptors.md`](./byom-descriptors.md) for developer notes.
>
> **What changed:**
> - New `app/packages/model_descriptor.py` — a declarative `ModelDescriptor`
>   (input/output/artifact/provenance sub-records) generated from the existing
>   registry via `descriptor_from_spec(spec)`. No new JSON, no DB.
> - New `app/runtimes/deepstream_runtime/decoders.py` — the finite
>   `DECODER_REGISTRY` (4 entries) + `get_decoder()`. The four `build_*_pgie_config`
>   byte-templates moved here **verbatim**; each decoder wraps one plus its
>   assembly facts (backend, engine location, artifact layout, parser lib,
>   labels source).
> - `build_model_runtime_plan` refactored: the four `if config_template == …`
>   branches collapse to one data-driven body (`spec → descriptor → decoder →
>   plan`). The builders are re-exported from `inference.py` so existing call
>   sites/tests are untouched.
>
> **What did NOT change:** runtime bytes, model outputs, model/package IDs,
> recommendations, the package catalog, api-server/dashboard. The PGIE config
> and every `ModelRuntimePlan` field are **byte-identical** for all 10 models.
>
> **Key design call:** the four families differ in nvinfer **key order and set**
> (not just values), so a single generic renderer could not be byte-identical.
> Phase 0 therefore keeps each family's exact byte-template inside its decoder;
> the descriptor *supplies the values it consumes* and *declares* the same
> preprocessing (a consistency test asserts they agree). Lifting the constants
> into a fully descriptor-emitted renderer is deferred to a later phase.
>
> **How it unlocks Phase 1:** the automated prepare pipeline now has concrete
> targets — it emits a `ModelDescriptor` (esp. the `artifact` block: onnx name,
> engine strategy, custom-parser lib, checksum) and the decoder's required-files
> contract is exactly what the validation gate must satisfy.
>
> **Tests:** new `tests/test_config_parity.py` (per-model byte-identical PGIE
> goldens + full plan-field parity + explicit `yolox_s` golden),
> `tests/test_model_descriptor.py`, `tests/test_decoder_registry.py`. Full
> suite **847 passed**, `mypy app` clean, `ruff check`/`ruff format` clean.

### Phase 1 — Automated build pipeline + validation gate ("prepare as a service")
**Why:** turn `prepare_*.sh` into a service; introduce the honesty gate.
- **processor (or a new build worker):** a Model Build Pipeline — normalize→ONNX,
  validate (`onnx.checker`, opset range, op-allowlist, input/output signature),
  embed-NMS/graph-simplify, build TensorRT engine, **smoke-test** (sample frame → confirm
  output parses to sane boxes), publish artifact + descriptor + manifest to a
  content-addressed store. Runs sandboxed. **Still serves via existing nvinfer.**
- **Exit criteria:** point the pipeline at a weights file → get a validated, runnable
  artifact + descriptor automatically (the automated version of the manual Ultralytics prep);
  invalid models fail with a clear reason; smoke test gates `ready`.
- **Unlocks:** repeatable, validated model prep with no hand-written C++.

### Phase 2 — Per-account registry + ingest + distribution (first end-to-end BYOM)
**Why:** customer-facing self-serve, on the existing serving path.
- **api-server:** `account_models` / `account_model_versions` tables + RLS + lifecycle +
  quotas + licensing attestation; ingest endpoints (§4.4); enqueue Phase-1 builds; a
  **distribution endpoint** the GPU host pulls from (manifest + checksum, atomic publish),
  per-account namespaced. Merge per-account models into `GET /v1/processing/packages`
  (gated, badged BYOM — reuse P5-K/P5-L spine).
- **processor:** poll/pull validated artifacts into a **per-account** store layout; resolve
  account-scoped model ids; (still nvinfer).
- **dashboard:** upload UI (resumable) + per-account model list + lifecycle status; surface
  in the P5-L model picker.
- **storage:** a dedicated `models` bucket (separate from `snapshots`), large/resumable
  upload, content-addressed.
- **Exit criteria:** a customer uploads a standard detector via the dashboard, watches it
  validate/build, selects it on a camera, and it runs (per-session nvinfer). **First
  self-serve BYOM.**
- **Unlocks:** the product capability, on proven serving.

### Phase 3 — Triton serving plane (efficiency + multi-model)
**Why:** remove the per-session/batch-1/single-GPU ceiling.
- **serving:** stand up Triton on the GPU host(s); model-repository integration + engine
  cache by GPU arch; dynamic batching; load/unload (LRU warm pool); per-account isolation
  (instance groups; MIG for premium).
- **processor:** switch the DeepStream pipeline's inference from `nvinfer` →
  `nvinferserver` (Triton), keeping decode→track→`nvdsanalytics` intact.
- **Exit criteria:** many models share GPUs with real batching/utilization; registered ≫
  resident via load/unload; A/B parity vs the Phase-2 nvinfer path on the existing models.
- **Unlocks:** efficient fleet-scale multi-model serving.

### Phase 4 — Breadth & hardening
**Why:** cover "the popular ones" across tasks + scale operationally.
- More decoder families (RT-DETR/DETR, SSD/RetinaNet/EfficientDet); classification /
  segmentation / pose tasks **+ their analytics integration**; INT8 calibration
  (customer-provided frames); fleet scheduler/autoscaling of GPU workers;
  per-model usage metering + billing; **"preview detections on a sample frame" UX**;
  best-effort arbitrary-ONNX with explicit rejection messaging.
- **Exit criteria:** supported-family matrix published; preview UX catches descriptor
  mis-detection; quotas/billing live.

---

## 7. Cross-cutting concerns

- **Testing:** Phase-0 golden parity tests; per-decoder unit tests on synthetic tensors;
  a smoke-test harness with sample frames per family; api-server contract tests for the
  ingest/lifecycle endpoints.
- **Observability:** build-pipeline metrics (convert/build/smoke pass rates + durations),
  GPU utilization + load/unload events, per-account model usage, distribution failures.
- **Licensing/provenance:** customer license attestation at upload; BYOM models are always
  the customer's (private, `first_party=false`); reuse the `catalog_first_party_only` gate.
- **Quotas/billing:** per-account `model_storage_mb` + GPU-minutes; dashboard quota viz
  (mirror the snapshot-quota pattern).
- **Rollout:** feature-flag the whole surface (mirror `NEXT_PUBLIC_FF_*` + the catalog gate);
  dogfood with first-party models before opening uploads.

---

## 8. Risks & mitigations

| Risk | Mitigation |
|---|---|
| TensorRT engines not portable across GPU archs | Build per arch; cache keyed by arch+TRT version; small build farm or build-on-placement |
| Cold-start latency from load/unload | LRU warm pool sized per tier; keep hot for active cameras; document first-run engine build cost |
| Descriptor mis-detection → silently wrong boxes | Smoke-test gate + **preview-on-sample-frame** UX before production assignment |
| GPU memory pressure with many models | Warm-pool sizing; per-tier limits; MIG/dedicated for premium |
| Security of customer artifacts | Data-not-code; op-allowlist; sandboxed no-network build; no custom plugins |
| Triton operational complexity | Introduce only in Phase 3, after the data-driven runtime + ingest are proven |
| Accuracy expectations on BYOM | Explicit "we run it, we don't certify accuracy"; preview + smoke test set expectations |

---

## 9. Open decisions (resolve before/at each phase)

- Build workers **on the GPU host** vs **dedicated build fleet** (engine builds need the
  target GPU arch).
- Triton as a **sidecar** on each GPU host vs a **central** inference tier.
- Op-allowlist strictness + which opset range to support.
- Non-square / dynamic input shapes in v1 (recommend: fixed square first).
- Shared-GPU multi-tenancy default vs dedicated/MIG per tier.

---

## 10. TL;DR

Evolve the curated, template-hardcoded, nvinfer-per-session runtime into a
**descriptor-driven, ONNX-normalized, embedded-NMS-first, Triton-served** platform with an
**automated, sandboxed, validation-gated prepare pipeline** and a **per-account model
registry**. Ship it in 5 phases (0–4): make the runtime data-driven → automate prepare →
land per-account ingest end-to-end (still nvinfer) → add Triton for efficiency → broaden
families/tasks + harden. The achievable target is **"bring any standard detector as
ONNX/weights and we ingest→validate→build→serve it efficiently on shared GPUs,"** degrading
safely (reject at validation) for the long tail.
