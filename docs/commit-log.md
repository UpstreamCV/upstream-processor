# Commit log

Reverse chronological. Newest at top.

---

## 2026-05-19 - P2: MetadataSource Protocol + bounded bridge queue (app/ seam)

### Summary
Promotes the neutral metadata shape proven by the manual pyds probe into
typed `app/`-side abstractions: pydantic models, a thread-safe bounded
queue with explicit drop policy, and an async-iterable `MetadataSource`
Protocol. Pure code, no GPU/Docker/live-stream test required. No pyds, gi,
or Gst imports in `app/` or tests. **NO observations, NO API writes, NO
runtime-worker pyds integration yet** — those land in the next milestones.

### What changed
- `app/observations/metadata_models.py` (new) — pydantic v2 models
  (`extra="forbid"`): `BoundingBoxPixels`, `BoundingBoxNormalized`,
  `NeutralObjectMetadata` (class_id / label? / confidence? / bbox_pixels?
  / bbox_normalized? / tracker_id?: str), `NeutralFrame` (NonNegativeInt
  frame_num / width / height / optional timestamp), `NeutralFrameMetadata`
  (runtime / metadata_source / optional stream_session_id / frame /
  objects[] / source + debug dicts). Helpers: `bbox_normalized_from_pixels`
  (same rounding as the manual probe) and `tracker_id_from_pyds`
  (`UNTRACKED_OBJECT_ID -> None`, else stringify uint64). The models accept
  the EXACT dict the manual probe emitted live (asserted by test).
- `app/observations/bridge_queue.py` (new) — `BoundedBridgeQueue` +
  `DropPolicy` (DROP_NEWEST / DROP_OLDEST) + `BridgeQueueStats`. Producer
  side `put_from_thread` is thread-safe and non-blocking (returns bool);
  consumer side `get` is awaitable, returns `NeutralFrameMetadata | None`.
  Internal `queue.Queue` (unbounded slot; maxsize gated under
  `threading.Lock` so DROP_OLDEST can evict atomically) + `asyncio.to_thread`
  for the async path + a close sentinel to wake waiting consumers.
  Designed for a single async consumer (out of scope: fan-out).
- `app/runtimes/metadata_source.py` (new) — `MetadataSource` Protocol
  (`@runtime_checkable`, `__aiter__ -> AsyncIterator[NeutralFrameMetadata]`)
  + `BridgeQueueMetadataSource` (canonical impl wrapping the queue).
- `tests/test_metadata_models.py` (new, 12 tests), `tests/test_bridge_queue.py`
  (new, 13 tests), `tests/test_metadata_source.py` (new, 6 tests): pydantic
  shape/extra/blank-runtime validation, JSON round-trip, both drop
  policies, close-wakes-on-empty, async-wakes-on-late-put, real
  threading.Thread producer + slow asyncio consumer with bounded memory
  and FIFO order, Protocol conformance.
- Docs: architecture.md (P2 `app/` seam section), local-dev.md (developer
  pointer), this entry.

### Not changed
NO `app/` runtime worker pyds integration. NO pad probe in `app/`. NO
ObservationWriter integration. NO `POST /observations`. NO nvtracker, no
nvdsanalytics, no line crossing, no zone analytics, no dashboard, no
rules, no Triton, no API contract change. The DeepStream worker
(decode_probe + inference_probe) is unchanged. The manual P2 pyds probe in
`scripts/` is unchanged (it stays runtime-independent of `app/`).

### Testing/checks
`uv run pytest -q` **111 passed** (79 prior + 32 new). `ruff check`,
`ruff format --check`, `mypy app`, `bash -n scripts/*.sh` — all clean. No
Docker/GPU/pyds/gi imports anywhere in tests.

### Follow-up TODOs
- **P2 — supervisor-managed metadata probe worker**: use the proven manual
  pyds probe path PLUS this MetadataSource/BridgeQueue seam under the real
  processor supervisor lifecycle. Still no observations, no API writes.
- Later: P3 (package / config generator) and P4 (observation adapter).

---

## 2026-05-19 - P2: manual DeepStream pyds metadata probe (live-verified)

### Summary
Out-of-tree, manual-only probe proving the DeepStream metadata seam: pyds
imports under `--gpus all`, a pad probe on `nvinfer.src` walks
`pyds.gst_buffer_get_nvds_batch_meta` → `NvDsFrameMeta` → `NvDsObjectMeta`
and pulls `class_id` / `obj_label` / `confidence` / `rect_params` /
`object_id`. Live against the active stream the probe produced
**671 frames / 673 `person` detections** with bbox + confidence + (correct)
`tracker_id=null`; neutral metadata sample matches the P2 spec exactly;
clean bounded exit (`docker_exit=124`), no leftover container, no raw RTSP
key or secret in any logged/committed file.

DeepStream 8.0 ships no pyds; the probe stands on a thin
`upstream-deepstream:8.0-pyds-v1.2.2` image built from
`docker/Dockerfile.pyds` (FROM `deepstream:8.0-samples-multiarch` +
`libpython3.12t64` + NVIDIA `pyds-1.2.2-cp312-cp312-linux_x86_64.whl`).
Build is reproducible; the in-build `import pyds` check was removed because
`libcuda.so.1` is only injected at run time by the NVIDIA Container Toolkit
(`--gpus all`) — the probe verifies the import naturally on first run.

### What changed
- `docker/Dockerfile.pyds` (new) — pinned base + wheel URL, SHELL `bash -lc`,
  apt: `python3-pip`, `libpython3.12t64`, `curl`, `ca-certificates`; curl
  purged after install; LABEL with pyds version + base.
- `scripts/_pyds_probe_helpers.py` (new) — pure helpers: `redact_rtsp`,
  `scrub_rtsp_inline`, `bbox_normalized`, `build_neutral_object_sample`,
  `UNTRACKED_OBJECT_ID`. No gi/pyds imports → testable on the Mac.
- `scripts/deepstream_pyds_metadata_probe.py` (new) — Python probe: builds
  `rtspsrc(tcp) → rtph264depay → h264parse → nvv4l2decoder → nvstreammux →
  nvinfer → nvvideoconvert → fakesink`; pad probe on `nvinfer.src`;
  near-zero work; bounded by `--seconds`; emits one
  `PYDS_PROBE_SUMMARY=<json>` line; SIGINT/SIGTERM-clean.
- `scripts/probe_deepstream_pyds_metadata.sh` (new) — driver: builds the
  pyds image on demand (`CONFIRM_BUILD=1`), generates a per-run temp
  `pgie.txt` (TrafficCamNet INT8 — same as M8.2C1-A/B), `docker run --rm
  --gpus all --network host -v wd:wd ...`, parses summary, exit-classifies,
  shreds the temp dir. RTSP redacted in every log line; `KEEP_TEMP_FILES=1`
  to retain.
- `tests/test_pyds_probe_helpers.py` (new, 9 tests) — redact / scrub /
  normalize / neutral-sample shape / JSON roundtrip / `UNTRACKED_OBJECT_ID`.
- Docs updated: `docs/architecture.md` (P2 section), `docs/deepstream-
  inference-smoke.md` (P2 pointer), `docs/deepstream-rtsp-probe.md`
  (forward pointer), README + local-dev (run-it pointer).

### Not changed
**No `app/` code touched.** No MetadataSource, no metadata adapter, no
ObservationWriter integration, no observations, no pyds in `app/`, no pad
probes in `app/`, no nvtracker, no nvdsanalytics, no Triton, no line
crossing, no zones, no dashboard, no rules. The runtime supervisor / worker
behavior is unchanged. API contracts unchanged. upstream-agent / api-server
/ streaming-server / dashboard untouched. The webhook secret never left the
API host (this probe doesn't call any UpStream API).

### Testing/checks
`uv run pytest -q` **79 passed** (70 prior + 9 new). `ruff check`,
`ruff format --check`, `mypy app`, `bash -n scripts/*.sh` — all clean.
Live: SSM eligible-sessions HTTP 200 / `validation.passed=true`;
`ffprobe` exit 0 (h264 1280×720 30/1); pyds 1.2.2 image built in ~15 s;
probe run produced `frames_seen=671 / frames_with_objects=671 /
objects_seen=673 / class_counts={"2":673} / label_counts={"person":673}`,
3 neutral-sample frames matching the P2 spec.

### Follow-up TODOs
- P2 (continued) — `MetadataSource` Protocol + bounded bridge queue in
  `app/`, consumed by an eventual ObservationWriter seam. Still no
  observations.
- Later: metadata adapter, tracking, nvtracker, nvdsanalytics, etc.

---

## 2026-05-19 - P1/M8.2C1-B: supervisor-managed DeepStream inference worker

### Summary
The M8.2C1-A-proven DeepStream 8.0 nvinfer pipeline now runs under the
existing WorkerManager as the `deepstream.inference_probe.v0` package. New
`DeepStreamInferenceProcessManager` owns a per-run temp work dir
(deepstream-app + PGIE config + KITTI dir, bind-mounted into the DS8
container) and runs the bounded `deepstream-app` RTSP→nvstreammux→nvinfer
(TrafficCamNet INT8)→fakesink pipeline. `DeepStreamWorker` gained a
mode (from the package's `supports`): `decode_probe` (M8.2B, unchanged) or
`inference_probe`. Status `starting→attached→processing→stopped`, or
`failed` with a redacted message + `deepstream_unsupported_sm`/
`deepstream_trt_error`/`deepstream_exit` code. Detection evidence
(KITTI frame/detection counts + engine-built/PERF log markers) is
status/log-only — **no observations, no pyds, no metadata extraction**.
Default DeepStream image bumped 7.1 → 8.0 (7.1 fails inference on Blackwell).
Implementation complete + unit-tested; **live verification pending**.

### Files touched
- `app/runtimes/deepstream_runtime/inference.py` (new) — pure builders
  (`build_pgie_config`, `build_app_config`, `build_inference_docker_command`)
  + `DeepStreamInferenceProcessManager` (temp dir, `kitti_stats()`, cleanup
  unless `keep_temp_files`, redacted logging).
- `app/runtimes/deepstream_runtime/worker.py` — `_mode`, mode-aware
  `_default_pm_factory`, `run()` dispatch, `_drive_decode()` (verbatim move,
  unchanged behaviour) + `_drive_inference()`, `_infer_payload`,
  `_kitti_stats`; `_post_best_effort` gained `payload`.
- `app/packages/package_registry.json` — `deepstream.inference_probe.v0`
  (`supports.inference_probe`); mock + decode entries unchanged.
- `app/config.py` — `deepstream_image` default → 8.0; new
  `deepstream_inference_timeout_seconds` (60),
  `deepstream_inference_output_dir` (""), `deepstream_inference_model`
  ("trafficcamnet"), `deepstream_keep_temp_files` (False).
- `tests/test_deepstream_inference.py` (new, 14 tests) +
  `tests/test_config.py` / `tests/test_package_resolver.py` additions.
- README / .env.example / docs/architecture / docs/local-dev (M8.2C1-B live
  checklist) / docs/deepstream-inference-smoke / docs/deepstream-rtsp-probe /
  `app/runtimes/deepstream_runtime/README.md`.

### Not changed
No API contract change. No pyds, pad probes, MetadataSource, metadata
adapter, observations, tracking, nvtracker, nvdsanalytics, line crossing,
zones, Triton, dashboard, package catalog, billing. Mock runtime and M8.2B
decode behaviour unchanged (decode loop moved verbatim into `_drive_decode`).
upstream-agent / api-server / streaming-server / dashboard untouched.

### Testing/checks
`uv run pytest -q` **70 passed** (53 prior + 17 new); `ruff check`,
`ruff format --check`, `mypy app`, `bash -n scripts/*.sh` all clean. No
Docker/GPU in the suite (scripted `ProcessManager` double feeds the exact
M8.2C1-A log lines; real PM temp-dir cleanup tested with a monkeypatched
subprocess).

### Follow-up TODOs
- Live verification on crowdcountS2 with an active stream
  (`UPSTREAM_FORCE_PACKAGE_ID=deepstream.inference_probe.v0`,
  `UPSTREAM_DEEPSTREAM_IMAGE=...8.0-samples-multiarch`).
- Later milestones: pyds / NvDsBatchMeta / metadata adapter / observations.

---

## 2026-05-19 - P1/M8.2C1-A: DeepStream 8.0 inference smoke PASSED on Blackwell

### Summary
Retried the M8.2C1-A inference probe on `deepstream:8.0-samples-multiarch`
(DeepStream 8.0.0, **TensorRT 10.9.0.34**, **CUDA 12.8.1**). The Blackwell
sm_120 block is **gone**: nvinfer built a TensorRT engine on the RTX 5070 Ti
(`serialize cuda engine to file: ...successfully`, ~30 s cold), ran at ~30
fps on the live RTSP stream, and TrafficCamNet produced **1571 detections
across 1728 KITTI frames** (class `person`). Clean SIGINT/timeout exit
(`App run successful`), no `Unsupported SM`, no core dump. **M8.2C1-A is
passed; next is M8.2C1-B.** Decode worker (M8.2B) unaffected. Still manual
probe only — no worker change, pyds, metadata, or observations.

### What changed
- `scripts/probe_deepstream_inference.sh` — default image bumped 7.1 → **8.0**
  (7.1 proven non-functional for inference on Blackwell); DEEPSTREAM_IMAGE
  still overridable. **Classifier fixed**: the cold-start
  `deserialize ... failed, try rebuild` line is the normal no-prebuilt-engine
  path and is no longer treated as failure; engine-build success is detected
  via `serialize cuda engine to file: ...successfully` (nvinfer persists the
  engine next to the ONNX, not at our model-engine-file path, so file
  presence is not a reliable signal); added an `[INCONCLUSIVE-DET]` exit-0
  branch for engine-built-but-zero-detections (scene/model mismatch).
- `docs/deepstream-inference-smoke.md` — verdict now PASS on 8.0; 7.1 kept as
  root-cause history; combined 7.1/8.0 evidence table; outcome → M8.2C1-B.
- `docs/commit-log.md`, `docs/deepstream-rtsp-probe.md` — updated.

### Not changed
No inference runtime worker, pyds, NvDsBatchMeta, metadata adapter,
observations, Triton, tracking, dashboard. App behaviour untouched. Probe is
read-only w.r.t. UpStream; processor secret sourced server-side (SSM/API
host), never printed/committed; no raw stream key committed.

### Testing/checks
`bash -n scripts/*.sh` clean; `uv run pytest -q` 53/53; ruff/format/mypy
unaffected (no Python changed). Live on crowdcountS2: SSM eligible-sessions
HTTP 200 COUNT=1; `ffprobe` exit 0; DS 8.0 manual run + fixed script run both
built the engine and produced detections; no leftover containers; temp dirs
removed/shredded.

### Follow-up TODOs
- M8.2C1-B: supervisor-managed inference worker, DeepStream image pinned to
  8.0 (TRT ≥ 10.7 / CUDA ≥ 12.8). Metadata/observations remain later
  milestones.

---

## 2026-05-19 - P1/M8.2C1-A: DeepStream nvinfer inference smoke probe (blocked: Blackwell)

### Summary
Manual-only probe to answer "can a DeepStream container run nvinfer on the
live RTSP stream on the RTX 5070 Ti?" Answer on `deepstream:7.1-samples-multiarch`:
**no.** nvinfer initialized but TensorRT 10.3 cannot build an engine for
Blackwell — `IBuilder::buildSerializedNetwork: Error Code 1: Internal Error
(Unsupported SM: 0xc00)` (compute 12.0 = sm_120), then core dump. Decode
(M8.2A/B) works on the same image because NVDEC bypasses TensorRT. Active
validated stream confirmed (h264 1280x720@30, `validation.passed=true`);
`ffprobe` clean. No runtime/worker change, no observations.

### What changed
- `docs/deepstream-inference-smoke.md` — new. Verdict, evidence table,
  TensorRT-vs-Blackwell explanation, recommended next image (DeepStream 8.x /
  TensorRT ≥ 10.7 / CUDA ≥ 12.8), how-to.
- `scripts/probe_deepstream_inference.sh` — new. RTSP→nvstreammux→nvinfer
  (bundled TrafficCamNet INT8)→fakesink via `deepstream-app`, no display;
  forces a fresh engine build (the SM-arch test); KITTI dump = detection
  evidence (no pyds). Temp config in a shredded `mktemp` dir, RTSP redacted,
  `CONFIRM_PULL=1` gate, exits non-zero on the SM block / any inference fail.

### Not changed
No inference runtime worker, pyds, NvDsBatchMeta, metadata adapter,
observations, Triton, tracking. App behaviour untouched. Probe is read-only
w.r.t. UpStream; processor secret sourced server-side (SSM/API host), never
printed/committed; no raw stream key committed.

### Testing/checks
`bash -n scripts/*.sh` clean; `uv run pytest -q` 53/53; ruff/mypy unaffected
(no Python changed). Live: SSM eligible-sessions HTTP 200 COUNT=1; crowdcountS2
`ffprobe` exit 0; deepstream-app reproduced `Unsupported SM: 0xc00`.

### Follow-up TODOs
- Re-run `scripts/probe_deepstream_inference.sh` with
  `DEEPSTREAM_IMAGE=nvcr.io/nvidia/deepstream:8.x-samples-multiarch` against
  an active stream. Only if it passes → M8.2C1-B inference worker.

---

## 2026-05-19 - P1/M8.2B cleanup: DeepStream shutdown defaults

### Summary
M8.2B live verification needed env overrides for clean DeepStream shutdown
(the M8.1/mock defaults were too tight for a Docker-backed worker). Made the
deployment-safe values first-class so future inference tests don't rely on
ad hoc overrides.

### What changed
- `app/config.py`: `final_status_timeout_seconds` 3.0→**15.0**;
  `overall_shutdown_timeout_seconds` 8.0→**25.0**;
  `deepstream_process_stop_timeout_seconds` 10.0→**5.0**.
  These are global/shared; documented as DeepStream-safe and still fine for
  mock (upper bounds, not delays). Env overrides unchanged.
- `.env.example`: added the three vars with a "shutdown budget" comment.
- README / architecture / local-dev / deepstream-rtsp-probe: noted the new
  defaults.
- `tests/test_config.py`: assert the new defaults + env-override still works.

### Not changed
No inference, no observations, no API-contract change, no runtime
architecture change. Mock + DeepStream tests use `fast_settings` overrides
and are unaffected. Deployment/readiness cleanup before M8.2C1 inference
smoke test.

### Testing/checks
`uv run pytest -q`, `ruff check`, `ruff format --check`, `mypy app` — all green.

---

## 2026-05-19 - P1/M8.2B: Minimal DeepStream Runtime Worker

### Summary
Turns the M8.2A manual decode probe into a supervisor-managed runtime worker.
New `deepstream.decode_probe.v0` package (runtime `deepstream`, decode only,
no observations). `DeepStreamWorker` runs the proven Docker/GStreamer NVDEC
pipeline as a managed subprocess and maps it onto the M8.0 status lifecycle
(`starting → attached → processing → stopped`/`failed`). Resolver now
dispatches by the package's runtime; `UPSTREAM_FORCE_PACKAGE_ID` routes a
mock-defaulted session to the DeepStream package without touching prod
config. RTSP URL is never logged (redacted); secret never handled here.
**No inference, no Triton, no tracking, no nvdsanalytics, no observations.**

### Files touched
- `app/packages/package_registry.json` — added `deepstream.decode_probe.v0`.
- `app/packages/resolver.py` — `resolve(package_id)` (drop runtime arg);
  `SUPPORTED_RUNTIMES={mock,deepstream}`; `PackageSpec.supports`.
- `app/config.py` — `force_package_id` + 5 `deepstream_*` settings.
- `app/supervisor/processor_status.py` — optional `runtime`/`package_id`
  overrides (mock unchanged; DeepStream reports its resolved runtime).
- `app/redact.py` (new) — `redact_rtsp` / `scrub_rtsp_in_text`.
- `app/runtimes/deepstream_runtime/{__init__,process_manager,worker}.py`
  (new) — `ProcessManager` protocol, `DeepStreamProcessManager`,
  `DeepStreamWorker`; `README.md` updated.
- `app/supervisor/worker_manager.py` — `effective_package_id` +
  `build_worker` runtime dispatch; `_Handle.worker: Runtime`.
- `tests/` — rewrote `test_package_resolver`; fixed `test_mock_runtime` /
  `test_worker_manager`; new `test_deepstream_runtime` (fake PM, no Docker).
- `docs/` — README, architecture, local-dev (M8.2B checklist),
  deepstream-rtsp-probe, this log.

### System interaction notes
`SessionPoller → WorkerManager (effective_package_id → resolve →
build_worker) → DeepStreamWorker → DeepStreamProcessManager (docker run
--gpus all --network host …) → fpsdisplaysink lines → POST /status`.
No observation writes.

### Testing/checks
- `uv run pytest -q` — 51/51 pass (was 39; +12).
- `uv run ruff check .` / `ruff format --check .` — clean.
- `uv run mypy app` — clean (28 files).
- Live DeepStream worker run = operator checklist (`docs/local-dev.md`).

### Follow-up TODOs
- Live-verify the worker against an active stream on crowdcountS2.
- Later: first inference probe (re-evaluate DeepStream tag for RTX 5070 Ti
  / Blackwell sm_120 — decode is fine, TensorRT is the open question).

---

## 2026-05-18 - M8.1: Mock Processor Handoff (initial repo)

### Summary
First commit of `upstream-processor`. A pure-asyncio supervisor that polls the
M8.0 `/v1/internal/processor/eligible-sessions` endpoint, runs one mock worker
per eligible session, posts the `starting → attached → processing → stopped`
status lifecycle (or `failed`), and emits the three `processor.mock_*`
observation types. Proves the processor handoff contract before any DeepStream
/ video work. **No video decode, no RTSP, no DeepStream** — `internal_rtsp_url`
is parsed but never read (AST-enforced).

### Files touched
- `app/config.py` — `Settings` (env_prefix `UPSTREAM_`, `SecretStr`).
- `app/errors.py` — typed `ProcessorError`/`ApiError` hierarchy.
- `app/logging_setup.py` — secret-redaction logging filter.
- `app/backoff.py` / `app/clock.py` — interruptible backoff; tz-aware time.
- `app/api_client/{base,upstream_api}.py` — `ProcessorApi` Protocol + async
  typed client (auth header isolated, no body/secret logging).
- `app/observations/{schemas,writer}.py` — contract mirror + observation
  chokepoint.
- `app/packages/{package_registry.json,resolver.py}` — single `mock.detect.v0`
  package; strict resolver.
- `app/runtimes/base.py`, `app/runtimes/mock_runtime/{worker,fake_observations}.py`
  — runtime seam + mock worker.
- `app/supervisor/{session_poller,worker_manager,processor_status}.py` —
  reconcile loop + status builder.
- `app/main.py` — entrypoint + signal handling + exit-code contract.
- `tests/*` — 39 tests; `docs/*`; `pyproject.toml`, `Makefile`, `.env.example`.

### Functions/classes changed
- `WorkerManager.run/_reap/_start_new/_stop_removed/_shutdown_all` — the
  single-writer reconcile + bounded shutdown.
- `MockWorker.run` — the lifecycle state machine; 404→quiet, 422→failed,
  transient→absorbed.
- `UpstreamApiClient` — failure→typed-error taxonomy.

### System interaction notes
`main → SessionPoller.fetch_eligible (GET) → WorkerManager reconcile →
MockWorker (POST /status, POST /observations)`. Auth via
`X-Upstream-Webhook-Secret`; tenant always server-derived from the session.

### Testing/checks
- `uv run pytest -q` — 39/39 pass.
- `uv run ruff check .` / `ruff format --check .` — clean.
- `uv run mypy app` — clean (strict; app code, matching api-server's
  `mypy src` convention).
- Boot smoke: start → transient-survive loop on unreachable API → SIGINT →
  exit 0, secret absent from logs.
- Read-only prod smoke: live `GET /eligible-sessions` parses into our schema.

### Follow-up TODOs
- M8.2: `runtimes/deepstream_runtime/` behind the same supervisor — first
  consumer of `internal_rtsp_url`; introduces real `lagging`.
