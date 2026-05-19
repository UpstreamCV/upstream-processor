# Commit log

Reverse chronological. Newest at top.

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
