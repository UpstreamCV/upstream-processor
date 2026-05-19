# upstream-processor

The UpStream processor service: a supervisor that runs one **runtime worker**
per eligible stream session. Runtimes are pluggable: **M8.1** added the
`mock` runtime (proves the handoff contract, no video); **M8.2B** added the
`deepstream` decode-probe runtime (GPU NVDEC decode — decode only);
**M8.2C1-B** adds the `deepstream` **inference-probe** package (DeepStream 8.0
`nvinfer` on the live RTSP stream — inference runs, **status/log-only
detection evidence, no observations, no metadata extraction**).

```
upstream-api (M8.0 /v1/internal/processor/*)
        ▲  GET eligible-sessions        │ POST status / observations
        │                               ▼
┌───────┴───────────────────────────────────────┐
│  upstream-processor                            │
│   main → SessionPoller → WorkerManager         │
│            └── MockWorker (one per session)    │
└────────────────────────────────────────────────┘
```

## M8.1 scope

- Poll `GET /v1/internal/processor/eligible-sessions`.
- Start one mock worker per eligible session; stop it when the session drops
  out of eligibility; never duplicate; quarantine crash-loopers.
- Post the status lifecycle: `starting → attached → processing… → stopped`
  (or `failed` on error).
- Emit the three mock observation types:
  `processor.mock_frame_seen`, `processor.mock_heartbeat`,
  `processor.mock_detection`.
- Survive transient API errors; treat a bad secret as a fatal config error;
  shut down cleanly and bounded on SIGINT/SIGTERM.

## M8.2B scope (DeepStream decode probe)

- New package `deepstream.decode_probe.v0` (runtime `deepstream`, `outputs: []`,
  `supports.decode_only: true`).
- `DeepStreamWorker` runs the M8.2A-proven pipeline as a managed Docker
  subprocess: `rtspsrc → rtph264depay → h264parse → nvv4l2decoder →
  nvvideoconvert → fpsdisplaysink` (`--gpus all --network host`).
- Maps decode onto the status lifecycle: `starting → attached` (first frames)
  `→ processing` (with `fps_processed`) `→ stopped`, or `failed` on process
  error. **No observations.**
- `UPSTREAM_FORCE_PACKAGE_ID` (dev/test only) routes a session to the
  DeepStream package without mutating production camera config.
- The RTSP URL is passed only to the subprocess; logs use a redacted form
  (`rtsp://host:port/sess_<redacted>`); the secret is never logged.

Run it (on the GPU/streaming host, with an active validated session):

```sh
UPSTREAM_FORCE_PACKAGE_ID=deepstream.decode_probe.v0 \
UPSTREAM_PROCESSOR_ID=proc_m82b_live \
uv run upstream-processor
```

## M8.2C1-B scope (DeepStream inference probe)

- New package `deepstream.inference_probe.v0` (runtime `deepstream`,
  `outputs: []`, `supports.inference_probe: true`). `mock.detect.v0` and
  `deepstream.decode_probe.v0` are unchanged.
- The resolved package's `supports` selects the `DeepStreamWorker` mode:
  `decode_probe` (M8.2B, unchanged) or `inference_probe` (this milestone).
- Inference mode runs the **M8.2C1-A-proven** DeepStream 8.0 `deepstream-app`
  pipeline as a managed Docker subprocess: RTSP/TCP → `nvstreammux` →
  `nvinfer` (bundled **TrafficCamNet** sample model, INT8) → `fakesink`, no
  display. The TensorRT engine is built fresh from the sample ONNX (the
  Blackwell sm_120 test). Bounded by `UPSTREAM_DEEPSTREAM_INFERENCE_TIMEOUT_SECONDS`.
- Status lifecycle: `starting → attached` (source/nvinfer up) `→ processing`
  (engine built / `**PERF` fps) `→ stopped`; `→ failed` on
  `Unsupported SM` / TensorRT / CUDA error (redacted `error_message`,
  `error_code=deepstream_unsupported_sm`). Status payload carries
  `inference_enabled`, `inference_engine_built`, `model`, `image`,
  `fps_processed`, and KITTI `detection_frames_count` / `detection_count`.
- **Detection evidence is status/log-only** (deepstream-app's own KITTI dump,
  counted by the process manager) — **NOT** metadata extraction. **No
  observations, no pyds, no pad probes, no MetadataSource, no metadata
  adapter, no tracking, no nvdsanalytics.**
- Default image is **DeepStream 8.0** (7.1's TensorRT 10.3 fails inference on
  Blackwell — see `docs/deepstream-inference-smoke.md`). The per-run
  deepstream-app/PGIE config + KITTI dir live only in a temp dir under the
  system tempdir (raw RTSP URL only there + in argv; never logged/committed)
  and are removed on stop unless `UPSTREAM_DEEPSTREAM_KEEP_TEMP_FILES=true`.

Run it (on the GPU/streaming host, with an active validated session):

```sh
UPSTREAM_FORCE_PACKAGE_ID=deepstream.inference_probe.v0 \
UPSTREAM_DEEPSTREAM_IMAGE=nvcr.io/nvidia/deepstream:8.0-samples-multiarch \
UPSTREAM_PROCESSOR_ID=proc_m82c1b_live \
uv run upstream-processor
```

## What it deliberately does NOT do (yet)

The DeepStream inference probe (M8.2C1-B) runs `nvinfer` but is still a
**probe**: detection evidence is status/log-only (KITTI dump) and is **not**
metadata extraction. Explicitly out of scope: **pyds, pad probes,
MetadataSource, metadata adapter, observations** (`POST /observations`),
tracking, `nvtracker`, `nvdsanalytics`, line crossing, zone analytics,
Triton/nvinferserver, dashboard UI, package catalog, billing, YOLO/custom
models, multi-stream batching. API contracts are unchanged. The `mock`
runtime and the M8.2B decode probe are unchanged; M8.1's
`internal_rtsp_url`-is-never-read rule still holds for the mock runtime.

## Run locally

```sh
make install                 # uv sync
cp .env.example .env         # set UPSTREAM_PROCESSOR_WEBHOOK_SECRET
make run                     # python -m app.main
# or: uv run upstream-processor
```

## Environment variables

| Variable | Required | Default | Purpose |
|---|---|---|---|
| `UPSTREAM_PROCESSOR_WEBHOOK_SECRET` | yes | — | Shared secret for `/v1/internal/processor/*` (matches api-server) |
| `UPSTREAM_API_BASE_URL` | no | `https://api.upstreamcv.com` | API base URL |
| `UPSTREAM_PROCESSOR_ID` | no | `proc_local_mock_001` | This processor's id |
| `UPSTREAM_POLL_INTERVAL_SECONDS` | no | `5` | Eligible-sessions poll cadence |
| `UPSTREAM_MOCK_OBSERVATION_INTERVAL_SECONDS` | no | `5` | Mock frame-seen cadence |
| `UPSTREAM_LOG_LEVEL` | no | `INFO` | Log level |
| `UPSTREAM_ELIGIBLE_SESSION_LIMIT` | no | `50` | `?limit=` on the poll |
| `UPSTREAM_FORCE_PACKAGE_ID` | no | — | **Dev/test only**: force every session to this package id (`deepstream.decode_probe.v0` or `deepstream.inference_probe.v0`) |
| `UPSTREAM_DEEPSTREAM_IMAGE` | no | `nvcr.io/nvidia/deepstream:8.0-samples-multiarch` | DeepStream Docker image. 8.0 = inference-capable on Blackwell; 7.1 decodes only |
| `UPSTREAM_DEEPSTREAM_RTSP_TRANSPORT` | no | `tcp` | RTSP transport (decode `rtspsrc protocols=`; inference `select-rtp-protocol`) |
| `UPSTREAM_DEEPSTREAM_PROBE_TIMEOUT_SECONDS` | no | `30` | Bounded **decode**-probe duration cap |
| `UPSTREAM_DEEPSTREAM_PROCESS_STOP_TIMEOUT_SECONDS` | no | `5` | Grace before SIGKILL of the DeepStream subprocess |
| `UPSTREAM_DEEPSTREAM_INFERENCE_TIMEOUT_SECONDS` | no | `60` | Bounded **inference**-probe duration cap (covers TensorRT engine build + run) |
| `UPSTREAM_DEEPSTREAM_INFERENCE_MODEL` | no | `trafficcamnet` | Bundled sample model for the inference probe (`trafficcamnet`/`sample_trafficcamnet`) |
| `UPSTREAM_DEEPSTREAM_INFERENCE_OUTPUT_DIR` | no | — (auto temp) | Base dir for the per-run config + KITTI output; empty = system tempdir |
| `UPSTREAM_DEEPSTREAM_KEEP_TEMP_FILES` | no | `false` | Keep the per-run temp dir after stop (still never logs the raw RTSP URL) |
| `UPSTREAM_DEEPSTREAM_DOCKER_NETWORK` | no | `host` | Docker network (host → container `127.0.0.1` = streaming host) |
| `UPSTREAM_FINAL_STATUS_TIMEOUT_SECONDS` | no | `15` | Per-worker budget for graceful stop + final `stopped` post (DeepStream-safe; mock exits faster) |
| `UPSTREAM_OVERALL_SHUTDOWN_TIMEOUT_SECONDS` | no | `25` | Overall supervisor shutdown cap before workers are force-cancelled |

The secret is a `SecretStr` and is never logged (a redaction filter on the log
handler scrubs it, the header, and JWT-shaped strings as defence-in-depth).

## Quality

```sh
make check     # ruff check + ruff format --check + mypy app + pytest
```
