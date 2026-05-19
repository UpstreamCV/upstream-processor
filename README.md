# upstream-processor

The UpStream processor service: a supervisor that runs one **runtime worker**
per eligible stream session. Runtimes are pluggable: **M8.1** added the
`mock` runtime (proves the handoff contract, no video); **M8.2B** adds the
`deepstream` decode-probe runtime (GPU NVDEC decode of the session's RTSP
stream — **decode only, no inference, no observations**).

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
UPSTREAM_DEEPSTREAM_IMAGE=nvcr.io/nvidia/deepstream:7.1-samples-multiarch \
UPSTREAM_PROCESSOR_ID=proc_m82b_live \
uv run upstream-processor
```

## What it deliberately does NOT do (yet)

DeepStream is **decode only** in M8.2B — no inference, no observations, no
tracking, no nvdsanalytics, no metadata adapter. Also out of scope: YOLO/CV,
Triton/nvinferserver, model registry, package marketplace, dashboard
zones/lines UI, rules engine, billing, webhooks, multi-stream batching. The
mock runtime is unchanged. M8.1's `internal_rtsp_url`-is-never-read rule still
holds for the mock runtime.

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
| `UPSTREAM_FORCE_PACKAGE_ID` | no | — | **Dev/test only**: force every session to this package id (e.g. `deepstream.decode_probe.v0`) |
| `UPSTREAM_DEEPSTREAM_IMAGE` | no | `nvcr.io/nvidia/deepstream:7.1-samples-multiarch` | DeepStream Docker image |
| `UPSTREAM_DEEPSTREAM_RTSP_TRANSPORT` | no | `tcp` | `rtspsrc protocols=` |
| `UPSTREAM_DEEPSTREAM_PROBE_TIMEOUT_SECONDS` | no | `30` | Bounded decode-probe duration cap |
| `UPSTREAM_DEEPSTREAM_PROCESS_STOP_TIMEOUT_SECONDS` | no | `5` | Grace before SIGKILL of the decode subprocess |
| `UPSTREAM_DEEPSTREAM_DOCKER_NETWORK` | no | `host` | Docker network (host → container `127.0.0.1` = streaming host) |
| `UPSTREAM_FINAL_STATUS_TIMEOUT_SECONDS` | no | `15` | Per-worker budget for graceful stop + final `stopped` post (DeepStream-safe; mock exits faster) |
| `UPSTREAM_OVERALL_SHUTDOWN_TIMEOUT_SECONDS` | no | `25` | Overall supervisor shutdown cap before workers are force-cancelled |

The secret is a `SecretStr` and is never logged (a redaction filter on the log
handler scrubs it, the header, and JWT-shaped strings as defence-in-depth).

## Quality

```sh
make check     # ruff check + ruff format --check + mypy app + pytest
```
