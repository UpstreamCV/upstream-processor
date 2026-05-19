# upstream-processor

The UpStream processor service: a supervisor that runs one **runtime worker**
per eligible stream session. Runtimes are pluggable; **M8.1 ships only the
mock runtime**, which proves the processor handoff contract end-to-end without
touching video.

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

## What it deliberately does NOT do (yet)

**No video decode. No DeepStream. No RTSP.** `internal_rtsp_url` from the API
is carried as opaque metadata and never read in M8.1. Also out of scope:
YOLO/CV, Triton, GPU/NVIDIA Docker, nvdsanalytics, model registry, package
marketplace, dashboard zones/lines UI, rules engine, billing, webhooks. Those
arrive with M8.2+ behind the same supervisor via a new runtime.

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

The secret is a `SecretStr` and is never logged (a redaction filter on the log
handler scrubs it, the header, and JWT-shaped strings as defence-in-depth).

## Quality

```sh
make check     # ruff check + ruff format --check + mypy app + pytest
```
