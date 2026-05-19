# Local dev & live verification

## Setup

```sh
make install
cp .env.example .env
# set UPSTREAM_PROCESSOR_WEBHOOK_SECRET to match the api-server's value
# (point UPSTREAM_API_BASE_URL at prod or a local api-server)
```

## End-to-end run-through (with a real agent stream)

1. Start a valid agent stream: in `upstream-agent`, `upstream run` against a
   camera.
2. Wait until the api-server marks the session validated
   (`stream_validations.passed = true` for the latest row, state `streaming`).
3. Confirm eligibility:
   `GET /v1/internal/processor/eligible-sessions` (with the processor secret)
   returns one session.
4. Run the processor: `make run` (or `uv run upstream-processor`).
5. Confirm in api-server data / logs: a `processor_statuses` row progresses
   `starting → attached → processing`, and `observations` rows of the three
   `processor.mock_*` types appear.
6. Stop the agent.
7. The session leaves eligibility; the processor stops the worker and posts
   `stopped`; `active_count` returns to 0.

## Live verification checklist (M8.1)

```
[ ] Deployed API M8.0 auth returns 200 for the correct processor secret
[ ] Start a valid upstream-agent stream
[ ] Wait for validation passed = true
[ ] GET /eligible-sessions returns one eligible session
[ ] Start upstream-processor
[ ] Processor starts one mock worker for that session
[ ] Processor posts `starting`
[ ] Processor posts `attached`
[ ] Processor posts `processing`
[ ] Processor emits `processor.mock_frame_seen`
[ ] Processor emits `processor.mock_heartbeat`
[ ] Processor emits `processor.mock_detection`
[ ] Stop the agent
[ ] Eligible session disappears
[ ] Processor stops the worker
[ ] Processor posts `stopped`
[ ] No webhook secret appears in logs
```

## Live verification checklist (M8.2B — DeepStream decode probe)

Run on the GPU/streaming host (crowdcountS2) so `127.0.0.1:18554` is the
streaming host. The DeepStream image (~9 GB) is cached there from M8.2A.

```
[ ] Start a valid upstream-agent stream
[ ] Wait for validation_status=true
[ ] eligible-sessions returns one session
[ ] Run upstream-processor with:
      UPSTREAM_FORCE_PACKAGE_ID=deepstream.decode_probe.v0
      UPSTREAM_DEEPSTREAM_IMAGE=nvcr.io/nvidia/deepstream:7.1-samples-multiarch
      UPSTREAM_PROCESSOR_ID=proc_m82b_live
[ ] DeepStream worker starts (log: "package override active", "deepstream worker")
[ ] Status posts `starting`
[ ] DeepStream container opens internal_rtsp_url (only redacted URL in logs)
[ ] Frames flow (fpsdisplaysink rendered: N)
[ ] Status posts `attached`
[ ] Status posts `processing` with fps_processed (~30)
[ ] Stop processor (SIGINT)
[ ] DeepStream subprocess stops cleanly (docker stop; no lingering container)
[ ] Status posts `stopped`
[ ] processor_statuses row: runtime=deepstream, package_id=deepstream.decode_probe.v0
[ ] No observations are posted
[ ] No secret or full RTSP URL appears in logs
```

### What was verified during M8.2B build

- Full suite **51 tests** green; ruff + ruff-format + `mypy app` clean.
- DeepStream worker lifecycle (starting/attached/processing/stopped/failed),
  no-observations, RTSP redaction, command builder, force override + runtime
  dispatch, and hung-process SIGKILL are all unit-tested with a fake process
  manager (no Docker/GPU in the suite).
- Live worker run against an active stream is the operator checklist above.

### What was verified during M8.1 build

- Full unit/integration suite (39 tests) green; ruff + mypy clean.
- Local boot smoke: starts, survives an unreachable API (transient backoff),
  SIGINT → clean exit 0, secret absent from logs.
- Read-only prod smoke: `GET /eligible-sessions` against the live
  `api.upstreamcv.com` with the real secret (sourced server-side, never
  printed) returns the expected `{"sessions":[…]}` envelope and parses into
  our schema. No writes to prod; the agent-stream rows above remain a manual
  operator step.
