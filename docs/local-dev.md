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

## Live verification checklist (M8.2C1-B — DeepStream inference probe)

Run on the GPU/streaming host (crowdcountS2). The DeepStream **8.0** image
(`nvcr.io/nvidia/deepstream:8.0-samples-multiarch`, ~33 GB) is cached there
from M8.2C1-A. Status: implementation complete; **live verification pending
until explicitly run** (do not claim live-verified otherwise).

```
[ ] Start a valid upstream-agent stream
[ ] Wait for validation_status=true
[ ] eligible-sessions returns one session
[ ] Run upstream-processor with:
      UPSTREAM_FORCE_PACKAGE_ID=deepstream.inference_probe.v0
      UPSTREAM_DEEPSTREAM_IMAGE=nvcr.io/nvidia/deepstream:8.0-samples-multiarch
      UPSTREAM_PROCESSOR_ID=proc_m82c1b_live
[ ] DeepStream inference worker starts (log: "package override active",
    "deepstream worker ... mode=inference_probe")
[ ] Status posts `starting`
[ ] DeepStream container opens internal_rtsp_url (only redacted URL in logs)
[ ] nvinfer initializes; TensorRT engine builds/loads (no "Unsupported SM")
[ ] Status posts `attached`
[ ] Inference runs; `**PERF` fps lines; KITTI frames/detections appear
[ ] Status posts `processing` (payload: inference_enabled=true,
    inference_engine_built=true, model, image, detection_*counts)
[ ] Stop processor (SIGINT) — or the bounded probe self-exits at the timeout
[ ] DeepStream subprocess stops cleanly (docker stop; no lingering container)
[ ] Status posts `stopped`
[ ] processor_statuses row: runtime=deepstream,
    package_id=deepstream.inference_probe.v0
[ ] No observations are posted
[ ] No raw RTSP URL, stream key, secret, Authorization or Bearer in logs
[ ] Temp config/output dir cleaned up (unless KEEP_TEMP_FILES=true)
```

Note: the inference probe is bounded by
`UPSTREAM_DEEPSTREAM_INFERENCE_TIMEOUT_SECONDS` (default 60). It exits `124`
(timeout, clean) and the supervisor will restart it next poll while the
session stays eligible — expected for a probe (same shape as decode probe).

### M8.2B cleanup — shutdown defaults

Live verification needed bigger shutdown timeouts than the mock defaults, so
they are now first-class defaults (no env override needed for a clean
DeepStream stop): `UPSTREAM_FINAL_STATUS_TIMEOUT_SECONDS=15`,
`UPSTREAM_OVERALL_SHUTDOWN_TIMEOUT_SECONDS=25`,
`UPSTREAM_DEEPSTREAM_PROCESS_STOP_TIMEOUT_SECONDS=5`. Still env-overridable;
mock runtime exits well inside these (upper bounds, not delays). No
inference/observation/API change — readiness cleanup before the M8.2C1
inference smoke test.

## P2 — manual pyds metadata probe (out-of-tree)

A separate, manual-only probe that proves DeepStream metadata access via
NVIDIA pyds. Does **not** touch `app/` or any runtime worker; does **not**
call any UpStream API; does **not** emit observations.

```sh
# on the GPU host, with an active validated stream's internal_rtsp_url:
# (first run builds upstream-deepstream:8.0-pyds-v1.2.2 from
# docker/Dockerfile.pyds; ~15s build on top of the cached DS 8.0 image)
CONFIRM_BUILD=1 PROBE_SECONDS=45 \
  scripts/probe_deepstream_pyds_metadata.sh "rtsp://127.0.0.1:18554/sess_XXXX"
```

What it does: builds `rtspsrc → nvstreammux → nvinfer (TrafficCamNet) →
fakesink` inside a pyds-enabled DS 8.0 container, attaches a pad probe to
`nvinfer.src`, walks
`pyds.gst_buffer_get_nvds_batch_meta → NvDsFrameMeta → NvDsObjectMeta`,
and emits one `PYDS_PROBE_SUMMARY=<json>` line with counters
(`frames_seen`, `frames_with_objects`, `objects_seen`, `class_counts`,
`label_counts`, `has_bbox/has_confidence/has_tracker_id`) and a bounded
neutral-metadata sample
(`runtime`, `metadata_source`, `frame.{frame_num,width,height}`,
`objects[*].{class_id,label,confidence,bbox_pixels,bbox_normalized,tracker_id}`).
The pad probe is near-zero-work (no HTTP, no async, no API writes). The
raw RTSP URL goes only into the temp `pgie.txt` (under `/tmp`, shredded on
exit unless `KEEP_TEMP_FILES=1`) and the subprocess argv; logs use the
redacted form.

**This is NOT the MetadataSource and NOT the metadata adapter.** Object
counts here are debug evidence only. No observations, no pyds in `app/`,
no tracking, no nvtracker, no nvdsanalytics, no Triton.

## P2 — `MetadataSource` Protocol + bounded bridge queue (no live run)

Pure code, no GPU/Docker, no live stream needed. The neutral shape the
manual probe emits is now typed pydantic in
`app/observations/metadata_models.py`
(``NeutralFrameMetadata`` / ``NeutralFrame`` / ``NeutralObjectMetadata``
/ ``BoundingBoxPixels`` / ``BoundingBoxNormalized``), with a thread-safe
``BoundedBridgeQueue`` (``app/observations/bridge_queue.py``) and a
runtime-checkable ``MetadataSource`` Protocol + ``BridgeQueueMetadataSource``
(``app/runtimes/metadata_source.py``). Tests:
``tests/test_metadata_models.py`` / ``tests/test_bridge_queue.py`` /
``tests/test_metadata_source.py``. Run as part of ``make check`` —
**no GPU host, no Docker, no live stream required.** See
``docs/architecture.md`` for the design rationale (queue choice, drop
policies, single-consumer scope).

## Live verification checklist (P2 — supervisor-managed metadata probe)

Run on the GPU/streaming host (crowdcountS2). The pyds-derived image
`upstream-deepstream:8.0-pyds-v1.2.2` must already be built (see the
manual P2 probe milestone — `scripts/probe_deepstream_pyds_metadata.sh
CONFIRM_BUILD=1 …` will build it). The first live run (commit `dcc61c3`)
proved metadata flows end-to-end under supervisor control; the
shutdown-hardening cleanup (P2 cleanup commit) makes SIGINT work
reliably and ensures temp dirs are cleaned.

### Launch (records the *real* python PID, not the nohup wrapper)

```sh
# rsync the repo + build venv as in the prior P2 / M8.2C1-B live tests.
# Place the webhook secret yourself (the auto-mode classifier blocks
# automating prod-secret transfer):
umask 077
mkdir -p /tmp/upstream-live-p2
printf 'UPSTREAM_PROCESSOR_WEBHOOK_SECRET=%s\n' '<value>' \
  > /tmp/upstream-live-p2/secret.env
chmod 600 /tmp/upstream-live-p2/secret.env

cd /tmp/upstream-live-p2/repo
set -a; . /tmp/upstream-live-p2/secret.env; set +a
UPSTREAM_API_BASE_URL=https://api.upstreamcv.com \
UPSTREAM_PROCESSOR_ID=proc_p2_metadata_live \
UPSTREAM_FORCE_PACKAGE_ID=deepstream.metadata_probe.v0 \
UPSTREAM_DEEPSTREAM_PYDS_IMAGE=upstream-deepstream:8.0-pyds-v1.2.2 \
UPSTREAM_DEEPSTREAM_METADATA_PROBE_TIMEOUT_SECONDS=60 \
UPSTREAM_LOG_LEVEL=INFO \
nohup /tmp/upstream-live-p2/venv/bin/python -m app.main \
  > /tmp/upstream-live-p2/run.log 2>&1 &
sleep 1
# IMPORTANT: capture the REAL python PID via pgrep, NOT `$!`. `$!` is the
# subshell/nohup wrapper PID, which exits promptly after exec(); the long-
# lived python lives at a slightly higher PID. The P2 live verification
# initially missed this and ended up SIGTERMing a dead wrapper.
pgrep -f "/tmp/upstream-live-p2/venv/bin/python -m app\.main" -n \
  > /tmp/upstream-live-p2/processor.pid
cat /tmp/upstream-live-p2/processor.pid
```

### Stop (SIGINT first; SIGTERM only as a backstop)

```sh
PID="$(cat /tmp/upstream-live-p2/processor.pid)"
kill -INT "$PID"
# With the P2 cleanup shutdown hardening, SIGINT should bring the
# processor down within the supervisor's overall_shutdown budget
# (default 25 s). The metadata worker now closes its bridge queue on
# request_stop and shields _safe_stop_pm so docker stop + workdir
# cleanup run even under cancellation.
for i in $(seq 1 30); do
  ps -p "$PID" >/dev/null 2>&1 || break
  sleep 1
done
ps -p "$PID" >/dev/null 2>&1 && kill -TERM "$PID"
```

### Lifecycle checklist

```
[ ] Start a valid upstream-agent stream
[ ] Wait for validation_status=true
[ ] eligible-sessions returns one session
[ ] Worker starts (log: "package override active",
    "deepstream worker ... mode=metadata_probe",
    "deepstream metadata start: image=... pyds-v1.2.2 model=trafficcamnet")
[ ] pyds container starts
[ ] RTSP opens; nvinfer engine builds (~30 s cold)
[ ] UPSTREAM_METADATA= lines flow on stdout
[ ] worker validates each as NeutralFrameMetadata
[ ] BoundedBridgeQueue accepts metadata; consumer drains via
    BridgeQueueMetadataSource
[ ] Status posts `starting`
[ ] Status posts `attached` (first valid metadata frame)
[ ] Status posts `processing` (payload includes
    metadata_enabled=true, frames_seen, frames_with_objects,
    objects_seen, class_counts, label_counts, has_bbox, has_confidence,
    queue_stats)
[ ] SIGINT to the *real* PID (see above)
[ ] Processor exits within ~25 s (no SIGTERM required after cleanup hardening)
[ ] pyds container stops cleanly (docker stop; no lingering `upstream-dsm-*`)
[ ] Status posts `stopped`
[ ] processor_statuses row: runtime=deepstream,
    package_id=deepstream.metadata_probe.v0
[ ] No observations are posted (`processor.*` types absent for this run)
[ ] No raw RTSP URL, stream key, secret, Authorization or Bearer in logs
[ ] Temp work dir cleaned (no /tmp/upstream-p2-meta.* leftovers unless
    KEEP_TEMP_FILES=true); no __pycache__ leftovers
    (PYTHONDONTWRITEBYTECODE=1 set on the container)
```

### Cleanup (operator)

```sh
# Verify nothing lingers.
docker ps --format "table {{.Names}}\t{{.Image}}\t{{.Status}}" \
  | grep -E "deepstream|metadata|upstream-dsm" || true
# Belt and suspenders — these should already be gone after a clean stop.
rm -rf /tmp/upstream-p2-meta.* 2>/dev/null || true
rm -rf /tmp/upstream-live-p2
```

The probe is bounded by `UPSTREAM_DEEPSTREAM_METADATA_PROBE_TIMEOUT_SECONDS`
(default 60). It exits `124` (timeout, clean) and the supervisor restarts
it next poll while the session stays eligible — expected for a probe,
same shape as decode/inference probes.

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
