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

## P2 scope (supervisor-managed metadata probe)

- New package `deepstream.metadata_probe.v0` (runtime `deepstream`,
  `outputs: []`, `supports.metadata_probe: true`). `mock.detect.v0`,
  `deepstream.decode_probe.v0`, `deepstream.inference_probe.v0` are unchanged.
- The package's `supports` selects `DeepStreamWorker` mode: `decode_probe`
  / `inference_probe` / **`metadata_probe`** (this milestone).
- Metadata mode spawns the **pyds-derived DS 8.0 image**
  (`upstream-deepstream:8.0-pyds-v1.2.2`, built from `docker/Dockerfile.pyds`
  in the manual P2 probe milestone) and runs
  `app/runtimes/deepstream_runtime/pyds_metadata_app.py` inside it
  (bind-mounted from the runtime work dir). The in-container app builds
  the proven RTSP → `nvstreammux` → `nvinfer` (TrafficCamNet) → `fakesink`
  pipeline, attaches a pad probe to `nvinfer.src`, walks
  `pyds.gst_buffer_get_nvds_batch_meta → NvDsFrameMeta → NvDsObjectMeta`,
  and emits one structured line **per frame** on stdout
  (`UPSTREAM_METADATA=<json>`) plus a terminal `UPSTREAM_METADATA_SUMMARY=<json>`.
- The outer supervisor's worker validates each line with
  `NeutralFrameMetadata`, pushes through the `BoundedBridgeQueue` and an
  async `BridgeQueueMetadataSource` consumer, and surfaces counters
  (`frames_seen`, `frames_with_objects`, `objects_seen`, `class_counts`,
  `label_counts`, `has_bbox`, `has_confidence`, `has_tracker_id`,
  `sample_count`) + the queue's own stats (`accepted`, `dropped`,
  `dropped_newest`, `dropped_oldest`, `consumed`, `current_size`) in the
  status payload. **The container never calls UpStream APIs** — the
  webhook secret never enters the container.
- Status lifecycle: `starting → attached` (first valid metadata frame)
  `→ processing` (throttled with counters) `→ stopped` (graceful);
  `→ failed` with redacted `error_message` +
  `deepstream_unsupported_sm` / `deepstream_trt_error` / `deepstream_exit`
  on hard TRT/SM/CUDA failure.
- **Status/debug evidence only — no observations, no metadata adapter
  yet.** KITTI files / per-frame samples are NOT stuffed into status.

Run it (on the GPU/streaming host, with an active validated session):

```sh
UPSTREAM_FORCE_PACKAGE_ID=deepstream.metadata_probe.v0 \
UPSTREAM_DEEPSTREAM_PYDS_IMAGE=upstream-deepstream:8.0-pyds-v1.2.2 \
UPSTREAM_PROCESSOR_ID=proc_p2_metadata_live \
uv run upstream-processor
```

## P3 scope (supervisor-managed tracker probe)

- New package `deepstream.tracker_probe.v0` (runtime `deepstream`,
  `outputs: []`, `supports.tracker_probe: true`). `mock.detect.v0`,
  `deepstream.decode_probe.v0`, `deepstream.inference_probe.v0`,
  `deepstream.metadata_probe.v0` are unchanged.
- The package's `supports` selects `DeepStreamWorker` mode:
  `decode_probe` / `inference_probe` / `metadata_probe` /
  **`tracker_probe`** (this milestone). Most-specific-wins ordering is
  `tracker > metadata > inference > decode`.
- Tracker mode reuses the P2 pyds-derived DS 8.0 image and the same
  in-container `pyds_metadata_app.py`, but splices an `nvtracker`
  element between `nvinfer` and `nvvideoconvert` (low-level: IOU, no
  ReID model required) and attaches the pad probe to the **tracker src
  pad** so `NvDsObjectMeta.object_id` is populated. The status payload
  gains `tracker_enabled`, `tracker_algorithm`,
  `unique_tracker_ids_count`, and a bounded `tracker_ids_sample`.
  `metadata_probe` mode is unchanged (no tracker fields), so operators
  can tell the two modes apart from the status row alone.
- Tracker config (`tracker_config.yml`) is generated by
  `build_tracker_config("iou")` and dropped into the per-run temp dir
  alongside `pgie.txt` + the pyds app, then bind-mounted into the
  container. Container name prefix flips from `upstream-dsm-` to
  `upstream-dst-` for tracker runs.
- **Status/debug evidence only — no observations, no metadata adapter
  yet.** No nvdsanalytics, no line crossing, no zones, no API change.

Run it (on the GPU/streaming host, with an active validated session):

```sh
UPSTREAM_FORCE_PACKAGE_ID=deepstream.tracker_probe.v0 \
UPSTREAM_DEEPSTREAM_PYDS_IMAGE=upstream-deepstream:8.0-pyds-v1.2.2 \
UPSTREAM_DEEPSTREAM_METADATA_PROBE_TIMEOUT_SECONDS=180 \
UPSTREAM_PROCESSOR_ID=proc_p3_tracker_live \
uv run upstream-processor
```

## P3.5 scope (supervisor-managed analytics probe)

- New package `deepstream.analytics_probe.v0` (runtime `deepstream`,
  `outputs: []`, `supports.analytics_probe: true`). Existing packages
  (mock, decode_probe, inference_probe, metadata_probe, tracker_probe)
  are unchanged. Mode dispatch is now most-specific-wins
  `analytics > tracker > metadata > inference > decode`.
- Analytics mode reuses the P3 tracker pipeline and adds an
  `nvdsanalytics` element after `nvtracker`. The pad probe attaches to
  `analytics.src` (one element further down than P3) and additionally
  walks `frame_user_meta_list` / `obj_user_meta_list` for the SDK's
  `NVDS_FRAME_META_NVDSANALYTICS` (→ `NvDsAnalyticsFrameMeta`:
  `objInROIcnt` / `objLCCumCnt` / `objLCCurrCnt` / `ocStatus`) and
  `NVDS_OBJ_META_NVDSANALYTICS` (→ `NvDsAnalyticsObjInfo`:
  `roiStatus` / `lcStatus` / `dirStatus` / `ocStatus`) user-meta.
  Frame-level analytics is folded under `frame.source["analytics"]`;
  per-object under `object.debug["analytics"]`.
- v0 ships a hand-authored config (`build_analytics_config(...)`):
  one ROI (central rectangle 20-80% wide × 10-90% tall) + one
  horizontal line crossing (x ∈ 25-75% at mid-height), `mode=loose`,
  `class-id=-1` (any class trips). Dashboard-driven geometry is a
  later milestone.
- Status payload (in addition to all metadata + tracker fields) adds
  `analytics_enabled`, `has_analytics_metadata`, `roi_metadata_count`,
  `line_metadata_count`, `overcrowding_metadata_count`,
  `direction_metadata_count`, `objects_with_roi_metadata`,
  `objects_with_line_metadata`, `analytics_labels_count`, and a
  FIFO-bounded `analytics_labels_sample`. Container name prefix
  flips from `upstream-dst-` to `upstream-dsa-`.
- **Status/debug evidence only — no observations, no metadata adapter
  yet.** No rules/events, no dashboard overlay, no API change.

Run it (on the GPU/streaming host, with an active validated session;
the operator must walk THROUGH the central ROI / across the middle line
for the SDK to emit ROI / LC events):

```sh
UPSTREAM_FORCE_PACKAGE_ID=deepstream.analytics_probe.v0 \
UPSTREAM_DEEPSTREAM_PYDS_IMAGE=upstream-deepstream:8.0-pyds-v1.2.2 \
UPSTREAM_DEEPSTREAM_METADATA_PROBE_TIMEOUT_SECONDS=180 \
UPSTREAM_PROCESSOR_ID=proc_p35_analytics_live \
uv run upstream-processor
```

## What it deliberately does NOT do (yet)

The DeepStream analytics probe (P3.5) runs `nvinfer` + `nvtracker` +
`nvdsanalytics` and produces neutral metadata with **real stable
tracker IDs** plus **ROI / line-crossing / direction events**, but it
is still a **probe** — the worker validates frames, accumulates
counters, and surfaces them in the status payload (`processing` lines
on `/v1/internal/processor/status`). It does NOT yet write
observations. Explicitly out of scope: **`ObservationWriter`
integration**, **`POST /observations`**, the final metadata adapter,
dashboard-driven ROI / line geometry (the analytics config is
hand-authored in `build_analytics_config`), rules / events, dashboard
UI, package catalog, billing, Triton/nvinferserver, YOLO/custom
models, multi-stream batching. API contracts are unchanged. The
`mock` runtime, the M8.2B decode probe, the M8.2C1-B inference probe,
the P2 metadata probe, and the P3 tracker probe are unchanged; M8.1's
`internal_rtsp_url`-is-never-read rule still holds for the mock
runtime.

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
| `UPSTREAM_DEEPSTREAM_PYDS_IMAGE` | no | `upstream-deepstream:8.0-pyds-v1.2.2` | Pyds-derived DeepStream 8.0 image for the **metadata probe** (built from `docker/Dockerfile.pyds`) |
| `UPSTREAM_DEEPSTREAM_METADATA_PROBE_TIMEOUT_SECONDS` | no | `60` | Bounded **metadata**-probe duration cap |
| `UPSTREAM_DEEPSTREAM_METADATA_QUEUE_MAXSIZE` | no | `256` | Max slots in the producer→consumer bridge queue inside the worker |
| `UPSTREAM_DEEPSTREAM_METADATA_QUEUE_DROP_POLICY` | no | `drop_newest` | `drop_newest` \| `drop_oldest` — eviction when the queue is full |
| `UPSTREAM_DEEPSTREAM_METADATA_SAMPLE_LIMIT` | no | `3` | In-container app: object samples per frame in each `UPSTREAM_METADATA=` line |
| `UPSTREAM_FINAL_STATUS_TIMEOUT_SECONDS` | no | `15` | Per-worker budget for graceful stop + final `stopped` post (DeepStream-safe; mock exits faster) |
| `UPSTREAM_OVERALL_SHUTDOWN_TIMEOUT_SECONDS` | no | `25` | Overall supervisor shutdown cap before workers are force-cancelled |

The secret is a `SecretStr` and is never logged (a redaction filter on the log
handler scrubs it, the header, and JWT-shaped strings as defence-in-depth).

## Quality

```sh
make check     # ruff check + ruff format --check + mypy app + pytest
```
