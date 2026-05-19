# Architecture

Pure asyncio, single event loop, one `asyncio.Task` per session. No threads:
the M8.1 workload is all I/O (HTTP polls + timed posts).

## Components

### `main`
Composition root. Loads `Settings`, configures logging (with the secret
redaction filter), installs SIGINT/SIGTERM handlers (`loop.add_signal_handler`
→ set an `asyncio.Event`), builds the shared `httpx.AsyncClient` /
`UpstreamApiClient`, the `SessionPoller`, the `WorkerManager`, and runs the
reconcile loop. Exit codes: `0` clean, `2` fatal auth, `3` fatal contract,
`1` unexpected.

### `SessionPoller`
Thin wrapper over `GET /eligible-sessions`. Unwraps the `{"sessions":[...]}`
envelope into a list; lets typed errors propagate so the manager owns policy.

### `WorkerManager` (the reconcile loop)
Single-writer over a `dict[session_id → handle]` (asyncio is cooperative, so
no locks). Each tick: poll → reap finished tasks → start workers for newly
eligible sessions → stop workers for sessions that dropped out → sleep.
Guarantees: never two workers for one session; a reappearing session gets a
fresh worker; a crashed worker is reaped and respawned next tick unless it
crash-loops (`max_restarts` within `restart_window` → quarantine); an
unresolvable package is reported `failed` once and quarantined, never spawned.
`ApiAuthError` (401) and a GET `ApiValidationError` (422) bubble out as fatal;
everything else on the poll is transient and backed off.

### `MockWorker` (the runtime, `runtimes/mock_runtime`)
One per session. Lifecycle: `starting → attached → processing` (status pulse
on an interval) with three independent observation pulses
(`mock_frame_seen` / `mock_heartbeat` / `mock_detection`). Terminal: `stopped`
on graceful stop, `failed` on a contract/internal error, or **quiet exit with
no terminal post** on 404 `session_not_found` (the session no longer exists to
post against). Transient API errors are absorbed with per-worker backoff
inside the pulses. The worker **never reads `internal_rtsp_url`** (enforced by
an AST-level test).

### Shutdown
Signal → event set → reconcile loop exits → `_shutdown_all` requests stop on
every worker, then awaits each best-effort final `stopped` under
`final_status_timeout`, all under an overall `wait_for(overall_timeout)` with a
hard `task.cancel()` escape hatch so a dead API cannot hang exit.

These timeouts are **global** (shared by all runtimes). M8.2B live
verification showed the original mock-tuned values (3s / 8s) were too tight
for a Docker-backed worker, so the defaults are now DeepStream-safe —
`final_status_timeout_seconds=15`, `overall_shutdown_timeout_seconds=25`,
`deepstream_process_stop_timeout_seconds=5` — and remain safe for mock
(it exits well inside them; they are upper bounds, not delays). All stay
env-overridable.

## DeepStream decode-probe runtime (M8.2B)

`runtimes/base.Runtime` is the seam. `runtimes/deepstream_runtime/`
implements the same `run()` / `request_stop()` / `state` / `session_id`
interface, so the supervisor, poller, status path, and schemas are
unchanged. Runtime selection: `resolver.resolve(package_id)` returns a
`PackageSpec`; `worker_manager.build_worker` dispatches on `spec.runtime`
(`mock` → `MockWorker`, `deepstream` → `DeepStreamWorker`).
`UPSTREAM_FORCE_PACKAGE_ID` lets a mock-defaulted session route to the
DeepStream package without changing production config.

- **`DeepStreamProcessManager`** owns a Docker subprocess running the
  M8.2A-proven pipeline (`rtspsrc → rtph264depay → h264parse →
  nvv4l2decoder → nvvideoconvert → fpsdisplaysink`, `--gpus all
  --network host`). It builds the argv, streams stdout, and on stop does
  `docker stop` → SIGTERM → SIGKILL-after-`stop_timeout`. The RTSP URL is
  only ever in the subprocess argv; everything logged is redacted
  (`rtsp://host:port/sess_<redacted>`), and any `rtsp://` echoed by
  `gst-launch -v` is scrubbed before it can reach a log/error.
- **`DeepStreamWorker`** consumes the eligible session's
  `internal_rtsp_url`, posts `starting`, starts the subprocess, and parses
  `fpsdisplaysink` lines: first `rendered: N>0` → `attached`; subsequent
  fps → throttled `processing` posts carrying `fps_processed`. On graceful
  stop / clean exit codes → `stopped`; on unexpected nonzero exit →
  `failed` (with `error_code`/redacted `error_message`). Status rows report
  `runtime="deepstream"` / the resolved `package_id` (the M8.0 session
  itself still says `mock`). **No observations** are emitted in M8.2B.

The contract-legal `lagging` status is still unused; it is reserved for a
later production inference runtime.

## DeepStream inference-probe runtime (M8.2C1-B)

Same seam, same `DeepStreamWorker`. The resolved package's `supports`
selects the mode: `supports.decode_only` → decode (M8.2B, unchanged);
`supports.inference_probe` (`deepstream.inference_probe.v0`) → inference.
`mock.detect.v0` and `deepstream.decode_probe.v0` are untouched.

- **`inference.py`** — pure builders (`build_pgie_config`,
  `build_app_config`, `build_inference_docker_command`) +
  **`DeepStreamInferenceProcessManager`**. The PM creates a per-run **temp
  work dir** (deepstream-app + PGIE config + a `kitti/` output dir),
  bind-mounted into the container at the same path, and runs the
  M8.2C1-A-proven DeepStream 8.0 pipeline: `timeout -s INT <t> deepstream-app`
  with RTSP/TCP `source` → `nvstreammux` → `nvinfer` (bundled TrafficCamNet
  ONNX, INT8, engine built fresh) → `fakesink`, no display
  (`--gpus all --network host -v wd:wd`). It streams stdout, does the same
  `docker stop` → SIGTERM → SIGKILL-after-`stop_timeout`, exposes
  `kitti_stats()` (frame/detection-line counts), and removes the temp dir on
  stop unless `keep_temp_files`. The RTSP URL is only in the temp `ds_app.txt`
  + argv; logs are redacted; `rtsp://` echoed in stdout is scrubbed.
- **`DeepStreamWorker` (inference mode)** posts `starting`, starts the
  subprocess, and parses the proven M8.2C1-A markers: first
  source/nvinfer line (`deserialize engine from file`, `NvDsInferContext`,
  `nvstreammux`, …) → `attached`; engine-built (`serialize cuda engine to
  file: …successfully`) or a `**PERF` fps line → `processing` (throttled),
  payload carrying `inference_enabled`, `inference_engine_built`, `model`,
  `image`, `fps_processed`, and KITTI `detection_frames_count` /
  `detection_count`. Graceful stop / clean exit (incl. `124` from the bounded
  `timeout`) → `stopped`; `Unsupported SM` / TensorRT / CUDA / core-dump →
  `failed` with `error_code` (`deepstream_unsupported_sm` /
  `deepstream_trt_error` / `deepstream_exit`) and a **redacted**
  `error_message`. Status rows report `runtime="deepstream"` /
  `deepstream.inference_probe.v0`.

Default image is **DeepStream 8.0** (`UPSTREAM_DEEPSTREAM_IMAGE`): 7.1's
TensorRT 10.3 fails inference on Blackwell sm_120 (M8.2C1-A); 8.0's TensorRT
10.9 / CUDA 12.8 works. **No observations** in either mode. The inference
probe runs `nvinfer` but is still a probe — KITTI output is debug/inference
evidence only, **not** metadata extraction. Out of scope: pyds, pad probes,
MetadataSource, metadata adapter, observations, tracking, nvtracker,
nvdsanalytics, line crossing, zones, Triton, dashboard, package catalog,
billing. API contracts unchanged.
