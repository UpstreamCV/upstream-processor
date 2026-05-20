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

## P2 — manual pyds metadata probe (out-of-tree)

A *manual* probe that proves the DeepStream metadata seam (no runtime worker
change, **no `app/` code touched**). It builds, inside a thin
`upstream-deepstream:8.0-pyds-v1.2.2` image (DS 8.0 +
NVIDIA-published `pyds-1.2.2` wheel for cp312 / Ubuntu 24.04), an RTSP →
`nvstreammux` → `nvinfer` (TrafficCamNet) → `fakesink` pipeline and attaches
a pad probe on the `nvinfer` src pad that walks
`pyds.gst_buffer_get_nvds_batch_meta` → `NvDsBatchMeta.frame_meta_list`
(`NvDsFrameMeta.cast`) → `obj_meta_list` (`NvDsObjectMeta.cast`) and pulls
`class_id` / `obj_label` / `confidence` / `rect_params` / `object_id`. The
probe emits a single `PYDS_PROBE_SUMMARY=<json>` line with counters and a
small bounded sample shaped per the P2 "neutral metadata" spec
(`runtime`, `metadata_source: "pyds_pad_probe"`, `frame.{frame_num,width,height}`,
`objects[*].{class_id,label,confidence,bbox_pixels,bbox_normalized,tracker_id}`).

Files: `docker/Dockerfile.pyds`, `scripts/_pyds_probe_helpers.py`,
`scripts/deepstream_pyds_metadata_probe.py`,
`scripts/probe_deepstream_pyds_metadata.sh`,
`tests/test_pyds_probe_helpers.py`. The probe is **not imported by `app/`**;
mypy/ruff cover only the pure helpers (the gi/pyds-touching code lives in
the probe script). The pad probe does near-zero work — no HTTP, no async,
no UpStream API writes. The raw RTSP URL appears only in the per-run temp
`pgie.txt` / subprocess argv; all logs use the redacted form. The
TrafficCamNet engine builds fresh on first run (Blackwell sm_120, ~30 s).

This is **not** the MetadataSource and **not** the metadata adapter — those
land in the next milestones. KITTI/log output here is debug evidence only.
Still **no observations, no pyds in `app/`, no pad probes in `app/`, no
nvtracker, no nvdsanalytics, no Triton, no dashboard, no API change**.

## P2 — `MetadataSource` Protocol + bounded bridge queue (`app/`)

Promotes the **neutral metadata shape** the manual probe proved into typed
``app/`` abstractions, so a future supervisor-managed pyds worker can
produce structured metadata into a thread-safe queue and an asyncio
consumer can drain it — without leaking pyds / Gst / DeepStream C structs
into ``app/``, and without writing any observations yet.

- **`app/observations/metadata_models.py`** — pydantic v2 models, all
  ``extra="forbid"``:
  - ``BoundingBoxPixels``, ``BoundingBoxNormalized``
  - ``NeutralObjectMetadata`` — ``class_id`` / ``label?`` / ``confidence?``
    / ``bbox_pixels?`` / ``bbox_normalized?`` / ``tracker_id?: str``.
    ``tracker_id`` is ``str`` to stay JSON-safe across uint64 (JS
    `Number.MAX_SAFE_INTEGER`).
  - ``NeutralFrame`` — ``frame_num`` (NonNegativeInt) / ``width`` /
    ``height`` / optional ``timestamp``.
  - ``NeutralFrameMetadata`` — ``runtime`` / ``metadata_source`` /
    optional ``stream_session_id`` / ``frame`` / ``objects[]`` /
    ``source`` + ``debug`` dicts (DeepStream-specific extras stay here,
    NOT as first-class fields, to keep the consumer side runtime-neutral).
  - Helpers: ``bbox_normalized_from_pixels(...)`` (same rounding as the
    probe) and ``tracker_id_from_pyds(raw_uint64)`` (maps
    ``UNTRACKED_OBJECT_ID`` → ``None``, else ``str(raw)``).
  These models accept the EXACT dict the manual probe emitted live — that
  is asserted by ``tests/test_metadata_models.py::test_parses_real_probe_sample``.
  They are **NOT** ``ObservationPayload`` (the final UpStream observation
  schema lives in ``app/observations/schemas.py`` and is a P4 concern).

- **`app/observations/bridge_queue.py`** — ``BoundedBridgeQueue`` +
  ``DropPolicy`` (``DROP_NEWEST`` / ``DROP_OLDEST``) +
  ``BridgeQueueStats``. Producer side is **non-blocking and thread-safe**
  (``put_from_thread`` returns ``bool``); consumer side is awaitable
  (``await q.get()`` returns the next ``NeutralFrameMetadata`` or
  ``None`` when closed + drained). Implementation: an internal
  ``queue.Queue`` (unbounded slot; we gate ``maxsize`` ourselves under a
  ``threading.Lock`` so ``DROP_OLDEST`` can evict atomically) and
  ``asyncio.to_thread`` for the async ``get`` path. A sentinel pushed on
  ``close()`` unblocks any waiting consumer. Designed for **a single
  async consumer** (the future supervisor worker); multi-consumer fan-out
  is out of scope.

- **`app/runtimes/metadata_source.py`** — ``MetadataSource`` Protocol
  (``__aiter__ -> AsyncIterator[NeutralFrameMetadata]``,
  ``@runtime_checkable``) + ``BridgeQueueMetadataSource``, the canonical
  implementation that wraps a ``BoundedBridgeQueue``. Pyds/Gst/DeepStream
  objects must not leak through this seam.

Coverage: ``tests/test_metadata_models.py``,
``tests/test_bridge_queue.py``, ``tests/test_metadata_source.py`` exercise
construction, JSON round-trip, ``extra="forbid"`` at every level, both
drop policies, close ordering, async-wakes-on-close, a real
``threading.Thread`` producer + slow asyncio consumer with bounded
memory + no deadlocks, and Protocol conformance. **No Docker, no GPU, no
pyds/gi imports in tests.**

Still **no `app/` runtime worker pyds integration, no pad probe in
`app/`, no `ObservationWriter` integration, no `POST /observations`, no
nvtracker, no nvdsanalytics, no line crossing / zones, no Triton, no
dashboard, no API change**. The manual P2 probe in ``scripts/`` continues
to be the standalone *runtime* proof; this milestone is purely the
**`app/`-side seam** the next milestone (supervisor-managed metadata
probe worker) will build on top of.

## P2 — supervisor-managed metadata probe worker

Connects the manual pyds metadata pipeline to the existing
``WorkerManager`` lifecycle via a new package
``deepstream.metadata_probe.v0`` (``supports.metadata_probe: true``).
``DeepStreamWorker``'s mode dispatch is now a 3-way: metadata > inference
> decode (most specific support wins). The container never calls UpStream
APIs — the webhook secret never enters the container; status posting
stays with the outer supervisor.

- **`app/runtimes/deepstream_runtime/pyds_metadata_app.py`** — the
  **in-container** pyds Python app. Bind-mounted into a DS 8.0 + pyds
  container; never imported by any other ``app/`` code. Builds the
  proven manual probe pipeline (``rtspsrc(tcp, latency=200) → rtph264depay
  → h264parse → nvv4l2decoder → nvstreammux(batch=1) → nvinfer →
  nvvideoconvert → fakesink``), attaches a pad probe on ``nvinfer.src``,
  walks ``pyds.gst_buffer_get_nvds_batch_meta → NvDsFrameMeta →
  NvDsObjectMeta``, and emits structured per-frame stdout lines
  ``UPSTREAM_METADATA=<json>`` plus a terminal
  ``UPSTREAM_METADATA_SUMMARY=<json>``. The pad probe does near-zero
  work: build dict → ``sys.stdout.write`` → ``flush``. No HTTP, no
  async, no ``ObservationWriter`` calls. The raw RTSP URL is consumed
  via ``RTSP_URL`` env (the outer worker passes it) and is redacted in
  any GST_ERROR text. mypy/ruff skip this module (``[[tool.mypy.overrides]]``)
  because gi/pyds only resolve at run time inside the container.

- **`app/runtimes/deepstream_runtime/metadata_probe.py`** —
  ``DeepStreamMetadataProcessManager`` + pure
  ``build_metadata_docker_command``. Owns a per-run temp work dir
  holding ``pgie.txt`` (reused ``build_pgie_config`` from
  ``inference.py``, TrafficCamNet INT8) and a copy of the in-container
  app. Runs ``docker run --rm --gpus all --network host -v wd:wd -e
  PYTHONDONTWRITEBYTECODE=1 -e RTSP_URL=... <pyds-image> bash -lc
  "timeout -s INT <t> python3 -B wd/pyds_metadata_app.py …"``. Streams
  stdout, does the same ``docker stop`` → SIGTERM → SIGKILL-after-
  ``stop_timeout``, and shreds the temp dir on stop unless
  ``keep_temp_files``.

- **`DeepStreamWorker._drive_metadata_probe`** — runs two asyncio tasks
  concurrently:
  1. **Reader**: pulls every stdout line, classifies via the pure
     ``parse_metadata_line`` (``UPSTREAM_METADATA=`` / ``UPSTREAM_METADATA_SUMMARY=``
     / other), JSON-parses, **validates with**
     :class:`~app.observations.metadata_models.NeutralFrameMetadata`,
     and pushes valid frames into a per-worker
     :class:`~app.observations.bridge_queue.BoundedBridgeQueue` via
     ``put_from_thread`` (the queue seam is exercised exactly the way a
     future real pyds thread will use it). Tracks
     ``hard_fail_code``/``last_err`` for terminal status. On EOF/stop,
     closes the queue.
  2. **Consumer**: iterates a
     :class:`~app.runtimes.metadata_source.BridgeQueueMetadataSource`
     and feeds an ``_MetadataCounters`` aggregator. On the first frame
     consumed, posts ``attached`` (gating); thereafter posts
     ``processing`` immediately, then throttled at
     ``status_interval_seconds``.

  On both tasks finishing, the worker posts terminal ``stopped`` (clean
  exit code + no hard-fail) or ``failed`` (otherwise) with the same
  status payload shape:

  ``{runtime, mode="metadata_probe", metadata_enabled, metadata_source,
    model, image, frames_seen, frames_with_objects, objects_seen,
    class_counts, label_counts, has_bbox, has_confidence,
    has_tracker_id, sample_count, queue_stats, app_summary?}``

- **Config** (``app/config.py``): ``deepstream_pyds_image`` (default
  ``upstream-deepstream:8.0-pyds-v1.2.2``),
  ``deepstream_metadata_probe_timeout_seconds`` (60),
  ``deepstream_metadata_queue_maxsize`` (256),
  ``deepstream_metadata_queue_drop_policy`` (``drop_newest``),
  ``deepstream_metadata_sample_limit`` (3). All env-overridable. Reuses
  ``deepstream_keep_temp_files`` and the existing shutdown-budget knobs.

- **Shutdown ordering** (P2 cleanup hardening): ``request_stop()``
  *synchronously* closes the bridge queue, so a consumer parked in
  ``asyncio.to_thread(queue.Queue.get)`` wakes within milliseconds
  instead of waiting for the reader's next 0.5 s poll. The driver wraps
  the inner ``asyncio.gather`` in a ``try`` / ``finally`` and runs the
  cleanup tail (close queue → ``_safe_stop_pm`` → terminal status post)
  inside ``asyncio.shield(...)``, so docker stop + workdir cleanup + the
  final status post complete even when the worker task is cancelled by
  the supervisor's force-stop path. ``DeepStreamMetadataProcessManager.stop()``
  and ``_cleanup_work_dir()`` are idempotent (a second call is a no-op).

Coverage: ``tests/test_deepstream_metadata_probe.py`` exercises mode
selection (and verifies decode/inference still resolve), the pure parser
(frame / summary / other), the pure command builder
(``--gpus all`` / ``--network host`` / ``-v wd:wd`` /
``PYTHONDONTWRITEBYTECODE`` env / ``RTSP_URL`` env / inner argv), the
full lifecycle (starting → attached → processing → stopped with counters
+ ``queue_stats`` + ``app_summary``), ``Unsupported SM`` → ``failed``
with redacted ``error_message``, nonzero exit without metadata,
graceful ``request_stop``, malformed-JSON / schema-invalid lines being
skipped, missing RTSP fast-fail, and the real PM's temp-dir write +
cleanup (monkeypatched ``create_subprocess_exec``). **No Docker, no GPU,
no pyds/gi anywhere in tests.**

Out of scope (later milestones): ``ObservationWriter`` integration,
``POST /observations``, the final metadata adapter, ``nvtracker``,
``nvdsanalytics``, line crossing, zones, Triton, dashboard, rules.

## P3 — Tracker Metadata Probe

Promotes the pyds metadata probe from "detection metadata only" to "real
stable tracker IDs across frames" by splicing ``nvtracker`` into the
existing pipeline. Same supervisor, same status lifecycle, same pyds
container — only the GStreamer pipeline + the in-container app's CLI +
the status payload gain a small surface. Still **no observations**.

- **Package** — ``deepstream.tracker_probe.v0`` with
  ``supports.tracker_probe: true``. ``mock.detect.v0``,
  ``deepstream.decode_probe.v0``, ``deepstream.inference_probe.v0``, and
  ``deepstream.metadata_probe.v0`` are untouched. Mode dispatch order in
  ``DeepStreamWorker._mode`` is most-specific-wins:
  tracker > metadata > inference > decode.

- **Tracker choice** — NvTracker low-level **IOU** algorithm
  (``%YAML:1.0`` config generated by
  :func:`~app.runtimes.deepstream_runtime.inference.build_tracker_config`,
  modelled on the DS 8.0 sample ``config_tracker_IOU.yml``). IOU needs
  no ReID model and exercises the stable-ID path; NvDCF / NvSORT /
  NvDeepSORT are reserved for future tuning. Library is the in-container
  ``libnvds_nvmultiobjecttracker.so`` (DS 8.0 default). Tracker
  width/height default to ``960 × 544`` (must be multiples of 32 for
  NvDCF / NvDeepSORT; harmless for IOU).

- **Pipeline shape** (additions in **bold**):

  ``rtspsrc(tcp, latency=200) → rtph264depay → h264parse →
  nvv4l2decoder → nvstreammux(batch=1) → nvinfer(<pgie.txt>) →
  **nvtracker(<tracker_config.yml>)** → nvvideoconvert → fakesink``

  The pad probe attaches to the **tracker src pad** instead of the
  nvinfer src pad, so ``NvDsObjectMeta.object_id`` is populated when
  the probe walks the meta lists. The in-container app
  (``pyds_metadata_app.py``) accepts optional
  ``--tracker-config-path`` / ``--tracker-lib-path`` /
  ``--tracker-width`` / ``--tracker-height`` CLI args; metadata-probe
  mode is unchanged when those args are absent.

- **Process manager** — :class:`~app.runtimes.deepstream_runtime.metadata_probe.DeepStreamMetadataProcessManager`
  gained a ``tracker_enabled`` flag and the four tracker knobs. When set,
  ``_prepare_work_dir`` also writes ``tracker_config.yml`` into the
  per-run temp dir; ``build_metadata_docker_command`` injects the
  tracker CLI args; the container name prefix flips from
  ``upstream-dsm-`` to ``upstream-dst-`` so the two modes are
  distinguishable in ``docker ps``. Cleanup, shutdown hardening, RTSP
  redaction, and the docker stop chain are unchanged.

- **Status payload** — when ``self._mode == "tracker_probe"``, the
  metadata payload gains:

  - ``tracker_enabled: true``
  - ``tracker_algorithm: "iou"`` (or whatever ``UPSTREAM_DEEPSTREAM_TRACKER_ALGORITHM`` was set to)
  - ``unique_tracker_ids_count`` — total distinct IDs observed
  - ``tracker_ids_sample`` — FIFO-bounded list, capped at
    ``UPSTREAM_DEEPSTREAM_TRACKER_IDS_SAMPLE_LIMIT`` (default 10), so
    the payload stays small under busy scenes.

  ``has_tracker_id`` continues to mean "did the producer ever attach a
  non-null ``tracker_id``?" — true in tracker mode the moment the first
  tracked detection arrives. ``metadata_probe`` mode is unchanged; it
  does NOT advertise the tracker fields, so operators can tell the two
  modes apart from the status row alone.

- **Config** (``app/config.py``): ``deepstream_tracker_lib_path``
  (default ``/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so``),
  ``deepstream_tracker_algorithm`` (``iou``),
  ``deepstream_tracker_width`` (960),
  ``deepstream_tracker_height`` (544),
  ``deepstream_tracker_ids_sample_limit`` (10). All env-overridable.
  Reuses every existing pyds-image / queue / sample / timeout knob.

Coverage: ``tests/test_deepstream_tracker_probe.py`` (13 tests) exercises
package resolution, mode dispatch, the pure ``build_tracker_config``
helper (and its unknown-algorithm rejection), the docker command
builder with and without tracker args, the real PM's tracker_config.yml
write + cleanup (+ unknown-algorithm pre-flight validation + the
container-name prefix flip), the full ``tracker_probe`` lifecycle (with
tracker_ids ``"101", "101", "102"`` → ``unique_tracker_ids_count = 2``
and bounded sample ``["101", "102"]``), bounded sample under 6 unique
IDs, and a regression that ``metadata_probe`` mode does NOT add tracker
fields. ``tests/test_package_resolver.py`` adds a tracker-package case.
**No Docker, no GPU, no pyds/gi anywhere in tests.**

Out of scope (still): ``ObservationWriter`` integration,
``POST /observations``, the final metadata adapter, ``nvdsanalytics``,
line crossing, zones, Triton, dashboard, rules.
