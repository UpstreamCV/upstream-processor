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

## P3.5 — Analytics Metadata Probe

Splices ``nvdsanalytics`` into the tracker pipeline so the SDK's ROI /
line-crossing / overcrowding / direction metadata is attached to each
``NvDsFrameMeta`` (under ``frame_user_meta_list``) and ``NvDsObjectMeta``
(under ``obj_user_meta_list``); the pad probe walks both and folds the
result into the worker status payload. v0 ships a hand-authored ROI +
line config; dashboard-driven geometry is a later milestone. Still
**no observations**.

- **Package** — ``deepstream.analytics_probe.v0`` with
  ``supports.analytics_probe: true``. Existing packages are untouched.
  Mode dispatch in ``DeepStreamWorker._mode`` is now
  ``analytics > tracker > metadata > inference > decode``.

- **Pipeline shape** (P3.5 addition in **bold**):

  ``rtspsrc(tcp, latency=200) → rtph264depay → h264parse →
  nvv4l2decoder → nvstreammux(batch=1) → nvinfer(<pgie.txt>) →
  nvtracker(<tracker_config.yml>) →
  **nvdsanalytics(<analytics_config.txt>)** → nvvideoconvert →
  fakesink``

  Pad probe attaches to **``analytics.src``** (one element further down
  than P3) so by the time it walks the meta lists, both the tracker has
  populated ``object_id`` *and* nvdsanalytics has attached its
  ``NVDS_*_META_NVDSANALYTICS`` user-meta to frames + tracked objects.

- **Config builder** —
  :func:`~app.runtimes.deepstream_runtime.inference.build_analytics_config`
  returns an INI body modelled on the DS 8.0 sample
  ``config_nvdsanalytics.txt`` but stripped to one ROI + one
  line-crossing rule. v0 geometry: ROI is a central polygon spanning
  20–80% of frame width and 10–90% of frame height; the line is
  horizontal across the middle of that ROI (x ∈ 25–75%). Direction
  reference points are seeded perpendicular to the line so "any
  crossing" trips the LC event in ``mode=loose``. Frame dimensions are
  validated up front (``ValueError`` on non-positive). Pure / side-effect
  free.

- **Process manager** —
  :class:`~app.runtimes.deepstream_runtime.metadata_probe.DeepStreamMetadataProcessManager`
  gains an ``analytics_enabled`` flag that requires
  ``tracker_enabled`` (nvdsanalytics operates on tracked objects;
  both the PM constructor and the pure docker command builder raise
  ``ValueError`` if asked to enable analytics without tracker). When
  set, ``_prepare_work_dir`` also writes ``analytics_config.txt`` into
  the per-run temp dir; the container name prefix flips from
  ``upstream-dst-`` to ``upstream-dsa-``. Cleanup, shutdown hardening,
  RTSP redaction, and the docker stop chain are unchanged.

- **In-container pyds app** —
  ``pyds_metadata_app.py`` accepts an optional ``--analytics-config-path``
  CLI arg; when set, it requires ``--tracker-config-path``, splices
  ``nvdsanalytics`` after ``nvtracker``, and attaches the pad probe
  to the analytics src pad. The probe walks
  ``frame_meta.frame_user_meta_list`` for entries whose
  ``user_meta.base_meta.meta_type == pyds.NVDS_FRAME_META_NVDSANALYTICS``
  and casts the payload via ``pyds.NvDsAnalyticsFrameMeta.cast(...)``;
  similarly for ``obj_meta.obj_user_meta_list`` →
  ``pyds.NVDS_OBJ_META_NVDSANALYTICS`` →
  ``pyds.NvDsAnalyticsObjInfo.cast(...)``. Frame-level fields surfaced:
  ``objInROIcnt`` / ``objLCCumCnt`` / ``objLCCurrCnt`` / ``ocStatus``;
  object-level: ``roiStatus`` / ``lcStatus`` / ``dirStatus`` /
  ``ocStatus``. Frame-level analytics lands under
  ``frame.source["analytics"]``; per-object analytics under
  ``object.debug["analytics"]``. ``NeutralObjectMetadata`` gained a
  ``debug: dict[str, Any]`` mirror of the frame-level field so the
  schema stays ``extra="forbid"`` at every other level.

- **Status payload** — when ``self._mode == "analytics_probe"`` the
  payload (in addition to all metadata + tracker fields) carries:
  ``analytics_enabled``, ``has_analytics_metadata``,
  ``roi_metadata_count`` (frames with ROI user-meta),
  ``line_metadata_count`` (frames with LC user-meta),
  ``overcrowding_metadata_count``, ``direction_metadata_count``,
  ``objects_with_roi_metadata``, ``objects_with_line_metadata``,
  ``analytics_labels_count`` (total distinct ROI / LC / direction
  labels observed), and ``analytics_labels_sample`` (FIFO-bounded by
  ``UPSTREAM_DEEPSTREAM_ANALYTICS_LABELS_SAMPLE_LIMIT`` = 20). The
  ``tracker_probe`` and ``metadata_probe`` modes are unchanged — they
  do NOT advertise the analytics fields.

- **Config** (``app/config.py``):
  ``deepstream_analytics_labels_sample_limit`` (20).
  Env-overridable. Reuses every existing pyds-image / tracker /
  queue / sample / timeout knob.

Coverage: ``tests/test_deepstream_analytics_probe.py`` (14 tests)
exercises package resolution, mode dispatch (analytics wins over
tracker / metadata; tracker / metadata still hold their own ground), the
pure ``build_analytics_config`` helper (ROI + LC sections present,
``ValueError`` on bad dimensions / empty labels), the docker command
builder (rejects analytics-without-tracker; includes
``--analytics-config-path`` when both flags are set), the real PM
(``analytics_config.txt`` is written, ``upstream-dsa-`` container
prefix, cleanup on stop, pre-flight ValueError on
analytics-without-tracker), the full ``analytics_probe`` lifecycle
(frames with per-object ROI / LC / direction + frame-level
cumulative counts → ``has_analytics_metadata=True``,
``unique_tracker_ids_count=1``, label sample
``{"entry-roi", "entry-line", "dirSouth"}``), label-sample is FIFO-
bounded under 5 distinct labels with limit=3 → first three kept, an
analytics_probe run that surfaces ``analytics_enabled=True`` even when
no nvdsanalytics meta arrived (scene didn't trigger events), a
regression that ``tracker_probe`` mode does NOT add analytics fields,
malformed analytics payloads don't crash the worker, and the
``Unsupported SM`` redaction path still works in analytics mode.
``tests/test_package_resolver.py`` adds an analytics-package case.
**No Docker, no GPU, no pyds/gi anywhere in tests.**

Out of scope (still): ``ObservationWriter`` integration,
``POST /observations``, the final metadata adapter, dashboard-driven
ROI / line geometry, rules / events, billing.

## P4-A — Observation Adapter Contract Probe

Defines the contract (payload shape, idempotency, adapter state) for
the first two real processor observation types **without** writing
anything yet. The P3.5 analytics probe surfaces line-crossing
cumulative counters + ROI counts in the worker status payload; P4-A
locks in HOW those signals will turn into ``processor.*`` observation
rows once both ends of the wire are widened. **No new package, no
worker wiring, no API writes, no api-server change in this milestone.**

### API contract findings (verified offline)
- Deployed schema: ``upstream-api-server`` at commit ``7a6057a``,
  migration ``20260518140000_processor_status_observations.sql``.
- ``POST /v1/internal/processor/observations`` accepts a
  ``ProcessorObservationRequest`` with strict pydantic ``Literal``
  type validation. **Today only the three ``processor.mock_*`` types
  are accepted** — anything else 422s at the schema layer.
- DB table ``observations`` has columns ``id, account_id, camera_id,
  stream_session_id, processor_id, package_id, type, payload, observed_at,
  created_at``. ``type`` is plain ``text`` (no DB CHECK), so the
  whitelist widens with a one-string Python change on each side; **no
  migration is needed** to add types.
- There is **no ``idempotency_key`` column** and no unique constraint
  preventing duplicate rows. Duplicate POSTs create duplicate rows.
  Client-side dedupe is therefore the only mechanism in v0; we embed
  ``idempotency_key`` in ``payload`` so future server dedupe is a
  cheap migration if/when wanted.
- Client side (this repo) mirrors that strictness:
  ``schemas.ObservationType`` is a matching ``Literal`` and
  ``ObservationWriter.emit`` defensively rejects any type not in
  ``OBSERVATION_TYPES``.

### First observation types (candidate strings, not yet whitelisted)
- ``processor.line_crossing`` — one event per
  ``objLCCumCnt[<label>]`` increment. Maps directly off P3.5's
  ``line_crossing_cumulative`` (we live-saw 0→1→2). **First target.**
- ``processor.roi_occupancy`` — periodic / on-change snapshot of an
  ROI's count. Off P3.5's ``objInROIcnt`` (we live-saw
  ``entry-roi: 1``). Design only in P4-A; emission gated by
  per-ROI change detection + a min-interval heartbeat.

Both names live for now ONLY in
``app/observations/observation_adapter.py`` as
``CANDIDATE_LINE_CROSSING_TYPE`` / ``CANDIDATE_ROI_OCCUPANCY_TYPE`` —
**deliberately NOT added to** ``schemas.ObservationType`` /
``OBSERVATION_TYPES`` because that whitelist is the gate
``ObservationWriter`` uses. P4-B widens both ends in lock-step with
the matching api-server PR. A regression test
(``test_candidate_types_not_in_server_whitelist_yet``) asserts the
guard holds.

### Payload shape (runtime-neutral at the top level)
``processor.line_crossing`` payload body (goes inside the outer
``ObservationPayload.payload`` dict):

```
{
  "line_id":            "entry-line",
  "line_label":         "entry-line",
  "count_delta":        1,
  "cumulative_count":   2,
  "idempotency_key":    "sess:<uuid>:processor.line_crossing:entry-line:2",
  "track_id":           "0",          # optional
  "class_id":           2,            # optional
  "class_label":        "person",     # optional
  "direction":          "dirSouth",   # optional
  "source": {
    "runtime":            "deepstream",
    "package_id":         "deepstream.analytics_probe.v0",
    "metadata_source":    "pyds_pad_probe",
    "analytics_plugin":   "nvdsanalytics",
    "tracker_algorithm":  "iou"
  },
  "debug": {
    "frame_num":          1234,
    "geometry_source":    "probe_default"
  }
}
```

The outer ``ObservationPayload`` continues to carry
``stream_session_id``, ``type``, ``observed_at``, ``processor_id``,
``package_id`` — those stay on the existing client schema.
DeepStream vocabulary lives ONLY under ``source`` / ``debug`` so a
future Triton / OpenCV runtime can emit the same observation type
with a different ``source.runtime``.

### Idempotency strategy
**Line crossing key:**
``sess:<session_id>:processor.line_crossing:<line_id>:<cumulative_count>``.
``track_id`` deliberately omitted — ``nvdsanalytics``'
``objLCCumCnt`` is a per-LINE aggregate, not per-track. Track context
attaches as optional payload, not as a key component.

**ROI occupancy key:**
``sess:<session_id>:processor.roi_occupancy:<roi_id>:<sequence>``
where ``sequence`` is the adapter's per-ROI emit counter (one per
change/interval), NOT the raw frame number. Frame-number keys would
collide on flapping counts and balloon the emit rate.

Server has no dedupe today. Client carries the keys so that:
1. ``LineCrossingState.recently_emitted_keys`` (bounded set) acts as
   a belt-and-suspenders against producer rewind.
2. A future api-server PR can add an ``idempotency_key`` column and
   re-derive the key without coordinating with the processor.

### Adapter state design (per-stream, per-worker)
``LineCrossingState`` (the load-bearing one):
- ``last_cumulative_by_line: dict[str, int]`` — watermark per line.
  Counter going from ``N`` to ``N + k`` between two frames emits
  ``k`` discrete events with cumulative counts ``N + 1 .. N + k``.
  Counter unchanged → emit nothing. Counter rewound → emit nothing
  AND do not lower the watermark.
- ``recently_emitted_keys: set[str]`` — bounded LRU-ish guard against
  replay; eviction at ``recently_emitted_keys_max`` (default 256).
- Resets when the supervisor restarts the worker (each bounded
  probe cycle starts fresh). Acceptable today because the
  cumulative counter itself resets per container.

``RoiOccupancyState`` (designed, not yet wired):
- ``last_counts_by_roi`` (total + per-class snapshot keyed by ROI).
- ``last_emit_at_by_roi`` (per-ROI timestamp of the last emit).
- ``sequence_by_roi`` (per-ROI emit counter for idempotency keys).
- Emit conditions OR'd: (a) ``emit_on_change=True`` and counts
  differ from the last snapshot, (b) ``min_interval_seconds`` (30s
  default) since the last emit. v0 enables both so the consumer
  sees both "person stepped into ROI" and a slow heartbeat
  thereafter.
- Designed to scale to per-class breakdowns when the producer
  surfaces them; v0 leaves ``class_counts`` empty if the producer
  didn't break it down.

### ObservationWriter / sink review
``ObservationWriter`` already exists at
``app/observations/writer.py`` and is used by the mock worker. It:
- builds an ``ObservationPayload`` from a typed ``obs_type`` + body
  dict + ``observed_at``,
- guards the type via the ``OBSERVATION_TYPES`` whitelist (defensive
  even though the parameter is a ``Literal``),
- delegates to ``ProcessorApi.post_observation``.

**No bridge / sink redesign is needed for P4-B.** The path is:
1. The analytics worker's consumer task already runs on the event
   loop and already calls ``api.post_processor_status`` per frame.
2. P4-B adds adapter state to the worker (``LineCrossingState`` keyed
   by ``stream_session_id``), calls
   ``state.absorb_cumulative(...)`` per frame, and forwards each
   ``LineCrossingEvent.to_payload()`` to
   ``ObservationWriter.emit(...)``.
3. Transient errors absorb through the existing backoff; the
   writer's strict-Literal guard prevents accidentally posting a
   not-yet-whitelisted type.

The pad probe inside the pyds container does NOT change — it still
emits raw analytics counters via stdout JSON. **All adapter logic
lives outside the container** (worker-side), so:
- The webhook secret continues to never enter the container.
- Per-event API writes don't block the GLib thread.
- A flaky network only delays the worker, never the GST pipeline.

### What must change in api-server before P4-B can POST
- One-string widening of ``ProcessorObservationType`` in
  ``upstream-api-server/src/upstream_api/schemas/processor.py`` to
  include ``processor.line_crossing`` (and later
  ``processor.roi_occupancy``).
- No DB migration. No route change. No new auth.
- Optionally (P4-C): add an ``idempotency_key`` column +
  ``UNIQUE(stream_session_id, idempotency_key)``. **Not required**
  for P4-B — client-side state dedupes within a container lifetime.

### Coverage
``tests/test_observation_adapter_contract.py`` (31 tests, no Docker /
GPU / pyds / api-server) exercises:
- both ``CANDIDATE_*`` strings are absent from the server whitelist
  (so the probe cannot accidentally write to production),
- both use the ``processor.<thing>`` namespace,
- idempotency keys are deterministic, session-scoped, and change
  with cumulative count / sequence,
- ROI keys use a per-emit sequence (not frame number),
- ``LineCrossingState`` emits nothing at initial zero, one event on
  0→1, two on 0→1→2, three on a 0→3 jump (with consecutive
  cumulative counts), nothing on repeat / rewind, and tracks
  multiple lines independently,
- the payload body is runtime-neutral at the top level and pushes
  DeepStream vocabulary under ``source`` / ``debug``,
- optional fields (``track_id`` / ``class_*`` / ``direction``) are
  omitted when absent (no ``null`` noise),
- ``RoiOccupancyState`` emits on first observation, on change, and
  on interval — but NOT every frame when unchanged,
- malformed analytics blobs (empty dicts, missing class counts)
  return safely without crashing,
- the adapter module does NOT import the writer / API client at
  import time (pure / dependency-free, by design).

### What's deliberately NOT in P4-A
- No call into ``ObservationWriter.emit`` from any worker.
- No widening of ``schemas.OBSERVATION_TYPES``.
- No api-server PR.
- No production POST.
- No dashboard-driven ROI/line geometry, no rules/events, no
  billing surface.

### Next step (P4-B)
1. Open the matching api-server PR adding ``processor.line_crossing``
   to ``ProcessorObservationType``.
2. In lock-step, widen ``schemas.ObservationType`` /
   ``OBSERVATION_TYPES`` in this repo (one tuple element + one
   Literal arm).
3. In ``DeepStreamWorker._drive_metadata_probe`` (analytics mode),
   instantiate ``LineCrossingState(stream_session_id=...)`` once per
   worker and feed each consumed frame's
   ``frame.source.analytics.line_crossing_cumulative`` to
   ``state.absorb_cumulative(...)``. For each returned
   ``LineCrossingEvent``, ``await self._writer.emit(...)``.
4. Add live verification on crowdcountS2: walk across the line,
   confirm one ``processor.line_crossing`` row per crossing in the
   ``observations`` table, with deterministic ``idempotency_key``
   values in ``payload``.
5. Defer ``processor.roi_occupancy`` to P4-C — the adapter state is
   ready, but production observability burden + class-count surface
   need a separate design pass.

## P4-A.5 — Observation Taxonomy + Emission Policy

Locks in **which** observation types UpStream will support, in **what
order**, and under **what emission policy** — before any api-server
widening or worker wiring happens. Lives as a typed table in
``app/observations/observation_taxonomy.py``; ``docs/architecture.md``
links to it but does not duplicate the data. The whole reason this
milestone exists is to prevent observation naming from becoming a
one-off as more types are added.

### Three emission-policy categories
Every ``processor.*`` type maps to exactly one of:

- **event** — one-shot, fires on a discrete signal (line crossed,
  zone entered, dwell threshold exceeded). Idempotency key encodes
  WHICH event so two crossings produce two distinct keys.
  No sampling, no rate-limiting — a missed event is a correctness
  bug. ``LineCrossingState`` is the canonical implementation.
- **state** — snapshot of a continuous quantity (count of objects
  in an ROI, queue length, per-track lifecycle phase). Emit on
  change OR after ``min_interval_seconds``. **Never** every frame.
  ``RoiOccupancyState`` is the canonical implementation.
- **telemetry** — high-volume diagnostic signal (per-track updates,
  per-frame summaries, scene quality warnings). Always sampled.
  Default off in production; opt-in via a config flag.

The policy class is encoded as a ``Literal["event", "state",
"telemetry"]`` on each ``ObservationSpec`` row so a careless future
edit can't add an event without thinking about idempotency.

### Initial type table
Twelve types, all under the ``processor.*`` namespace
(customer-facing names like ``person.crossed_line`` are a later
adapter on top of these — out of scope here). Verbatim from
``observation_taxonomy.py``:

| Type | Category | Status | Notes |
| --- | --- | --- | --- |
| ``processor.line_crossing`` | event | **proposed_first_api_add** | P3.5-proven counter delta; no tracker needed |
| ``processor.zone_entered`` | event | deferred | needs per-(track, zone) state |
| ``processor.zone_exited`` | event | deferred | depends on ``zone_entered`` |
| ``processor.dwell_exceeded`` | event | deferred | depends on both zone events |
| ``processor.direction_detected`` | event | deferred | needs tracker + debounce |
| ``processor.roi_occupancy`` | state | deferred | adapter state ready; volume policy needs live measurement |
| ``processor.zone_occupancy`` | state | deferred | needs dashboard-driven geometry |
| ``processor.queue_length`` | state | deferred | needs dashboard-driven geometry |
| ``processor.track_state`` | state | deferred | per-track lifecycle |
| ``processor.track_update`` | telemetry | deferred | per-track snapshots; sampled / debug-only |
| ``processor.detection_seen`` | telemetry | deferred | per-class bucketed counts |
| ``processor.frame_summary`` | telemetry | deferred | per-frame counter snapshot |
| ``processor.scene_quality_warning`` | telemetry | deferred | probe-detected health signal |

### First API addition (explicit decision)
**Only ``processor.line_crossing`` widens the api-server
``ProcessorObservationType`` ``Literal`` in P4-B.** Rationale:

- P3.5 already proved the source signal works
  (``objLCCumCnt`` ramped 0 → 1 → 2 on two live operator crossings).
- Event category, so dedupe is straightforward (one event per
  cumulative-count increment; no sampling decisions to make in flight).
- Lowest volume of every proposed type — proves the write path
  without flooding the ``observations`` table.
- Does not require dashboard-driven geometry — P3.5's hand-authored
  line is sufficient.

ROI occupancy is **second**, in P4-C, after we have live measurements
of how often the counter actually changes on a real camera (the
per-frame ``objInROIcnt`` is updated continuously; emit volume
depends on the scene). Zone events, dwell, and direction all need
adapter state we haven't built; they sit in P4-C+. Telemetry types
are opt-in and not on the critical path.

### Payload + idempotency policy (one set of rules)
Every type follows the same rules (encoded as taxonomy invariants
in ``tests/test_observation_taxonomy.py``):

1. **Top level is runtime-neutral.** No ``runtime`` / ``package_id``
   / ``analytics_plugin`` / ``tracker_algorithm`` at the top of
   ``payload``; those live under ``source``.
2. **Universal keys.** Every payload carries ``idempotency_key``,
   ``source``, ``debug`` (taxonomy ``payload_keys`` lists ONLY the
   type-specific keys to keep rows skimmable).
3. **Idempotency key shape:**
   ``sess:<session_id>:<type>:<discriminator>``. The discriminator is
   chosen per type so two distinct events of the same type get
   distinct keys:
   - line_crossing → ``<line_id>:<cumulative_count>``
   - zone_entered → ``<zone_id>:<track_id>:<enter_sequence>``
   - zone_exited → ``<zone_id>:<track_id>:<exit_sequence>``
   - dwell_exceeded → ``<zone_id>:<track_id>:<dwell_session_id>:<threshold_seconds>``
   - roi_occupancy → ``<roi_id>:<sequence>``
   - track_state / track_update → ``<track_id>:<sequence|bucket_id>``
   - frame_summary → ``<bucket_id>``
4. **State idempotency keys MUST use a per-emit sequence or bucket
   id, NEVER ``<frame_num>``.** Frame numbers explode under
   continuous emission; sequences only advance when the adapter
   actually emits.
5. **Idempotency key lives in the body** (``payload.idempotency_key``)
   until the api-server adds a first-class column. The pattern is
   deterministic so a future server-side dedupe migration can
   re-derive the key without coordinating with the processor.

### Adapter state requirements (per stream, per worker)
Implemented (P4-A):

- ``LineCrossingState`` — ``last_cumulative_by_line``,
  ``recently_emitted_keys``. Independent state per line. No event
  for unchanged count. No event for counter rewind (watermark held).
- ``RoiOccupancyState`` — ``last_counts_by_roi``,
  ``last_emit_at_by_roi``, ``sequence_by_roi``. Emits on first
  observation, on change, and on ``min_interval_seconds``; never
  every frame.

Designed but not yet implemented (deferred to P4-C+):

- ``ZoneMembershipState`` — per-(track, zone) presence tracking.
  Required for ``zone_entered`` / ``zone_exited``. Source is the
  per-object ``NvDsAnalyticsObjInfo.roiStatus`` list, which P3.5
  already surfaces under ``object.debug.analytics.roi_labels`` but
  the worker does not yet diff between frames.
- ``DwellState`` — ``active_zone_entries_by_tracker``,
  ``emitted_thresholds_by_dwell_session``. Depends on
  ``ZoneMembershipState`` for entry/exit events; cannot ship
  standalone. Idempotency key includes a per-entry
  ``dwell_session_id`` so re-entering the same zone after exit
  starts a fresh threshold count.
- ``TrackLifecycleState`` — per-tracker first-seen / last-seen
  timestamps + emit sequence. For ``processor.track_state`` (state)
  and ``processor.track_update`` (telemetry, sampled).
- ``DirectionDebounceState`` — per-(track, direction-label) last-
  emitted timestamp. Prevents a person walking back-and-forth from
  flooding direction events.

### Deferred types (explicit list)
The taxonomy module's ``DEFERRED_TYPES`` tuple is the authoritative
list. As of P4-A.5 it is: everything except
``processor.line_crossing``. Each row in the taxonomy records why it
is deferred and (where relevant) which other type(s) it depends on,
so a future agent shipping zone events knows ``dwell_exceeded`` is
trivially next.

### What's deliberately NOT in P4-A.5
- No widening of ``schemas.OBSERVATION_TYPES`` (still only the three
  mock types; the taxonomy is allowed to describe types the server
  doesn't accept yet).
- No api-server PR.
- No worker wiring; ``ObservationWriter`` still only flows mock
  observations.
- No live verification.
- No customer-facing event names — those are an adapter on top of
  ``processor.*``, a separate later milestone.

### Coverage
``tests/test_observation_taxonomy.py`` (24 tests, no Docker / GPU /
pyds / api-server). Highlights:
- Taxonomy non-empty, every type unique, every type under
  ``processor.*``, every category / status value valid.
- **Exactly one type** is marked ``proposed_first_api_add`` (the
  load-bearing P4-A.5 decision) and that type is
  ``processor.line_crossing``.
- ``proposed_first_api_add`` and ``api_supported`` are mutually
  exclusive.
- ``api_supported`` rows must also be in
  ``schemas.OBSERVATION_TYPES``; non-``api_supported`` rows must NOT
  be. Keeps the local Literal whitelist and the taxonomy from
  drifting.
- Every ``CANDIDATE_*`` adapter constant has a matching taxonomy
  row.
- Every idempotency key pattern starts with
  ``sess:<session_id>:processor.<thing>:``.
- Every event-style key has a uniqueness token; no state key uses
  ``<frame_num>``.
- Zone-dependent types require zone geometry + tracker_id; dwell
  has explicit ``dependencies`` on zone_entered / zone_exited.
- ``ObservationSpec`` is frozen; ``payload_keys`` does NOT
  redundantly include the universal ``idempotency_key`` / ``source``
  / ``debug`` keys.

### Next step (P4-B)

**P4-B — Line-Crossing Observation Write Path** is the next
milestone:

1. Open the api-server PR adding ``processor.line_crossing`` to
   ``ProcessorObservationType``. No DB migration. Existing tests
   pass with the wider Literal.
2. In lock-step, add ``processor.line_crossing`` to this repo's
   ``schemas.ObservationType`` / ``OBSERVATION_TYPES``. Flip the
   taxonomy row's ``status`` from ``proposed_first_api_add`` to
   ``api_supported``; the taxonomy invariant test will then move
   the type from the deferred list into the supported list.
3. Thread ``ObservationWriter`` into ``DeepStreamWorker.__init__``
   (mirroring ``MockWorker``). Update ``WorkerManager.build_worker``.
4. In ``DeepStreamWorker._drive_metadata_probe`` analytics mode,
   instantiate ``LineCrossingState`` once per worker; feed each
   consumed frame's
   ``frame.source.analytics.line_crossing_cumulative`` through it;
   ``await self._writer.emit(...)`` for each returned event.
5. Live-verify on crowdcountS2: operator crosses the line twice,
   exactly 2 ``processor.line_crossing`` rows appear in the
   ``observations`` table with deterministic ``idempotency_key``
   values in ``payload``, zero rows on a steady scene.
6. ROI occupancy stays deferred (P4-C).
