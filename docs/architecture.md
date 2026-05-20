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

**P4-B is now implemented** — see the **P4-B** section below.

## P4-B — Line-Crossing Observation Write Path

Closes the loop the previous P4 milestones designed: an operator
crossing the configured midline now produces a discrete
``processor.line_crossing`` row in the ``observations`` table via
the existing ``ObservationWriter`` / ``POST /v1/internal/processor/
observations`` path. Coordinated across two repos
(``upstream-api-server`` widens the type whitelist; this repo wires
the write path) and event-only — ROI occupancy, zone events, dwell,
direction all remain deferred per the P4-A.5 taxonomy.

### Lockstep schema widening (no DB migration)
- **upstream-api-server**:
  ``src/upstream_api/schemas/processor.py`` —
  ``ProcessorObservationType`` Literal gains
  ``"processor.line_crossing"``. ``observations.type`` is plain
  ``text`` (verified at migration
  ``20260518140000_processor_status_observations.sql``); no DDL is
  needed. ``tests/test_processor.py`` gains
  ``test_observation_line_crossing_accepted`` (pinned positive
  case); the existing ``test_observation_invalid_type_rejected``
  still uses ``processor.real_detection`` for the rejection path.
- **upstream-processor**:
  ``app/observations/schemas.py`` — ``ObservationType`` Literal +
  ``OBSERVATION_TYPES`` tuple gain the same string. The taxonomy
  row's ``status`` flips from ``proposed_first_api_add`` to
  ``api_supported``; the P4-A.5 invariant tests now assert the
  post-P4-B state (``test_at_most_one_type_proposed_for_first_api_add``,
  ``test_line_crossing_is_now_api_supported``).

The P4-A "candidate not in whitelist" regression for
``processor.line_crossing`` is replaced by
``test_line_crossing_now_in_server_whitelist``; the matching guard
for ``processor.roi_occupancy`` survives unchanged
(``test_roi_occupancy_still_not_in_server_whitelist``) so a future
careless edit cannot silently widen the wrong type.

### Writer integration in the DeepStream worker
- ``DeepStreamWorker.__init__`` accepts an optional
  ``writer: ObservationWriter | None``. Stored as ``self._writer``;
  only used by the analytics mode driver.
- ``WorkerManager.build_worker`` passes the same writer it gives
  ``MockWorker``. No new instance is created — single chokepoint
  preserved.
- Non-analytics modes (decode / inference / metadata / tracker)
  ignore the writer. Tests pin both that they ignore it and that
  the optional ``writer=None`` path is safe (a misconfigured
  supervisor can't crash the worker).

### Adapter integration in the analytics consumer
Inside ``_drive_metadata_probe``, when
``self._mode == _ANALYTICS_MODE and self._writer is not None``:
1. A ``LineCrossingState(stream_session_id=self._session.session_id)``
   is constructed once.
2. A ``deepstream_source_block(package_id=...,
   tracker_algorithm=...)`` dict is built once.
3. In the consumer loop, immediately after
   ``counters.absorb(frame)``, the worker calls
   ``await self._emit_line_crossings(frame, line_state,
   source_block, counters)``. The helper extracts
   ``frame.source["analytics"]["line_crossing_cumulative"]``
   (skipping cleanly if absent / non-dict), validates the value
   shape, and hands the dict to ``state.absorb_cumulative(...)``
   along with ``occurred_at = frame.frame.timestamp or now_utc()``.
4. For each returned ``LineCrossingEvent`` (one per integer
   increment), the helper calls
   ``await self._writer.emit(stream_session_id=...,
   obs_type="processor.line_crossing",
   payload=event.to_payload(), package_id=...,
   observed_at=event.occurred_at)``.

### Per-emit error handling (mirrors ``_post`` shape)
- ``ApiSessionGoneError`` propagates — outer ``run()`` ends the
  worker quietly, same as the rest of the worker.
- ``ApiValidationError`` is logged and counted
  (``counters.observation_write_errors``) but does NOT crash the
  consumer. Wire-contract drift between the processor and
  api-server is unrecoverable in flight; the next worker spawn
  uses a fresh schema. **Important property**: the writer's
  defensive ``OBSERVATION_TYPES`` guard prevents an accidentally-
  un-widened processor from emitting the new type (would raise
  ``ProcessorError``, also caught and counted).
- Anything in ``TRANSIENT_API_ERRORS`` (connection /
  rate-limit / contract / server) is logged + counted; the next
  event tries again. No bounded retry / backoff in P4-B —
  intentionally — because each integer increment already gets a
  unique deterministic ``idempotency_key`` so a duplicate POST
  (after a hypothetical client-side retry) would be safe once the
  server gains a UNIQUE constraint. For now duplicates would
  manifest as duplicate rows; P4-C can add that constraint if the
  data shows it's needed.

### Status-payload counters
``_MetadataCounters`` gains:
- ``observations_emitted: int``
- ``line_crossing_observations_emitted: int``
- ``observation_write_errors: int``
- ``last_observation_at: datetime | None``

``_metadata_payload`` surfaces them ONLY when the worker actually
has a writer (analytics + writer). Otherwise the payload carries
``"observations_enabled": False`` and omits the counters, keeping
the tracker / metadata payloads tight. Operators can tell from one
``processing`` post whether observations are flowing:

```
"observations_enabled": true,
"observations_emitted_count": 2,
"line_crossing_observations_emitted": 2,
"observation_write_errors_count": 0,
"last_observation_at": "2026-05-20T16:42:31.512+00:00"
```

### Container boundary unchanged
The pyds container still does not call the UpStream API. The
webhook secret stays in the outer worker process. ``ObservationWriter``
runs in the asyncio consumer loop on the host; the container's
only job is to walk ``NvDsAnalyticsFrameMeta`` and write JSON to
stdout. A flaky API only stalls the consumer, never the GST
pipeline.

### Coverage
- ``tests/test_deepstream_line_crossing_emit.py`` (15 new tests).
  Highlights: ``WorkerManager.build_worker`` threads the writer
  through; the lifecycle emits one observation on 0→1 and two on
  0→1→2 with deterministic idempotency keys; repeated cumulative
  emits nothing; counter rewind emits nothing while the
  watermark holds and a later genuine increment emits one event;
  payload top-level is runtime-neutral with DeepStream vocabulary
  under ``source`` / ``debug``; no ``processor.roi_occupancy``
  writes; tracker- and metadata-only modes don't emit; a worker
  with ``writer=None`` is silent + surfaces
  ``observations_enabled=False``; a transient
  ``ApiServerError`` ends in ``stopped`` with the error count
  surfaced + no stream-key leakage in any status payload; emit
  call surface (obs_type / package_id / processor_id /
  stream_session_id / observed_at) matches the wire contract.
- ``tests/test_observation_adapter_contract.py`` —
  ``test_candidate_types_not_in_server_whitelist_yet`` is replaced
  by ``test_line_crossing_now_in_server_whitelist`` +
  ``test_roi_occupancy_still_not_in_server_whitelist`` so the
  half-and-half post-P4-B state is what's asserted.
- ``tests/test_observation_taxonomy.py`` —
  ``test_exactly_one_type_proposed_for_first_api_add`` relaxed to
  ``test_at_most_one_type_proposed_for_first_api_add``;
  ``test_first_api_add_is_line_crossing`` replaced by
  ``test_line_crossing_is_now_api_supported``. The
  ``test_no_processor_type_is_already_in_observation_types``
  lock-step assertion did most of the work — it already enforced
  that ``api_supported`` rows and ``OBSERVATION_TYPES`` widen
  together.

### Live verification
See ``docs/local-dev.md`` for the operator checklist.

### What's deliberately NOT in P4-B
- ``processor.roi_occupancy`` (P4-C).
- ``processor.zone_entered`` / ``zone_exited`` / ``dwell_exceeded``
  (need ``ZoneMembershipState`` — design lives in
  ``observation_taxonomy.py`` but no adapter implementation yet).
- Server-side ``idempotency_key`` column + UNIQUE constraint.
  Only needed if duplicate rows actually appear; client-side
  state dedupes within a container lifetime.
- Dashboard-driven line geometry — P4-B uses P3.5's hand-authored
  central horizontal line.
- Direction / track-update / telemetry types.

## P4-C — ROI Occupancy Observation Write Path

Closes P4-B's deferred item: ``processor.roi_occupancy`` is now a
real write path through the same ``ObservationWriter`` chokepoint.
ROI counts arrive at frame rate (~30 fps), so the emit policy is
**first-seen + on-change + interval heartbeat**; never per-frame.
The line-crossing path from P4-B is unchanged and runs in parallel.

### Lockstep schema widening (no DB migration)
- **upstream-api-server** —
  ``src/upstream_api/schemas/processor.py``:
  ``ProcessorObservationType`` Literal gains
  ``"processor.roi_occupancy"``. ``tests/test_processor.py`` gains
  ``test_observation_roi_occupancy_accepted``; the line-crossing +
  unknown-type tests are unchanged.
- **upstream-processor**:
  ``app/observations/schemas.py`` — ``ObservationType`` Literal +
  ``OBSERVATION_TYPES`` tuple gain the same string.
  ``app/observations/observation_taxonomy.py`` — flips
  ``processor.roi_occupancy``'s row from ``deferred`` to
  ``api_supported`` (parallel to the P4-B flip).

The post-P4-B ``test_roi_occupancy_still_not_in_server_whitelist``
guard is replaced by ``test_roi_occupancy_now_in_server_whitelist``
+ a new ``test_zone_dwell_track_types_still_unsupported`` that
iterates the deferred-future-types — same guard pattern, re-aimed
at the next set of types.

### Adapter extension — emit reason rides under ``debug``
``RoiOccupancyState._should_emit`` (which returned ``bool``) becomes
``_classify_emit`` returning ``"first" | "change" | "interval" |
None``. ``absorb_roi_counts`` merges ``{"emit_reason": <reason>}``
into the event's ``debug`` dict so operators can tell a state
change from a heartbeat at a glance.

``RoiOccupancyEvent`` itself is unchanged — the reason rides under
``debug["emit_reason"]``, not at the top level of the payload
(same pattern as ``frame_num``, ``raw_roi_counts``, etc.). Existing
P4-A unit tests pick up the new key directly:
``test_roi_first_observation_always_emits`` →
``debug.emit_reason == "first"``, ``test_roi_emits_on_change`` →
``"change"``, ``test_roi_emits_on_interval_even_when_unchanged``
→ ``"interval"``. Plus two new adapter tests
(``test_roi_emit_reason_lives_in_debug_dict``,
``test_roi_emit_reason_does_not_overwrite_caller_debug``).

### Settings knob — interval is env-overridable
``app/config.py`` adds
``deepstream_roi_min_interval_seconds: float = 30.0``. Tests
override via ``fast_settings.model_copy(...)``; production
operators raise it for less noise or lower it for more frequent
heartbeats. Setting to ``0.0`` makes every unchanged frame emit
(test-only path).

### Worker integration — parallel to ``_emit_line_crossings``
In ``_drive_metadata_probe``, when
``self._mode == _ANALYTICS_MODE and self._writer is not None``:

1. Build ``roi_state = RoiOccupancyState(...)`` next to the
   existing ``line_state = LineCrossingState(...)``.
2. In the consumer loop, immediately after the line-crossing
   helper call, run a parallel
   ``await self._emit_roi_occupancies(...)``.

The new ``_emit_roi_occupancies`` method mirrors
``_emit_line_crossings`` exactly: pulls
``frame.source["analytics"]["roi_counts"]`` (skipping if missing
/ non-dict / empty), defensively coerces to ``dict[str, int]``,
computes ``occurred_at = frame.frame.timestamp or now_utc()``,
hands to ``roi_state.absorb_roi_counts(...)``, emits one
``processor.roi_occupancy`` observation per returned event. Same
error-handling shape: ``ApiSessionGoneError`` propagates;
``ApiValidationError`` / ``TRANSIENT_API_ERRORS`` are logged +
counted + continue.

### Status-payload counters
``_MetadataCounters`` gains
``roi_occupancy_observations_emitted: int = 0``. The shared
``observation_write_errors`` counter is bumped on any emit error
(line OR ROI); the writer error path doesn't distinguish.

``_metadata_payload`` surfaces the new counter alongside
``line_crossing_observations_emitted``:

```
"observations_enabled": true,
"observations_emitted_count": 6,
"line_crossing_observations_emitted": 2,
"roi_occupancy_observations_emitted": 4,
"observation_write_errors_count": 0,
"last_observation_at": "2026-05-20T16:42:31.512+00:00"
```

### Per-class breakdown is still empty in v0
``RoiOccupancyState.absorb_roi_counts`` accepts an optional
``class_counts_by_roi`` arg; the worker does NOT yet build this
dict (P3.5's pyds probe only surfaces per-ROI totals via
``NvDsAnalyticsFrameMeta.objInROIcnt``). Surfacing per-class
breakdowns would require walking each ``NvDsObjectMeta``'s
``roiStatus`` list inside the container and aggregating — a
probe-side change held for a later milestone. Today
``payload["class_counts"]`` is always ``{}`` for production ROI
observations.

### Coverage
- ``tests/test_deepstream_roi_occupancy_emit.py`` (18 new tests).
  Highlights: first-emit + on-change + interval; no-emit on
  unchanged within interval; 50-frame steady-state produces exactly
  1 emit; idempotency key uses ``sequence`` not ``frame_num``;
  payload top-level is runtime-neutral (``debug`` carries
  ``emit_reason`` + raw counters + frame_num); ``tracker_probe`` /
  ``metadata_probe`` modes emit zero ROI; ``writer=None`` is
  silent; transient ``ApiServerError`` raises the error counter
  without crashing or leaking the stream key; combined line + ROI
  in the same frame produces both observation types; status
  payload counters track the two types independently.
- ``tests/test_observation_adapter_contract.py`` — 6 existing ROI
  tests gain ``emit_reason`` assertions; 2 new tests pin the
  ``debug.emit_reason`` location + non-overwrite of caller debug.
- ``tests/test_observation_taxonomy.py`` — new
  ``test_roi_occupancy_is_now_api_supported``.
- ``tests/test_deepstream_line_crossing_emit.py`` — P4-B's
  ``test_roi_occupancy_still_unsupported_by_writer`` is flipped;
  ``test_no_roi_occupancy_writes_in_p4b`` is renamed to
  ``test_line_crossing_unchanged_by_p4c`` and asserts the
  line-crossing emit count is unaffected by the new ROI path.

### Live verification
See ``docs/local-dev.md`` for the operator checklist.

### What's deliberately NOT in P4-C
- ``processor.zone_entered`` / ``zone_exited`` / ``dwell_exceeded``
  (need a new ``ZoneMembershipState`` adapter).
- Per-class ROI breakdown (probe-side change).
- Server-side ``idempotency_key`` column + UNIQUE constraint.
- Dashboard-driven ROI geometry — still hand-authored in
  ``build_analytics_config``.
- Direction / track-update / telemetry types.

## P4-C Cleanup — ROI Debounce / Hysteresis

### Why
P4-C's live verification produced **205 observation POSTs in a
single 180s cycle** — far higher than the design anticipated. Root
cause: TrafficCamNet's detection count is jittery on edge-case
frames (the same person at the edge of the frame might be detected
on frame ``N`` and missed on frame ``N+1``), so a `0 ↔ 1 ↔ 0`
flap pattern fires three legitimate "change" emits in a single
second. The signal is real (the count IS changing); the
observability cost is what's wrong.

### Policy
A per-ROI debounce / hysteresis layer in ``RoiOccupancyState``:

- **First emit** unchanged — first observation of any ROI still
  fires immediately so consumers know the ROI exists.
- **Change emit** gated on **stability**: a new count must hold
  for ``change_stability_frames`` consecutive frames before it
  commits and emits. Default is **5** frames (≈166 ms at 30 fps).
  A flap that returns to the last-emitted count before the
  threshold is met clears the pending candidate; a different new
  candidate replaces the pending one and restarts the counter at
  1.
- **Interval heartbeat** unchanged in cadence (30 s default) AND
  unchanged in invariant: when the heartbeat fires while a
  candidate is pending but not yet stable, **the heartbeat reports
  the last STABLE count, not the unstable candidate**. The
  pending state survives the heartbeat; consumers cannot mistake a
  heartbeat for a real state change.

### Configuration
``app/config.py`` adds
``deepstream_roi_change_stability_frames: int = 5`` (env-overridable
via ``UPSTREAM_DEEPSTREAM_ROI_CHANGE_STABILITY_FRAMES``).
``RoiOccupancyState.change_stability_frames`` defaults to ``1`` in
the dataclass to keep P4-A / P4-C unit-test invariants intact;
``Settings`` carries the production value and the worker passes it
through. ``fast_settings`` overrides to ``1`` so the existing P4-C
emit tests stay valid.

Values < 1 are rejected at adapter construction (``ValueError``).
Setting to 1 reproduces the P4-C immediate-change behavior.

### Worker integration
``_drive_metadata_probe`` passes
``self._settings.deepstream_roi_change_stability_frames`` into
``RoiOccupancyState``. The status payload surfaces
``roi_change_stability_frames`` so an operator can tell "no
observations because steady scene" apart from "no observations
because debounce too aggressive". Line-crossing behavior is
completely unchanged.

### State machine
Per ROI:
- ``last_counts_by_roi[roi]`` — last EMITTED count snapshot.
- ``last_emit_at_by_roi[roi]`` — last emit time.
- ``sequence_by_roi[roi]`` — per-ROI emit counter (idempotency
  key sequence; advances only on actual emits).
- ``pending_count_by_roi[roi]`` — candidate count not yet emitted
  (None / missing when no candidate).
- ``pending_seen_frames_by_roi[roi]`` — how many consecutive
  frames the candidate has been observed.

On each frame:
1. If ROI is new → emit ``first`` with the count. Done.
2. If count == last_emitted → clear pending; maybe emit
   ``interval`` with that count.
3. If count != last_emitted:
   - Update pending (same candidate → seen += 1; different
     candidate → reset to {count, 1}).
   - If seen >= threshold → emit ``change`` with the new count;
     clear pending.
   - Else if interval elapsed → emit ``interval`` with the LAST
     STABLE count (pending survives).

### Tests
- ``tests/test_observation_adapter_contract.py`` — 9 new tests.
  Threshold validation; 1-equals-immediate-change; debounce=3
  suppresses single-frame change; debounce=3 commits on third
  consecutive frame; flap clears pending; different new candidate
  replaces pending; heartbeat reports stable count during
  pending; pending eventually commits after heartbeat; sequence
  increments only on actual emits.
- ``tests/test_deepstream_roi_occupancy_emit.py`` — 6 new tests:
  worker-level flap suppression; stable-count commit; heartbeat
  emits stable count under debounce; Settings default is 5;
  invalid threshold rejected; status payload exposes the
  threshold.
- ``tests/conftest.py`` — ``fast_settings`` adds
  ``deepstream_roi_change_stability_frames=1`` so existing P4-C
  emit tests keep working without per-test overrides.

### What's NOT in this cleanup
- No new observation types.
- No API changes.
- No DB migration.
- No worker-side hysteresis for the count returning to a previous
  value (the count must explicitly hit the last emitted value to
  clear pending, not an arbitrary back-off curve).
- No emit-reason expansion (still only ``first`` / ``change`` /
  ``interval``).
- No retry / backoff loop around writer.emit.
- ``processor.line_crossing`` behavior unchanged.

## P4-D — Zone Entered / Zone Exited Observations

Combines the two raw signals from P3 + P3.5 — stable ``tracker_id``
and per-object ``debug.analytics.roi_labels`` — into per-track zone
transition events. When the same tracked object is stably inside a
zone for N frames, emit ``processor.zone_entered``. When the same
tracked object is observed stably outside the zone for N frames,
emit ``processor.zone_exited``.

### Lockstep schema widening (no DB migration)
- **upstream-api-server**:
  ``src/upstream_api/schemas/processor.py`` — ``ProcessorObservationType``
  Literal gains ``"processor.zone_entered"`` and
  ``"processor.zone_exited"``. ``tests/test_processor.py`` adds
  two pinned positive tests + ``test_future_observation_types_still_rejected``
  (regression guard against accidentally widening dwell / direction /
  track / telemetry types).
- **upstream-processor**:
  ``app/observations/schemas.py`` adds the same two strings to
  ``ObservationType`` Literal + ``OBSERVATION_TYPES`` tuple.
  ``app/observations/observation_taxonomy.py`` flips both rows from
  ``deferred`` to ``api_supported``.

The post-P4-C ``test_zone_dwell_track_types_still_unsupported`` guard
becomes ``test_dwell_track_telemetry_types_still_unsupported``: the
two zone types move out of the assertion (now supported), the dwell
+ direction + track + telemetry types stay in it.

### Adapter design — ZoneMembershipState
**File:** ``app/observations/observation_adapter.py``.

New constants ``CANDIDATE_ZONE_ENTERED_TYPE`` /
``CANDIDATE_ZONE_EXITED_TYPE`` join the existing CANDIDATE tuple.
``ZoneTransitionEvent`` is a new frozen dataclass with top-level
runtime-neutral fields (``zone_id`` / ``zone_label`` / ``track_id``
/ ``sequence`` / ``idempotency_key`` + optional ``class_id`` /
``class_label``). DeepStream provenance lives under ``source``;
``debug`` carries ``transition`` (``outside_to_inside`` /
``inside_to_outside``), ``stability_frames_required``, ``frame_num``,
``geometry_source``, and the raw ``roi_labels`` list.

**State machine** (per ``(track_id, zone_id)``):

- ``stable_membership[(t, z)]`` — committed state.
  ``None`` (never seen inside) / ``True`` (stably inside) /
  ``False`` (stably outside, only reachable after a prior True).
- ``pending_membership[(t, z)]`` — candidate value being
  accumulated.
- ``pending_seen_frames[(t, z)]`` — consecutive observed-frames count
  for the candidate.
- ``sequence_by_track_zone_transition[(t, z, type)]`` — per-(track,
  zone, transition-type) emit counter. Re-enter after exit gives
  ``zone_entered`` sequence=2 while ``zone_exited`` is independently
  at sequence=1.

**Per-frame logic** in ``absorb_object(track_id, zone_set,
roi_labels, occurred_at, ...)``:

1. For each zone in ``zone_set``:
   - ``observed = zone in roi_labels``.
   - ``committed = stable_membership.get((track_id, zone))``.
   - **Case A** (``committed is None and observed is False``):
     no-op. We don't track "outside" for tracks we've never seen
     inside this zone.
   - **Case B** (``observed == committed``): clear pending.
   - **Case C** (``observed != committed``): update pending
     (matching → seen += 1; different → reset to seen=1). If
     threshold met, commit + emit. Emit reason is encoded in
     ``debug["transition"]`` as ``"outside_to_inside"`` (→ emit
     ``zone_entered``) or ``"inside_to_outside"`` (→ emit
     ``zone_exited``).

**Crucial invariant: object disappearance does NOT emit
``zone_exited``.** A tracker drop-frame is indistinguishable from a
real exit in this milestone; we only act on observed frames. Pending
state survives an arbitrary number of missing frames. This is the
brief's explicit "do not infer exit from total disappearance"
guidance.

Default ``membership_stability_frames`` is ``1`` in the dataclass
(legacy / backward-compat for direct unit-test callers) and ``5``
in production via
``Settings.deepstream_zone_membership_stability_frames``
(env-overridable via
``UPSTREAM_DEEPSTREAM_ZONE_MEMBERSHIP_STABILITY_FRAMES``). Values
``< 1`` are rejected at ``__post_init__``.

### Worker integration
``DeepStreamWorker._drive_metadata_probe`` instantiates ``zone_state``
alongside the existing ``line_state`` / ``roi_state`` when in
analytics mode with a writer. The consumer loop calls a new
``_emit_zone_transitions`` helper directly after
``_emit_roi_occupancies``. The new helper:

- skips frames whose ``source.analytics.roi_counts`` is missing or
  empty (no known zones for this frame),
- iterates ``frame.objects`` and skips any object without a
  ``tracker_id`` or without a ``debug.analytics.roi_labels`` list,
- coerces ``roi_labels`` to ``set[str]`` defensively,
- hands ``track_id`` / ``zone_set`` (= keys of ``roi_counts``) /
  ``roi_labels`` to ``zone_state.absorb_object(...)``,
- for each returned event, emits via the existing
  ``ObservationWriter`` with the same try/except shape as
  ``_emit_line_crossings`` / ``_emit_roi_occupancies``.

### Status payload extension
The analytics-mode payload (when a writer is wired) adds:
``zone_transition_observations_enabled``,
``zone_entered_observations_emitted``,
``zone_exited_observations_emitted``,
``zone_membership_stability_frames``. The shared
``observation_write_errors_count`` covers writer errors for all
three observation types.

### Coverage
- ``tests/test_zone_membership_state.py`` — **19 new tests** for
  the pure adapter: threshold validation; default-1; first-observed-
  outside is no-op; first stable inside emits after N frames;
  stability=1 is immediate; repeated inside doesn't duplicate;
  stable outside emits exited; single-frame outside flap does not
  emit false exit; inside-outside-inside short flap is safe;
  multi-track and multi-zone independence; re-enter increments
  per-type sequence (1, 2 for enter; 1, 2 for exit); disappearance
  does not emit; class_id / class_label preservation; debug shape
  + transition string; idempotency key builders match event keys;
  pending resets on return-to-committed.
- ``tests/test_deepstream_zone_transition_emit.py`` — **17 new
  tests** for the worker integration: schema lock-step; stable
  inside emits zone_entered; stable outside emits zone_exited;
  short ROI label flap doesn't emit false exit; tracker_id None /
  missing roi_labels / non-list roi_labels / empty roi_counts all
  skip safely; ``writer=None`` is silent; tracker_probe and
  metadata_probe modes don't emit; transient ApiServerError raises
  shared error counter without crashing; status payload exposes
  threshold + per-type counters; counters distinguish entered vs
  exited; combined frame emits line + ROI + zone independently;
  re-enter / re-exit increments the per-type sequence.
- ``tests/test_observation_adapter_contract.py`` — flips the P4-C
  zone-deferred guard to two now-supported asserts plus a
  re-aimed guard at dwell / direction / track / telemetry.
- ``tests/test_observation_taxonomy.py`` — two new
  ``test_zone_*_is_now_api_supported`` invariants.
- ``tests/conftest.py`` — ``fast_settings`` adds
  ``deepstream_zone_membership_stability_frames=1`` for same-shape
  test brevity as the P4-C cleanup pattern.

### Accuracy caveat (operator-facing)
Live results will not be perfectly accurate. The TrafficCamNet probe
misses frames; nvtracker drops / reassigns IDs; nvdsanalytics
labels flap. **P4-D is judged by adapter correctness given the
metadata it sees, not by whether the camera scene is understood
correctly.** Live verification is a smoke test (no schema failures,
no observation flood, no crashes, no leaks). Unit tests are the
primary correctness gate.

When live results look messy, classify the cause before flagging
P4-D as broken:

1. **Detection issue** — TrafficCamNet missed or falsely detected.
2. **Tracker issue** — NvTracker changed IDs or lost a track.
3. **Analytics metadata issue** — nvdsanalytics ROI labels don't
   match the expected geometry.
4. **Adapter issue** — given stable metadata, ZoneMembershipState
   produced a wrong observation.

Only #4 is a P4-D regression.

### Out of scope (defer to P4-E+)
- Dwell observations
  (``processor.dwell_started`` / ``dwell_updated`` / ``dwell_exceeded``).
  ``ZoneMembershipState``'s ``stable_membership`` map is the
  foundation a future ``DwellState`` builds on (timer per
  ``(track, zone, dwell_session)``).
- Object-disappearance-emits-exit policy.
- Per-zone configurable stability thresholds (one global setting
  for now).
- Server-side ``idempotency_key`` column + UNIQUE constraint.
- Dashboard-driven ROI / line geometry.
- Direction / track-update / telemetry types.
- Class-id-conditional zone counting (P4-D treats every tracked
  object as eligible regardless of class).

## P4-D Follow-up — ROI Label / Analytics Class Filter Fix

### Why
P4-D's live verification produced **zero zone events** — not because
the adapter was broken, but because the upstream metadata pipeline
never attached ``roi_labels`` to any tracked object. The triage
classified this as a detection / analytics issue at the time, but a
second look at the live evidence pinpointed the real root cause:
**ROI geometry, not class filtering.**

Live evidence:
- Operator bbox bottom-center: ``(418, 719)`` on a 1280×720 frame.
- ROI rectangle: ``(rx0, ry0) = (256, 72)`` to ``(rx1, ry1) = (1024, 648)``
  — covering 20%-80% horizontally and 10%-90% vertically.
- nvdsanalytics's per-object reference point for ROI containment is
  the bbox bottom-center. ``y=719`` falls **71 pixels below the
  ROI's y-max of 648**, so every observed frame correctly registered
  the object as outside the ROI; ``roiStatus`` stayed empty and
  ``roi_labels`` was never attached to the per-object meta.
- ``class-id=-1`` was already in the analytics config, so the
  brief's "class filter" hypothesis was wrong; the class filter
  was already wide-open. The geometry was the smoking gun.

### Fix
1. **Widen the probe-default ROI**. New default is **5% margin on
   each side** (5%-95% × 5%-95%, covering 90% × 90% of the frame
   area). Far more forgiving for the camera-above-operator scene
   pattern.
2. **Make the geometry env-configurable**. New setting
   ``deepstream_analytics_roi_margin_pct: float = 0.05``
   (env: ``UPSTREAM_DEEPSTREAM_ANALYTICS_ROI_MARGIN_PCT``). Range
   ``[0.0, 0.5)``; values outside fail at config generation.
3. **Add a class-filter env knob** even though the existing
   ``class-id=-1`` was already correct — this is the production-
   tuning seam the brief requested. New setting
   ``deepstream_analytics_class_ids: str = "all"``
   (env: ``UPSTREAM_DEEPSTREAM_ANALYTICS_CLASS_IDS``). Accepts
   ``"all"`` / ``"-1"`` / ``""`` → ``-1`` (every class) or a single
   integer like ``"2"`` → that class only. Comma-lists are rejected
   with a clear error (nvdsanalytics' single-section ``class-id=``
   takes one int).
4. **Add live-debug diagnostics** to the analytics-mode status
   payload so an operator can triage the next live run without
   guessing:
   - ``analytics_class_ids`` — the configured filter.
   - ``analytics_roi_margin_pct`` — the configured geometry.
   - ``analytics_object_class_id_counts`` — count of objects
     observed per class_id.
   - ``objects_with_roi_labels_count`` — count of objects whose
     per-object ``roiStatus`` list was non-empty. When this stays
     0 while ``analytics_object_class_id_counts`` is non-zero, the
     issue is geometry / class filter — NOT the adapter.
   - ``roi_labels_sample`` — bounded sample of distinct labels
     seen on objects.

### What this follow-up did NOT change
- ``ZoneMembershipState`` (no bug found).
- ``RoiOccupancyState`` / ``LineCrossingState`` (semantics unchanged).
- ``processor.*`` observation Literal (no new types).
- api-server (no schema or route change).
- Pyds metadata app (already extracts ``roiStatus`` correctly).
- Tracker config / model.

### Out of scope (still deferred)
- Dashboard-driven per-camera ROI / line geometry.
- Multi-class filtering in a single nvdsanalytics section
  (needs per-class sections).
- Dwell observations (P4-E).
- Server-side idempotency.
