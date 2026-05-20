# deepstream_runtime (decode / inference / metadata / tracker / analytics probes)

The DeepStream runtime. Implements the same `app/runtimes/base.py:Runtime`
interface as `mock_runtime`, so the supervisor / poller / status paths are
unchanged. One `DeepStreamWorker` class, **five modes** selected by the
resolved package's `supports` (most specific wins, in order:
`analytics > tracker > metadata > inference > decode`):

- `analytics_probe` — `deepstream.analytics_probe.v0`,
  `supports.analytics_probe` (**P3.5**). Same pyds pad probe as
  `tracker_probe`, but the pipeline gains an `nvdsanalytics` element
  after `nvtracker`; the pad probe attaches to the analytics src pad
  and additionally walks `frame_user_meta_list` /
  `obj_user_meta_list` for the SDK's `NVDS_FRAME_META_NVDSANALYTICS`
  (→ `NvDsAnalyticsFrameMeta`: `objInROIcnt` / `objLCCumCnt` /
  `objLCCurrCnt` / `ocStatus`) and `NVDS_OBJ_META_NVDSANALYTICS`
  (→ `NvDsAnalyticsObjInfo`: `roiStatus` / `lcStatus` / `dirStatus` /
  `ocStatus`) user-meta. Status payload adds `analytics_enabled`,
  `has_analytics_metadata`, `roi_metadata_count`,
  `line_metadata_count`, `objects_with_roi_metadata`,
  `objects_with_line_metadata`, `analytics_labels_count` and a bounded
  `analytics_labels_sample`. v0 ships a hand-authored ROI +
  line-crossing config (1280×720). Still **no observations**.
- `tracker_probe` — `deepstream.tracker_probe.v0`,
  `supports.tracker_probe` (**P3**). Same pyds pad probe as
  `metadata_probe`, but the pipeline gains an `nvtracker` element
  (default low-level: IOU, no ReID model) between `nvinfer` and
  `nvvideoconvert`; the pad probe attaches to the tracker src pad so
  `NvDsObjectMeta.object_id` is a real stable tracker id across frames.
  Status payload adds `tracker_enabled`, `tracker_algorithm`,
  `unique_tracker_ids_count`, and a bounded `tracker_ids_sample`. Still
  **no observations**.
- `metadata_probe` — `deepstream.metadata_probe.v0`,
  `supports.metadata_probe` (**P2**). Runs the pyds pad probe in a
  DS 8.0 + pyds container, walks NvDsBatchMeta / NvDsFrameMeta /
  NvDsObjectMeta, and surfaces neutral metadata via the
  `BoundedBridgeQueue` / `BridgeQueueMetadataSource` seam into the status
  payload. Status/debug evidence only — **no observations**.
- `inference_probe` — `deepstream.inference_probe.v0`,
  `supports.inference_probe` (M8.2C1-B). The M8.2C1-A-proven DS 8.0
  `nvinfer` pipeline. Inference runs; detection evidence is
  status/log-only (KITTI dump).
- `decode_probe` — `deepstream.decode_probe.v0`, `supports.decode_only`
  (M8.2B). The M8.2A-proven NVDEC decode pipeline. **Decode only.**

Files:

- `process_manager.py` — `ProcessManager` Protocol +
  `DeepStreamProcessManager`: builds + runs the M8.2A-proven
  Docker/GStreamer **decode** pipeline as an asyncio subprocess.
- `inference.py` — pure builders (`build_pgie_config`, `build_app_config`,
  `build_inference_docker_command`) + `DeepStreamInferenceProcessManager`:
  the inference probe owns a per-run temp work dir
  (deepstream-app + PGIE config + KITTI output), bind-mounted into the
  DS 8.0 container; runs the M8.2C1-A-proven `deepstream-app` nvinfer
  pipeline (RTSP/TCP → `nvstreammux` → `nvinfer` TrafficCamNet INT8 →
  `fakesink`, no display); exposes `kitti_stats()` (frame/detection
  counts as status evidence).
- `metadata_probe.py` — pure `build_metadata_docker_command` +
  `DeepStreamMetadataProcessManager`: the metadata / tracker / analytics
  probe owns a per-run temp work dir with a PGIE config + a copy of
  `pyds_metadata_app.py` (plus a generated `tracker_config.yml` when
  tracker is on, plus `analytics_config.txt` when analytics is on),
  bind-mounted into the pyds-derived DS 8.0 image
  (`upstream-deepstream:8.0-pyds-v1.2.2`). Runs the in-container app
  under `python3 -B` with `PYTHONDONTWRITEBYTECODE=1` so no root-owned
  `__pycache__` lands in the host work dir. Cleans the temp dir on stop.
  Container naming distinguishes the modes: `upstream-dsm-*` for
  metadata-only, `upstream-dst-*` for tracker-on (P3),
  `upstream-dsa-*` for analytics-on (P3.5).
- `pyds_metadata_app.py` — the **in-container** pyds Python app
  (bind-mounted, never imported by other `app/` code). Builds the
  proven pipeline programmatically; in tracker mode (CLI
  `--tracker-config-path` set) an `nvtracker` element is spliced
  between `nvinfer` and `nvvideoconvert`; in analytics mode (CLI
  `--analytics-config-path` set; requires tracker) an `nvdsanalytics`
  element is appended after `nvtracker` and the pad probe attaches to
  the analytics src pad instead. The probe additionally walks
  `frame_user_meta_list` / `obj_user_meta_list` for
  `NVDS_FRAME_META_NVDSANALYTICS` /
  `NVDS_OBJ_META_NVDSANALYTICS` user-meta and folds the result under
  `frame.source["analytics"]` (ROI / LC / overcrowding frame counters)
  and `object.debug["analytics"]` (per-object ROI / LC / direction
  labels). Walks `pyds.gst_buffer_get_nvds_batch_meta → NvDsFrameMeta →
  NvDsObjectMeta`, and emits structured stdout:
  - `UPSTREAM_METADATA=<json>` per frame (matches
    `NeutralFrameMetadata`).
  - terminal `UPSTREAM_METADATA_SUMMARY=<json>` with counters
    (`frames_seen`, `frames_with_objects`, `objects_seen`,
    `class_counts`, `label_counts`, `metadata_emitted`,
    `samples_emitted`).
  The pad probe does near-zero work — no HTTP, no async, no
  `ObservationWriter` calls. Mypy/ruff skip this module
  (`[[tool.mypy.overrides]]` in `pyproject.toml`) because gi/pyds only
  resolve at runtime inside the container.
- `worker.py` — `DeepStreamWorker(Runtime)`: maps all three modes onto
  the M8.0 status lifecycle (`starting → attached → processing →
  stopped` / `failed`).
  - Decode parses `fpsdisplaysink`.
  - Inference parses `**PERF` fps + the engine-build / `Unsupported SM`
    markers and folds KITTI counts into the status payload.
  - Metadata runs two asyncio tasks: a reader that parses each stdout
    line via the pure `parse_metadata_line`, validates frame lines with
    `NeutralFrameMetadata`, and pushes via `put_from_thread` into a
    `BoundedBridgeQueue`; and a consumer that iterates a
    `BridgeQueueMetadataSource` and feeds `_MetadataCounters`. The
    status payload carries `frames_seen` / `frames_with_objects` /
    `objects_seen` / `class_counts` / `label_counts` / `has_bbox` /
    `has_confidence` / `has_tracker_id` / `sample_count` / `queue_stats`
    (`accepted` / `dropped` / `dropped_newest` / `dropped_oldest` /
    `consumed` / `current_size` / `closed`) + optional `app_summary`.
    **Shutdown (P2 cleanup hardening):** `request_stop()` synchronously
    closes the bridge queue (waking a parked `to_thread` consumer
    immediately), and the driver's inner `gather` is wrapped in a
    `finally` that `asyncio.shield`'s a `_finalize_metadata_probe` —
    docker stop + workdir cleanup + terminal status post run to
    completion even if the worker task is cancelled.
    `DeepStreamMetadataProcessManager.stop()` and `_cleanup_work_dir()`
    are idempotent.

**P4-B and P4-C widen this:** the `analytics_probe` mode now
emits **two** real DeepStream-backed observation types via the
existing `ObservationWriter`:

- `processor.line_crossing` (P4-B) — one per `objLCCumCnt`
  increment. Event-style; no policy gating needed.
- `processor.roi_occupancy` (P4-C) — emits on first-seen, on
  count change, and on a 30 s heartbeat interval (env-overridable
  via `UPSTREAM_DEEPSTREAM_ROI_MIN_INTERVAL_SECONDS`). **Never
  per-frame.** ``payload["debug"]["emit_reason"]`` records which
  policy branch fired.

The ``decode_probe`` / ``inference_probe`` / ``metadata_probe`` /
``tracker_probe`` modes remain observation-free. The pyds
container still does not call the UpStream API; the adapters +
writer run in the host asyncio consumer loop. Everything else is
still deferred: **no zone enter/exit / dwell / direction
events yet, no rules/events, no dashboard-driven geometry, no
Triton.** KITTI / `UPSTREAM_METADATA` output is debug evidence —
*not* final metadata extraction. The webhook secret never enters the pyds container
(the in-container app never calls UpStream APIs; status posting stays
with the outer supervisor). Default images: **DeepStream 8.0** for
decode/inference (`UPSTREAM_DEEPSTREAM_IMAGE`,
`nvcr.io/nvidia/deepstream:8.0-samples-multiarch`), **pyds-derived DS 8.0**
for metadata/tracker (`UPSTREAM_DEEPSTREAM_PYDS_IMAGE`,
`upstream-deepstream:8.0-pyds-v1.2.2` built from `docker/Dockerfile.pyds`).
The `deepstream` runtime is selected when the resolved package's
`runtime == "deepstream"` (reachable in dev via
`UPSTREAM_FORCE_PACKAGE_ID`). Unit tests inject a fake `ProcessManager`,
so no Docker / GPU is needed.
