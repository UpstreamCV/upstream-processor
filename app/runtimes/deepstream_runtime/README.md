# deepstream_runtime (decode / inference / metadata probes)

The DeepStream runtime. Implements the same `app/runtimes/base.py:Runtime`
interface as `mock_runtime`, so the supervisor / poller / status paths are
unchanged. One `DeepStreamWorker` class, **three modes** selected by the
resolved package's `supports` (most specific wins):

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
  `DeepStreamMetadataProcessManager`: the metadata probe owns a per-run
  temp work dir with a PGIE config + a copy of `pyds_metadata_app.py`,
  bind-mounted into the pyds-derived DS 8.0 image
  (`upstream-deepstream:8.0-pyds-v1.2.2`). Runs the in-container app
  under `python3 -B` with `PYTHONDONTWRITEBYTECODE=1` so no root-owned
  `__pycache__` lands in the host work dir. Cleans the temp dir on stop.
- `pyds_metadata_app.py` — the **in-container** pyds Python app
  (bind-mounted, never imported by other `app/` code). Builds the
  proven pipeline programmatically, attaches a pad probe to
  `nvinfer.src`, walks `pyds.gst_buffer_get_nvds_batch_meta → NvDsFrameMeta
  → NvDsObjectMeta`, and emits structured stdout:
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

**No observations** in any mode. **No `ObservationWriter`** is wired up
anywhere in `app/`. The metadata probe runs `pyds` but is still a probe:
**no metadata adapter, no nvtracker, no nvdsanalytics, no line crossing /
zones, no Triton.** KITTI / `UPSTREAM_METADATA` output is debug evidence
only — *not* final metadata extraction. The webhook secret never enters
the pyds container (the in-container app never calls UpStream APIs;
status posting stays with the outer supervisor). Default images:
**DeepStream 8.0** for decode/inference (`UPSTREAM_DEEPSTREAM_IMAGE`,
`nvcr.io/nvidia/deepstream:8.0-samples-multiarch`), **pyds-derived DS 8.0**
for metadata (`UPSTREAM_DEEPSTREAM_PYDS_IMAGE`,
`upstream-deepstream:8.0-pyds-v1.2.2` built from `docker/Dockerfile.pyds`).
The `deepstream` runtime is selected when the resolved package's
`runtime == "deepstream"` (reachable in dev via
`UPSTREAM_FORCE_PACKAGE_ID`). Unit tests inject a fake `ProcessManager`,
so no Docker / GPU is needed.
