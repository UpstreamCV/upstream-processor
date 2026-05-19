# deepstream_runtime (M8.2B decode probe + M8.2C1-B inference probe)

The DeepStream runtime. Implements the same `app/runtimes/base.py:Runtime`
interface as `mock_runtime`, so the supervisor/poller/status paths are
unchanged. One `DeepStreamWorker` class, two modes selected by the resolved
package's `supports`:

- `decode_probe` — `deepstream.decode_probe.v0`, `supports.decode_only`
  (M8.2B). The M8.2A-proven NVDEC decode pipeline. **Decode only.**
- `inference_probe` — `deepstream.inference_probe.v0`,
  `supports.inference_probe` (M8.2C1-B). The M8.2C1-A-proven DeepStream 8.0
  `nvinfer` pipeline. Inference runs; detection evidence is status/log-only.

Files:

- `process_manager.py` — `ProcessManager` Protocol + `DeepStreamProcessManager`:
  builds + runs the M8.2A-proven Docker/GStreamer **decode** pipeline as an
  asyncio subprocess, streams stdout, bounded graceful stop +
  SIGKILL-after-timeout. RTSP URL only in the subprocess argv; logs redacted.
- `inference.py` — pure config/command builders (`build_pgie_config`,
  `build_app_config`, `build_inference_docker_command`) +
  `DeepStreamInferenceProcessManager`: owns a per-run **temp work dir**
  (deepstream-app + PGIE config + KITTI output), bind-mounted into the
  DeepStream 8.0 container; runs the M8.2C1-A-proven `deepstream-app` nvinfer
  pipeline (RTSP/TCP → `nvstreammux` → `nvinfer` TrafficCamNet INT8 →
  `fakesink`, no display); exposes `kitti_stats()` (frame/detection counts as
  status evidence); removes the temp dir on stop unless `keep_temp_files`.
  RTSP URL only in the temp `ds_app.txt` + argv; logs redacted.
- `worker.py` — `DeepStreamWorker(Runtime)`: maps both modes onto the M8.0
  status lifecycle `starting → attached → processing → stopped` (or
  `failed`). Decode parses `fpsdisplaysink`; inference parses
  `**PERF` fps + the engine-build / `Unsupported SM` markers from the proven
  M8.2C1-A run and folds KITTI counts into the status payload.

**No observations** in either mode. The inference probe runs `nvinfer` but is
still a probe: **no pyds, no pad probes, no MetadataSource, no metadata
adapter, no tracking, no nvtracker, no nvdsanalytics, no Triton.** KITTI
output is debug/inference evidence only — *not* metadata extraction. Default
image is **DeepStream 8.0** (7.1's TensorRT 10.3 fails inference on Blackwell
sm_120; 8.0's TensorRT 10.9 works — see `docs/deepstream-inference-smoke.md`).
The `deepstream` runtime is selected when the resolved package's
`runtime == "deepstream"` (reachable in dev via `UPSTREAM_FORCE_PACKAGE_ID`).
Unit tests inject a fake `ProcessManager`, so no Docker/GPU is needed.
