# deepstream_runtime (M8.2B — decode probe only)

The DeepStream decode-probe runtime. Implements the same
`app/runtimes/base.py:Runtime` interface as `mock_runtime`, so the
supervisor/poller/status paths are unchanged.

- `process_manager.py` — `ProcessManager` Protocol + `DeepStreamProcessManager`:
  builds + runs the M8.2A-proven Docker/GStreamer decode pipeline as an
  asyncio subprocess, streams stdout, supports bounded graceful stop +
  SIGKILL-after-timeout. The RTSP URL is only in the subprocess argv; logs
  use the redacted form.
- `worker.py` — `DeepStreamWorker(Runtime)`: maps decode onto the M8.0 status
  lifecycle `starting → attached → processing → stopped` (or `failed`),
  parsing `fpsdisplaysink` for `fps_processed`.

**Decode only.** No inference, no Triton, no tracking, no nvdsanalytics, no
metadata adapter, **no observations**. The `deepstream` runtime is selected
when the resolved package's `runtime == "deepstream"` (e.g.
`deepstream.decode_probe.v0`, reachable in dev via
`UPSTREAM_FORCE_PACKAGE_ID`). Unit tests inject a fake `ProcessManager`, so
no Docker/GPU is needed to test the worker.
