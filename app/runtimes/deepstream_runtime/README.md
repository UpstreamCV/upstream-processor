# deepstream_runtime (placeholder — no code yet)

Reserved for the **M8.2B** DeepStream runtime worker. It will implement the
same `app/runtimes/base.py:Runtime` interface as `mock_runtime` (so the
supervisor/poller/status/observation paths are unchanged) and will be the
first consumer of an eligible session's `internal_rtsp_url`.

M8.2A (this milestone) only proved the *environment + RTSP read/decode* —
see `docs/deepstream-rtsp-probe.md` and `scripts/probe_deepstream_*.sh`.
There is intentionally **no Python module here yet**: no DeepStream worker,
no inference, no Triton, no observation writing. Do not add code here until
the RTSP decode probe has passed against an active stream and M8.2B starts.
