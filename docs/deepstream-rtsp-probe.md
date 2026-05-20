# P1 / M8.2A — DeepStream environment + RTSP read probe

Goal of M8.2A: prove the GPU host can run DeepStream and that DeepStream can
**read + decode** the internal MediaMTX RTSP stream from a M8.0 eligible
session. No inference, no observations — decode proof only. M8.2B is the
first real DeepStream runtime worker.

## Findings (2026-05-19)

Environment probed on **crowdcountS2** (the GPU host); the `upstream-processor`
repo lives/auths on the Mac. crowdcountS2 has no `uv` and the repo is not
cloned there (private, no creds, not needed — the probe scripts are
standalone and take the RTSP URL as an argument).

| Question | Answer | Evidence |
|---|---|---|
| crowdcountS2 can run NVIDIA/DeepStream? | **Yes** | Ubuntu 24.04.4, 2× NVIDIA GeForce RTX 5070 Ti, driver 580.142 |
| Docker sees the GPU? | **Yes** | `docker run --rm --gpus all nvidia/cuda:12.6.3-base-ubuntu24.04 nvidia-smi` listed both GPUs |
| NVIDIA Container Toolkit installed? | **Yes** | `nvidia-ctk` 1.19.0; docker `Runtimes:` includes `nvidia` |
| Docker usable without sudo? | **Yes** | `docker ps` works as `crowdcounts2` |
| Host GStreamer NVDEC? | No (expected) | `nvv4l2decoder`/`nvcodec` absent on host; they ship inside the DeepStream image. `rtspsrc`/`avdec_h264` present |
| MediaMTX RTSP reachable on crowdcountS2? | **Yes (port open)** | container `upstream-mediamtx` maps host `0.0.0.0:18554 -> 8554/tcp`; `nc 127.0.0.1:18554` open |
| Live RTSP decode proven? | **Pending an active stream** | `GET /eligible-sessions` returned `sessions=[]` at probe time |

### Networking note (important)

M8.0 sets `UPSTREAM_MEDIAMTX_RTSP_BASE=rtsp://127.0.0.1:18554`, so
eligible-session `internal_rtsp_url` = `rtsp://127.0.0.1:18554/<stream_key>`.
On **crowdcountS2** this is **correct**: the UpStream MediaMTX container
publishes RTSP on host `127.0.0.1:18554` there. A DeepStream container must
use `--network host` so its `127.0.0.1` is the host's (the probe script does).
If the processor is ever deployed on a *different* host than the
streaming-server, `UPSTREAM_MEDIAMTX_RTSP_BASE` must be repointed at the
streaming-server's reachable address — that is an api-server env change, not
a code change, and is out of scope until it actually blocks.

(Note: a second, unrelated `mediamtx` on host port 8554 belongs to the
crowdcounts app — the UpStream one is the `upstream-mediamtx` container on
**18554**.)

## DeepStream image recommendation

- **Recommended:** newest available `nvcr.io/nvidia/deepstream` `*-samples` /
  `*-triton` tag — start from `deepstream:7.1-samples-multiarch`; move to
  **8.x** if published.
- **Reason:** the probe only needs NVDEC (`nvv4l2decoder`) + `rtspsrc`, which
  the `-samples` image provides; multiarch covers x86_64 dGPU.
- **Blackwell concern (must heed):** RTX 5070 Ti is **Blackwell (sm_120)**
  with driver 580 / CUDA 13. DeepStream ≤ 7.0 (CUDA ≤ 12.2) will **not** have
  kernels for sm_120. Use a DeepStream build whose CUDA ≥ 12.6 (ideally
  12.8+/13) and TensorRT 10.x — i.e. DeepStream 7.1's latest patch or 8.x.
  Decode (NVDEC) is more tolerant than TensorRT, but pin the newest tag and
  verify with the probe before M8.2B. Override via `DEEPSTREAM_IMAGE=`.
- **Where it runs:** **crowdcountS2** (co-located with MediaMTX → RTSP over
  host loopback, lowest latency, GPUs present). Not the API server (no GPU).
  A dedicated processor host is a later concern.
- The image (~7–9 GB) was **not pulled** during this probe (no active stream
  to test against; instructions say don't pull huge images unnecessarily).
  `probe_deepstream_rtsp.sh` pulls it only with `CONFIRM_PULL=1`.

## How to run (on crowdcountS2, when a stream is active)

```sh
# 1. environment (read-only; pulls only a ~150MB CUDA base for the gpu test)
scripts/probe_deepstream_env.sh

# 2. get an eligible session's internal_rtsp_url from the deployed API
#    (secret stays server-side; never printed/committed) then:
scripts/probe_rtsp_url.sh "rtsp://127.0.0.1:18554/sess_XXXXXXXX"

# 3. GPU NVDEC decode proof (opts into the large image pull)
CONFIRM_PULL=1 scripts/probe_deepstream_rtsp.sh "rtsp://127.0.0.1:18554/sess_XXXXXXXX"
```

All three are read-only, secret-free, repeatable, and take the RTSP URL as an
argument/env (never hard-coded, never committed).

## Status / next step

M8.2A is **complete**: GPU + Docker + Container Toolkit + RTSP reachability
proven, and on an active stream `ffprobe` + the DeepStream/GStreamer
container both opened `internal_rtsp_url` and decoded ~30 fps with 0 dropped
frames (`scripts/probe_deepstream_rtsp.sh`).

**M8.2B is implemented**: that proven manual pipeline now runs as a
supervisor-managed worker — `app/runtimes/deepstream_runtime/` +
`deepstream.decode_probe.v0`. Run it via
`UPSTREAM_FORCE_PACKAGE_ID=deepstream.decode_probe.v0` (see README /
`docs/local-dev.md` M8.2B checklist). Still decode only — no inference,
Triton, tracking, nvdsanalytics, or observations.

**M8.2B cleanup:** live verification showed DeepStream needs a longer
shutdown budget than mock. Defaults are now DeepStream-safe —
`FINAL_STATUS_TIMEOUT_SECONDS=15`, `OVERALL_SHUTDOWN_TIMEOUT_SECONDS=25`,
`DEEPSTREAM_PROCESS_STOP_TIMEOUT_SECONDS=5` — so future inference tests do
not rely on ad hoc env overrides. No behavior change to inference /
observations (not implemented). Env overrides still supported.

**M8.2C1-A inference smoke (2026-05-19):** the Blackwell concern above was
**confirmed real on DeepStream 7.1** (TensorRT 10.3 → `Unsupported SM: 0xc00`
= sm_120, core dump) but is **resolved on DeepStream 8.0**
(`deepstream:8.0-samples-multiarch`, TensorRT 10.9 / CUDA 12.8): nvinfer
builds the engine on the RTX 5070 Ti and runs TrafficCamNet on the live RTSP
stream (~30 fps, detections via KITTI). **M8.2C1-A passed; next M8.2C1-B.**
The inference probe's default image is now 8.0. Full write-up + reusable
probe: `docs/deepstream-inference-smoke.md` /
`scripts/probe_deepstream_inference.sh`.

**M8.2C1-B (2026-05-19):** the M8.2C1-A inference pipeline now runs under the
supervisor as `deepstream.inference_probe.v0` (`DeepStreamWorker` inference
mode + `DeepStreamInferenceProcessManager`). Decode probe (M8.2B) unchanged.
Default DeepStream image bumped to 8.0 for both. Still **no observations / no
metadata extraction / no pyds**. Implementation complete + unit-tested; live
verification pending. See `docs/local-dev.md` M8.2C1-B checklist.

**P2 — manual pyds metadata probe (2026-05-19):** out-of-tree probe
(`scripts/probe_deepstream_pyds_metadata.sh` +
`scripts/deepstream_pyds_metadata_probe.py`, built on
`upstream-deepstream:8.0-pyds-v1.2.2`) walks NvDsBatchMeta/NvDsFrameMeta/
NvDsObjectMeta via the NVIDIA pyds 1.2.2 wheel and emits neutral
intermediate metadata. Proves the metadata seam without touching the
runtime worker or `app/`. Still **no observations, no pyds in `app/`, no
MetadataSource, no metadata adapter** — those land in later milestones.
