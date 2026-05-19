# P1 / M8.2C1-A — DeepStream nvinfer inference smoke probe

Goal of M8.2C1-A: prove whether a DeepStream container can run an **inference**
pipeline (nvinfer + a bundled sample model) on the live internal MediaMTX RTSP
stream from a M8.0 eligible session. Decode was already proven in M8.2A/B.
This is a **manual probe only** — no pyds, no `NvDsBatchMeta`, no metadata
adapter, no observations, no runtime-worker changes.

## Verdict (2026-05-19)

**Inference is blocked by DeepStream/TensorRT/CUDA Blackwell incompatibility.**
nvinfer initialized, but TensorRT in `nvcr.io/nvidia/deepstream:7.1-samples-multiarch`
**cannot build an engine for the RTX 5070 Ti**:

```
ERROR: [TRT]: IBuilder::buildSerializedNetwork: Error Code 1:
        Internal Error (Unsupported SM: 0xc00)
timeout: the monitored command dumped core
```

`SM 0xc00` = compute capability **12.0 = Blackwell sm_120** (RTX 5070 Ti).
The bundled TensorRT has no sm_120 kernels, so the engine build aborts and
deepstream-app core-dumps. This is exactly the Blackwell concern flagged in
`docs/deepstream-rtsp-probe.md` — decode (NVDEC) is forward-compatible and
worked; the TensorRT engine **builder** is not.

| Question | Answer | Evidence |
|---|---|---|
| Active validated stream? | **Yes** | `GET /eligible-sessions?limit=1` → `COUNT=1`, `validation.passed=true`, approved h264 1280x720@30 |
| `internal_rtsp_url` works from crowdcountS2? | **Yes** | `ffprobe` exit 0: `h264 1280x720 yuv420p 30/1` |
| DeepStream container + GPU? | **Yes** | `--gpus all` lists both RTX 5070 Ti; DeepStream **7.1.0** |
| nvinfer initialized? | **Yes** | `gst_nvinfer_logger:<primary_gie>` ran; tried engine deserialize → rebuild |
| TensorRT engine built? | **No** | `Unsupported SM: 0xc00` → core dump; no `.engine` produced |
| Inference ran / detections? | **No** | engine never built; 0 KITTI frames, 0 detections |
| Probe exited cleanly? | Process core-dumped; container `--rm`, no leftovers; temp configs shredded |

## Environment

- **Host:** crowdcountS2 — 2× NVIDIA GeForce RTX 5070 Ti, driver 580.142,
  host CUDA 13.0 (Blackwell sm_120).
- **Image:** `nvcr.io/nvidia/deepstream:7.1-samples-multiarch` (cached, ~20 GB;
  not pulled — already present from M8.2A/B).
- **DeepStream:** 7.1.0 (2024-10-04). **TensorRT: 10.3.0.26** (`+cuda12.5`).
  Container CUDA runtime 12.6.
- **Model:** bundled `Primary_Detector` / TrafficCamNet
  (`resnet18_trafficcamnet_pruned.onnx`, INT8, `cal_trt.bin`). No prebuilt
  `.engine` ships, so first run forces a build — the SM-arch test.
- **Pipeline:** `deepstream-app` config — RTSP/TCP `source0` →
  `nvstreammux` → `nvinfer` (TrafficCamNet) → `fakesink`. No tiled-display,
  no OSD, no tracker (no X11). Detection evidence path =
  `gie-kitti-output-dir` (deepstream-app's own KITTI dump), not pyds.

## Why TensorRT 10.3 fails on Blackwell

Blackwell (sm_120 / compute 12.0) GPU codegen support landed in **TensorRT
10.7–10.8 with CUDA 12.8**. DeepStream 7.1 ships **TensorRT 10.3 / CUDA
12.5–12.6**, which predates Blackwell — the engine *builder* has no sm_120
target, hence `Unsupported SM: 0xc00`. The NVDEC decode path used in
M8.2A/B does not go through TensorRT, which is why decode worked on the same
image while inference does not.

## Recommended next step

Re-run this exact probe against a **Blackwell-capable DeepStream image** —
one whose bundled **TensorRT ≥ 10.7 and CUDA ≥ 12.8** with sm_120 support.
Candidate: NVIDIA **DeepStream 8.x** `*-samples` / `*-triton` multiarch
(`nvcr.io/nvidia/deepstream:8.0-samples-multiarch` or newest published tag —
confirm on NGC). The probe script already supports this:

```sh
# on crowdcountS2, with an active validated stream's internal_rtsp_url:
DEEPSTREAM_IMAGE=nvcr.io/nvidia/deepstream:8.0-samples-multiarch \
CONFIRM_PULL=1 \
scripts/probe_deepstream_inference.sh "rtsp://127.0.0.1:18554/sess_XXXX"
```

Only after this probe **passes** (engine builds on sm_120, frames flow,
detections appear) should M8.2C1-B (supervisor-managed inference worker)
start. The decode-only worker (M8.2B) is unaffected.

## How to run

```sh
# Eligible session + internal_rtsp_url come from the deployed API; the
# processor secret is sourced server-side via SSM/the API host and is
# NEVER printed or committed. Then, on crowdcountS2:
scripts/probe_deepstream_inference.sh "rtsp://127.0.0.1:18554/sess_XXXX"
```

The script takes the RTSP URL as an argument/env, writes the temp
`deepstream-app`/PGIE config only into a `mktemp` dir under `/tmp` (shredded
on exit), redacts the RTSP key in all output, gates the multi-GB image pull
behind `CONFIRM_PULL=1`, and **exits non-zero** on the Blackwell SM block (or
any inference failure). It is read-only w.r.t. UpStream — no observations,
no API writes.

## Scope (unchanged by this probe)

Still **not** implemented and explicitly out of scope: the inference runtime
worker, pyds, pad probes, `NvDsBatchMeta`, MetadataSource / metadata adapter,
observations / `POST /observations`, tracking, line-crossing, zones, Triton,
dashboard, package catalog. This milestone only answers "can nvinfer run on
Blackwell here?" — and the answer on DeepStream 7.1 is **no**.
