# P1 / M8.2C1-A — DeepStream nvinfer inference smoke probe

Goal of M8.2C1-A: prove whether a DeepStream container can run an **inference**
pipeline (nvinfer + a bundled sample model) on the live internal MediaMTX RTSP
stream from a M8.0 eligible session. Decode was already proven in M8.2A/B.
This is a **manual probe only** — no pyds, no `NvDsBatchMeta`, no metadata
adapter, no observations, no runtime-worker changes.

## Verdict — PASSED on DeepStream 8.0 (2026-05-19)

**M8.2C1-A is PASSED.** On `nvcr.io/nvidia/deepstream:8.0-samples-multiarch`
(DeepStream 8.0.0, **TensorRT 10.9.0.34**, **CUDA 12.8.1**) nvinfer built a
TensorRT engine on the RTX 5070 Ti / Blackwell sm_120 and ran inference on
the live RTSP stream — no `Unsupported SM`, no core dump:

```
nvinfer ... buildModel(): serialize cuda engine to file:
  .../Primary_Detector/resnet18_trafficcamnet_pruned.onnx_b1_gpu0_int8.engine successfully
**PERF:  30.00 (29.98) ...                 # ~30 fps sustained
App run successful                          # clean SIGINT (timeout) exit
# KITTI dump: 1571 detections across 1728 frames, e.g.
person 0.0 0 0.0 167.5 0.0 634.8 544.3 ... 0.26
```

Engine build took ~30 s (cold), then ~30 fps inference with TrafficCamNet
detecting `person` objects (the only class present in the scene). The
benign cold-start `deserialize ... failed, try rebuild` line is the normal
no-prebuilt-engine path, **not** a failure. nvinfer persists the built
engine next to the ONNX (in the samples dir), **not** at the configured
`model-engine-file` path — so engine-file presence is not a reliable signal;
the `serialize cuda engine to file: ... successfully` log line is.

The default image in `scripts/probe_deepstream_inference.sh` is now
**DeepStream 8.0**; DEEPSTREAM_IMAGE stays overridable.

### Earlier attempt — DeepStream 7.1 (BLOCKED, root cause)

The first probe on `nvcr.io/nvidia/deepstream:7.1-samples-multiarch`
(TensorRT 10.3 / CUDA 12.6) **could not build an engine for the RTX 5070 Ti**:

```
ERROR: [TRT]: IBuilder::buildSerializedNetwork: Error Code 1:
        Internal Error (Unsupported SM: 0xc00)
timeout: the monitored command dumped core
```

`SM 0xc00` = compute capability **12.0 = Blackwell sm_120** (RTX 5070 Ti).
TensorRT 10.3 has no sm_120 kernels, so the engine build aborted and
deepstream-app core-dumped. Decode (NVDEC) is forward-compatible and worked
on the same image (M8.2A/B); only the TensorRT engine **builder** did not.
DeepStream 8.0's TensorRT 10.9 resolves this.

| Question | DS 7.1 | DS 8.0 | Evidence |
|---|---|---|---|
| Active validated stream? | **Yes** | **Yes** | `GET /eligible-sessions?limit=1` → `COUNT=1`, `validation.passed=true`, approved h264 1280x720@30 |
| `internal_rtsp_url` works from crowdcountS2? | **Yes** | **Yes** | `ffprobe` exit 0: `h264 1280x720 yuv420p 30/1` |
| DeepStream container + GPU? | **Yes** | **Yes** | `--gpus all` lists both RTX 5070 Ti |
| nvinfer initialized? | **Yes** | **Yes** | `gst_nvinfer_logger:<primary_gie>` ran |
| TensorRT engine built on sm_120? | **No** | **Yes** | 7.1: `Unsupported SM: 0xc00` → core dump. 8.0: `serialize cuda engine to file: ...successfully` (~30 s) |
| Inference ran / detections? | **No** | **Yes** | 8.0: 1571 detections / 1728 KITTI frames, ~30 fps; class `person` |
| Probe exited cleanly? | core-dumped | **Yes** | 8.0: SIGINT (timeout) → `App run successful`; container `--rm`, temp shredded |

## Environment

- **Host:** crowdcountS2 — 2× NVIDIA GeForce RTX 5070 Ti, driver 580.142,
  host CUDA 13.0 (Blackwell sm_120).
- **DS 7.1 image (blocked):** `nvcr.io/nvidia/deepstream:7.1-samples-multiarch`
  — DeepStream 7.1.0, **TensorRT 10.3.0.26** (`+cuda12.5`), CUDA rt 12.6.
- **DS 8.0 image (passed):** `nvcr.io/nvidia/deepstream:8.0-samples-multiarch`
  — DeepStream 8.0.0 (2025-08-01), **TensorRT 10.9.0.34** (`+cuda12.8`),
  CUDA rt 12.8.1. ~33 GB; pulled once, now cached on crowdcountS2.
- **Model:** bundled `Primary_Detector` / TrafficCamNet
  (`resnet18_trafficcamnet_pruned.onnx`, INT8, `cal_trt.bin`) — same paths in
  7.1 and 8.0. No prebuilt `.engine` ships, so first run forces a build —
  the SM-arch test. Classes: Car / Bicycle / Person / Roadsign.
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

## Outcome / next step

M8.2C1-A is **passed on DeepStream 8.0**: nvinfer builds a TensorRT engine
on Blackwell sm_120, frames flow at ~30 fps, and TrafficCamNet produces
detections on the live RTSP stream. **Next: M8.2C1-B** — bring this proven
manual pipeline under the supervisor as an inference worker (still no pyds /
NvDsBatchMeta / metadata adapter / observations; those are later milestones).
The DeepStream image pin for that work must be **8.0** (or newer with
TensorRT ≥ 10.7 / CUDA ≥ 12.8) — **7.1 is not usable for inference on
Blackwell**. The M8.2B decode-only worker is unaffected.

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
Blackwell here?" — **DeepStream 8.0: yes; DeepStream 7.1: no**. Metadata
extraction is **not** solved here: the probe reads no `NvDsBatchMeta` and
emits no observations; detection evidence is deepstream-app's own KITTI dump.
