#!/usr/bin/env bash
# P1/M8.2C1-A — manual DeepStream nvinfer inference smoke probe (GPU).
#
# Runs a DeepStream sample model (Primary_Detector / TrafficCamNet, INT8)
# against the live internal MediaMTX RTSP stream inside a GPU-enabled
# DeepStream container, with NO display: RTSP -> nvstreammux -> nvinfer ->
# fakesink. It forces a fresh TensorRT engine build from the bundled ONNX so
# this *is* the GPU/TensorRT SM-architecture compatibility test. Detection
# evidence comes from deepstream-app's own KITTI dump (gie-kitti-output-dir)
# and the PERF/FPS lines — NO pyds, NO NvDsBatchMeta, NO observations.
#
# Run ON THE GPU HOST (crowdcountS2), not the Mac. Uses --network host so a
# rtsp://127.0.0.1:18554/... URL resolves to the host's MediaMTX. The RTSP
# URL is not a secret but is NOT committed: it is written only into a
# mktemp dir under /tmp and shredded on exit; all printed output is redacted.
# The container is --rm; nothing persistent is modified.
#
# NOTE: the DeepStream image is large (~20 GB). If it is not cached locally
# you must opt in to a pull with CONFIRM_PULL=1.
#
# Usage:
#   scripts/probe_deepstream_inference.sh "rtsp://127.0.0.1:18554/sess_xxxx"
#   or:  RTSP_URL="rtsp://..." scripts/probe_deepstream_inference.sh
# Env:
#   DEEPSTREAM_IMAGE  (default below — override to test a Blackwell-capable tag)
#   PROBE_SECONDS     (default 120 — must cover TensorRT engine build + run)
#   CONFIRM_PULL=1    (allow pulling the multi-GB image if not cached)
#
# Exit codes:
#   0  inference actually ran (engine built + frames/PERF + detections)
#   1  inference failed (TensorRT/CUDA/SM block, RTSP error, no frames, ...)
#   2  bad usage
#   3  image not cached and CONFIRM_PULL!=1

set -uo pipefail

URL="${1:-${RTSP_URL:-}}"
DEEPSTREAM_IMAGE="${DEEPSTREAM_IMAGE:-nvcr.io/nvidia/deepstream:7.1-samples-multiarch}"
PROBE_SECONDS="${PROBE_SECONDS:-120}"

redact() { sed -E 's#(rtsp://[^/ ]*/)[A-Za-z0-9_.:-]+#\1sess_<redacted>#g'; }

if [ -z "$URL" ]; then
  echo "usage: $0 <internal_rtsp_url>   (or set RTSP_URL)" >&2
  exit 2
fi
command -v docker >/dev/null 2>&1 || { echo "FAIL docker missing" >&2; exit 1; }

echo "== inference probe =="
echo "  image=$DEEPSTREAM_IMAGE"
echo "  url=$(printf '%s' "$URL" | redact)"
echo "  seconds=$PROBE_SECONDS"

if ! docker image inspect "$DEEPSTREAM_IMAGE" >/dev/null 2>&1; then
  if [ "${CONFIRM_PULL:-0}" != "1" ]; then
    echo "FAIL image not cached locally and CONFIRM_PULL!=1." >&2
    echo "     Re-run with CONFIRM_PULL=1 to pull (~20 GB, several minutes)." >&2
    exit 3
  fi
  echo "  pulling $DEEPSTREAM_IMAGE (large, please wait) ..."
  docker pull "$DEEPSTREAM_IMAGE" || { echo "FAIL docker pull" >&2; exit 1; }
fi

WORK="$(mktemp -d /tmp/upstream-m82c1.XXXXXX)"
cleanup() {
  find "$WORK" -type f -exec shred -u {} + 2>/dev/null || true
  rm -rf "$WORK" 2>/dev/null || true
}
trap cleanup EXIT
mkdir -p "$WORK/kitti"

M=/opt/nvidia/deepstream/deepstream/samples/models/Primary_Detector

# PGIE: bundled TrafficCamNet ONNX, INT8, batch 1. No model-engine-file that
# exists -> nvinfer must BUILD the TensorRT engine (the SM-arch test).
cat > "$WORK/pgie.txt" <<EOF
[property]
gpu-id=0
net-scale-factor=0.00392156862745098
onnx-file=$M/resnet18_trafficcamnet_pruned.onnx
model-engine-file=$WORK/trafficcamnet_b1_gpu0_int8.engine
labelfile-path=$M/labels.txt
int8-calib-file=$M/cal_trt.bin
batch-size=1
process-mode=1
model-color-format=0
network-mode=1
num-detected-classes=4
interval=0
gie-unique-id=1
cluster-mode=2
[class-attrs-all]
topk=20
nms-iou-threshold=0.5
pre-cluster-threshold=0.2
[class-attrs-0]
topk=20
nms-iou-threshold=0.5
pre-cluster-threshold=0.4
EOF

# deepstream-app: RTSP/TCP source -> nvinfer -> fakesink. No tiled-display,
# no OSD, no tracker -> no X11. KITTI dump = detection evidence.
cat > "$WORK/ds_app.txt" <<EOF
[application]
enable-perf-measurement=1
perf-measurement-interval-sec=5
gie-kitti-output-dir=$WORK/kitti
[tiled-display]
enable=0
[source0]
enable=1
type=4
uri=$URL
num-sources=1
gpu-id=0
cudadec-memtype=0
select-rtp-protocol=4
latency=200
[streammux]
gpu-id=0
batch-size=1
batched-push-timeout=40000
width=1280
height=720
live-source=1
nvbuf-memory-type=0
[primary-gie]
enable=1
gpu-id=0
gie-unique-id=1
batch-size=1
config-file=$WORK/pgie.txt
[sink0]
enable=1
type=1
sync=0
[osd]
enable=0
[tracker]
enable=0
[tests]
file-loop=0
EOF

echo "== running deepstream-app (timeout ${PROBE_SECONDS}s) =="
set +e
docker run --rm --gpus all --network host -v "$WORK:$WORK" "$DEEPSTREAM_IMAGE" \
  bash -lc "timeout -s INT ${PROBE_SECONDS} deepstream-app -c $WORK/ds_app.txt" \
  > "$WORK/run.log" 2>&1
DEXIT=$?
set -e 2>/dev/null || true

KITTI_FRAMES="$(find "$WORK/kitti" -type f -name '*.txt' 2>/dev/null | wc -l | tr -d ' ')"
DETECTIONS="$(cat "$WORK"/kitti/*.txt 2>/dev/null | wc -l | tr -d ' ')"

echo "-- key log lines (redacted) --"
grep -aiE 'unsupported sm|internal error|\[trt\]|cuda error|deserialize|engine.*(built|fail)|serialized|PERF|playing|ERROR|could not|core' \
  "$WORK/run.log" 2>/dev/null | redact | tail -25 | sed 's/^/  /'
echo "  docker_exit=$DEXIT  kitti_frames=$KITTI_FRAMES  detection_lines=$DETECTIONS"

# --- classify -------------------------------------------------------------
if grep -aqiE 'Unsupported SM|Unsupported SM:' "$WORK/run.log"; then
  echo "FAIL TensorRT cannot build an engine for this GPU's SM architecture"
  echo "     (Blackwell sm_120 / compute 12.0 needs a newer DeepStream image"
  echo "      whose bundled TensorRT >= 10.7 and CUDA >= 12.8 — see"
  echo "      docs/deepstream-inference-smoke.md)."
  exit 1
fi
if grep -aqiE 'failed to build network|buildSerializedNetwork.*Error|deserialize backend context.*failed.*try rebuild' "$WORK/run.log" \
   && [ ! -s "$WORK/trafficcamnet_b1_gpu0_int8.engine" ]; then
  echo "FAIL TensorRT engine build failed (see log above)"
  exit 1
fi
if [ "$KITTI_FRAMES" -gt 0 ] && [ "$DETECTIONS" -gt 0 ]; then
  echo "OK   nvinfer ran on the live RTSP stream — TensorRT engine built and"
  echo "     produced detections ($DETECTIONS objects across $KITTI_FRAMES frames)"
  exit 0
fi
if grep -aqiE 'PERF:.*fps' "$WORK/run.log" && [ -s "$WORK/trafficcamnet_b1_gpu0_int8.engine" ]; then
  echo "OK   engine built and pipeline ran with PERF/FPS (no objects in view)"
  exit 0
fi
echo "FAIL inference did not run (no engine + frames + detections). Likely an"
echo "     RTSP/source error or an engine/model problem — see log above."
exit 1
