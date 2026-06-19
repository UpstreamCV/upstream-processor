#!/usr/bin/env bash
# P5-K — operator-run Ultralytics (YOLO11 / YOLOv8) artifact preparation
# (GPU host). IN-REPO PREP: the custom decode parser is authored but NOT yet
# GPU-validated; the packages that reference these models are 'experimental'.
#
# ⚠ LICENSE: Ultralytics YOLO11/YOLOv8 are AGPL-3.0. Upstream holds an
# Ultralytics Enterprise license for development; first-party hosting is gated
# on legal sign-off (the model is registered first_party=false / BYOM-only).
# This script requires ACK_ULTRALYTICS_LICENSE=1 to proceed.
#
# Populates the host model store dir for one Ultralytics model:
#   1. exports the official ONNX from the .pt weights via the `ultralytics`
#      package (default no-NMS detection output — matches the Upstream parser);
#   2. compiles the Upstream custom decode parser
#      (app/runtimes/deepstream_runtime/ultralytics_parser/nvdsparsebbox_ultralytics.cpp)
#      inside the DeepStream 8.0 + pyds image (g++ + DS headers; no GPU needed).
#
# The TensorRT engine is NOT built here — nvinfer builds it on the first run
# and caches it next to the ONNX (the model dir is bind-mounted rw).
#
# NEVER run by tests or the runtime worker; artifacts NEVER get committed
# (.gitignore covers *.onnx / *.so / *.engine / models/). No secrets here.
#
# Usage (on the GPU host, from the repo root):
#   CONFIRM_DOWNLOAD=1 ACK_ULTRALYTICS_LICENSE=1 \
#     ULTRA_MODEL=yolo11n scripts/prepare_ultralytics_model.sh
# Env:
#   ULTRA_MODEL            (required; one of yolo11n|yolo11s|yolo11m|yolov8n)
#   MODEL_DIR             (default /opt/upstream/models/<model>_v1 — writable)
#   ULTRALYTICS_IMAGE     (default ultralytics/ultralytics:latest — exports ONNX)
#   DEEPSTREAM_PYDS_IMAGE (default upstream-deepstream:8.0-pyds-v1.2.2 — parser)
#   CONFIRM_DOWNLOAD=1    (required for the network download)
#   ACK_ULTRALYTICS_LICENSE=1 (required — confirms the AGPL/Enterprise posture)
#   FORCE_REBUILD=1       (recompile the parser even if the .so exists)
#
# Exit codes: 0 ready; 1 export/compile failure; 2 bad usage/prereq.

set -uo pipefail

ULTRA_MODEL="${ULTRA_MODEL:-}"
case "$ULTRA_MODEL" in
  yolo11n) MODEL_ID="yolo11n_v1" ;;
  yolo11s) MODEL_ID="yolo11s_v1" ;;
  yolo11m) MODEL_ID="yolo11m_v1" ;;
  yolov8n) MODEL_ID="yolov8n_v1" ;;
  *) echo "FAIL set ULTRA_MODEL to one of yolo11n|yolo11s|yolo11m|yolov8n" >&2; exit 2 ;;
esac

MODEL_DIR="${MODEL_DIR:-/opt/upstream/models/$MODEL_ID}"
ULTRALYTICS_IMAGE="${ULTRALYTICS_IMAGE:-ultralytics/ultralytics:latest}"
DEEPSTREAM_PYDS_IMAGE="${DEEPSTREAM_PYDS_IMAGE:-upstream-deepstream:8.0-pyds-v1.2.2}"
ONNX_PATH="$MODEL_DIR/${ULTRA_MODEL}.onnx"
LIB_PATH="$MODEL_DIR/libnvdsparsebbox_ultralytics.so"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PARSER_CPP="$SCRIPT_DIR/../app/runtimes/deepstream_runtime/ultralytics_parser/nvdsparsebbox_ultralytics.cpp"

command -v docker >/dev/null 2>&1 || { echo "FAIL docker missing" >&2; exit 2; }
[ -f "$PARSER_CPP" ] || { echo "FAIL parser source missing: $PARSER_CPP" >&2; exit 2; }
docker image inspect "$DEEPSTREAM_PYDS_IMAGE" >/dev/null 2>&1 \
  || { echo "FAIL image $DEEPSTREAM_PYDS_IMAGE not present (build it first; see docs/local-dev.md)" >&2; exit 2; }

if [ "${ACK_ULTRALYTICS_LICENSE:-0}" != "1" ]; then
  echo "FAIL Ultralytics is AGPL-3.0. Re-run with ACK_ULTRALYTICS_LICENSE=1 to" >&2
  echo "     confirm you have an Ultralytics Enterprise license / AGPL compliance." >&2
  exit 2
fi

echo "== prepare $MODEL_ID artifacts (Ultralytics $ULTRA_MODEL) =="
echo "  model_dir=$MODEL_DIR"
echo "  ⚠ AGPL model — first_party=false (BYOM); not auto-provided in production"

mkdir -p "$MODEL_DIR" 2>/dev/null \
  || { echo "FAIL cannot create $MODEL_DIR (need sudo? or override MODEL_DIR)" >&2; exit 2; }

# --- 1. ONNX export (no embedded NMS — matches the Upstream parser) ---------
if [ -f "$ONNX_PATH" ]; then
  echo "  onnx: already present, skipping export"
else
  if [ "${CONFIRM_DOWNLOAD:-0}" != "1" ]; then
    echo "FAIL $ONNX_PATH missing and CONFIRM_DOWNLOAD!=1." >&2
    echo "     Re-run with CONFIRM_DOWNLOAD=1 to export ${ULTRA_MODEL}.onnx via $ULTRALYTICS_IMAGE." >&2
    exit 2
  fi
  echo "  exporting ${ULTRA_MODEL}.onnx via $ULTRALYTICS_IMAGE ..."
  # Export inside the ultralytics image; opset 12, static shapes, no NMS.
  docker run --rm -v "$MODEL_DIR:$MODEL_DIR" -w "$MODEL_DIR" "$ULTRALYTICS_IMAGE" \
    bash -lc "yolo export model=${ULTRA_MODEL}.pt format=onnx opset=12 dynamic=False simplify=True nms=False" \
    || { echo "FAIL onnx export failed" >&2; exit 1; }
  [ -f "$ONNX_PATH" ] || { echo "FAIL ${ULTRA_MODEL}.onnx not produced by export" >&2; exit 1; }
fi

SHA256="$(sha256sum "$ONNX_PATH" | awk '{print $1}')"
SIZE="$(stat -c%s "$ONNX_PATH" 2>/dev/null || stat -f%z "$ONNX_PATH")"
echo "  onnx sha256=$SHA256 size=${SIZE}B"

# --- 2. custom parser compile (in-container; no GPU needed) -----------------
if [ -f "$LIB_PATH" ] && [ "${FORCE_REBUILD:-0}" != "1" ]; then
  echo "  parser: already compiled, skipping (FORCE_REBUILD=1 to recompile)"
else
  echo "  compiling NvDsInferParseUltralytics inside $DEEPSTREAM_PYDS_IMAGE ..."
  echo "  ⚠ parser is authored-not-GPU-validated — verify detections before relying on them"
  cp "$PARSER_CPP" "$MODEL_DIR/nvdsparsebbox_ultralytics.cpp"
  docker run --rm -v "$MODEL_DIR:$MODEL_DIR" "$DEEPSTREAM_PYDS_IMAGE" \
    bash -lc "CUDA_INC=\$(for d in /usr/local/cuda/include /usr/local/cuda-*/targets/x86_64-linux/include; do
        [ -f \"\$d/crt/host_defines.h\" ] && { echo \"\$d\"; break; }; done);
      [ -n \"\$CUDA_INC\" ] || { echo 'no complete CUDA include dir found' >&2; exit 1; };
      echo \"  cuda_include=\$CUDA_INC\";
      g++ -shared -fPIC -O2 -o $LIB_PATH $MODEL_DIR/nvdsparsebbox_ultralytics.cpp \
        -I/opt/nvidia/deepstream/deepstream/sources/includes \
        -I\"\$CUDA_INC\"" \
    || { echo "FAIL parser compile failed" >&2; exit 1; }
fi
[ -f "$LIB_PATH" ] || { echo "FAIL parser .so missing after compile" >&2; exit 1; }

echo "== ready (experimental — validate detections on a real stream first) =="
echo "  $ONNX_PATH"
echo "  $LIB_PATH"
echo "  (TensorRT engine builds on first run and is cached here)"
exit 0
