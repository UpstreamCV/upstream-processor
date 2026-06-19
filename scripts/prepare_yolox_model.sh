#!/usr/bin/env bash
# P5-B — operator-run YOLOX-S artifact preparation (GPU host).
#
# Populates the host model store dir for `yolox_s_v1`:
#   1. downloads the OFFICIAL prebuilt raw ONNX from the Megvii YOLOX
#      0.1.1rc0 GitHub release (Apache-2.0) — no torch/export toolchain;
#   2. compiles the Upstream custom bbox parser
#      (app/runtimes/deepstream_runtime/yolox_parser/nvdsparsebbox_yolox.cpp)
#      inside the existing DeepStream 8.0 + pyds image (g++ + DS headers are
#      already in the image; no GPU needed for the compile).
#
# The TensorRT engine is NOT built here — nvinfer builds it on the first
# probe run and caches it next to the ONNX (the model dir is bind-mounted rw).
#
# NEVER run by tests or the runtime worker; artifacts NEVER get committed
# (.gitignore covers *.onnx / *.so / *.engine / models/). No RTSP URLs, no
# secrets anywhere in this script.
#
# Usage (on crowdcountS2, from the repo root):
#   CONFIRM_DOWNLOAD=1 scripts/prepare_yolox_model.sh            # YOLOX-S (default)
#   CONFIRM_DOWNLOAD=1 YOLOX_SIZE=nano scripts/prepare_yolox_model.sh  # P5-K sizes
# Env:
#   YOLOX_SIZE             (default s; one of s|m|nano|tiny — P5-K added the
#                           nano/tiny/m sizes, all Apache-2.0, same parser)
#   MODEL_DIR              (default /opt/upstream/models/yolox_<size>_v1 — must be
#                           writable; override together with
#                           UPSTREAM_DEEPSTREAM_MODEL_STORE_DIR)
#   ONNX_URL               (default: official 0.1.1rc0 yolox_s.onnx release asset)
#   EXPECTED_SHA256        (optional; enforce a pinned digest)
#   DEEPSTREAM_PYDS_IMAGE  (default upstream-deepstream:8.0-pyds-v1.2.2)
#   CONFIRM_DOWNLOAD=1     (required for the network download)
#   FORCE_REBUILD=1        (recompile the parser even if the .so exists)
#
# Exit codes: 0 ready; 1 download/compile failure; 2 bad usage/prereq.

set -uo pipefail

# P5-K: size-parameterized via YOLOX_SIZE (s|m|nano|tiny). Default 's' is the
# original P5-B live-verified path — byte-identical when YOLOX_SIZE is unset.
# nano/tiny use 416x416 input; s/m use 640x640. All four share the same
# Apache-2.0 custom grid-decode parser; only the ONNX artifact differs.
YOLOX_SIZE="${YOLOX_SIZE:-s}"
case "$YOLOX_SIZE" in
  s)    MODEL_ID="yolox_s_v1" ;;
  m)    MODEL_ID="yolox_m_v1" ;;
  nano) MODEL_ID="yolox_nano_v1" ;;
  tiny) MODEL_ID="yolox_tiny_v1" ;;
  *) echo "FAIL unknown YOLOX_SIZE=$YOLOX_SIZE (use s|m|nano|tiny)" >&2; exit 2 ;;
esac
MODEL_DIR="${MODEL_DIR:-/opt/upstream/models/$MODEL_ID}"
ONNX_URL="${ONNX_URL:-https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_${YOLOX_SIZE}.onnx}"
DEEPSTREAM_PYDS_IMAGE="${DEEPSTREAM_PYDS_IMAGE:-upstream-deepstream:8.0-pyds-v1.2.2}"
ONNX_PATH="$MODEL_DIR/yolox_${YOLOX_SIZE}.onnx"
LIB_PATH="$MODEL_DIR/libnvdsparsebbox_yolox.so"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PARSER_CPP="$SCRIPT_DIR/../app/runtimes/deepstream_runtime/yolox_parser/nvdsparsebbox_yolox.cpp"

command -v docker >/dev/null 2>&1 || { echo "FAIL docker missing" >&2; exit 2; }
[ -f "$PARSER_CPP" ] || { echo "FAIL parser source missing: $PARSER_CPP" >&2; exit 2; }
docker image inspect "$DEEPSTREAM_PYDS_IMAGE" >/dev/null 2>&1 \
  || { echo "FAIL image $DEEPSTREAM_PYDS_IMAGE not present (build it first; see docs/local-dev.md)" >&2; exit 2; }

echo "== prepare $MODEL_ID artifacts (YOLOX_SIZE=$YOLOX_SIZE) =="
echo "  model_dir=$MODEL_DIR"
echo "  image=$DEEPSTREAM_PYDS_IMAGE"

mkdir -p "$MODEL_DIR" 2>/dev/null \
  || { echo "FAIL cannot create $MODEL_DIR (need sudo? or override MODEL_DIR + UPSTREAM_DEEPSTREAM_MODEL_STORE_DIR)" >&2; exit 2; }

# --- 1. ONNX download ------------------------------------------------------
if [ -f "$ONNX_PATH" ]; then
  echo "  onnx: already present, skipping download"
else
  if [ "${CONFIRM_DOWNLOAD:-0}" != "1" ]; then
    echo "FAIL $ONNX_PATH missing and CONFIRM_DOWNLOAD!=1." >&2
    echo "     Re-run with CONFIRM_DOWNLOAD=1 to fetch the official Apache-2.0 asset:" >&2
    echo "     $ONNX_URL" >&2
    exit 2
  fi
  echo "  downloading official YOLOX-$YOLOX_SIZE ONNX (Apache-2.0) ..."
  curl -fL --retry 3 --max-time 300 -o "$ONNX_PATH.tmp" "$ONNX_URL" \
    || { echo "FAIL download failed" >&2; rm -f "$ONNX_PATH.tmp"; exit 1; }
  mv "$ONNX_PATH.tmp" "$ONNX_PATH"
fi

SHA256="$(sha256sum "$ONNX_PATH" | awk '{print $1}')"
SIZE="$(stat -c%s "$ONNX_PATH" 2>/dev/null || stat -f%z "$ONNX_PATH")"
echo "  onnx sha256=$SHA256 size=${SIZE}B"
if [ -n "${EXPECTED_SHA256:-}" ] && [ "$SHA256" != "$EXPECTED_SHA256" ]; then
  echo "FAIL onnx sha256 mismatch (expected $EXPECTED_SHA256)" >&2
  exit 1
fi

# --- 2. custom parser compile (in-container; no GPU needed) ----------------
if [ -f "$LIB_PATH" ] && [ "${FORCE_REBUILD:-0}" != "1" ]; then
  echo "  parser: already compiled, skipping (FORCE_REBUILD=1 to recompile)"
else
  echo "  compiling NvDsInferParseYolox inside $DEEPSTREAM_PYDS_IMAGE ..."
  cp "$PARSER_CPP" "$MODEL_DIR/nvdsparsebbox_yolox.cpp"
  # The DS 8.0 samples image's default /usr/local/cuda/include is PARTIAL
  # (crt/host_defines.h missing) — probe the cuda-* toolkit dirs for a
  # complete include tree (live-found: /usr/local/cuda-12.5/targets/...).
  docker run --rm -v "$MODEL_DIR:$MODEL_DIR" "$DEEPSTREAM_PYDS_IMAGE" \
    bash -lc "CUDA_INC=\$(for d in /usr/local/cuda/include /usr/local/cuda-*/targets/x86_64-linux/include; do
        [ -f \"\$d/crt/host_defines.h\" ] && { echo \"\$d\"; break; }; done);
      [ -n \"\$CUDA_INC\" ] || { echo 'no complete CUDA include dir found' >&2; exit 1; };
      echo \"  cuda_include=\$CUDA_INC\";
      g++ -shared -fPIC -O2 -o $LIB_PATH $MODEL_DIR/nvdsparsebbox_yolox.cpp \
        -I/opt/nvidia/deepstream/deepstream/sources/includes \
        -I\"\$CUDA_INC\"" \
    || { echo "FAIL parser compile failed" >&2; exit 1; }
fi
[ -f "$LIB_PATH" ] || { echo "FAIL parser .so missing after compile" >&2; exit 1; }

echo "== ready =="
echo "  $ONNX_PATH"
echo "  $LIB_PATH"
echo "  (TensorRT engine builds on first probe run and is cached here)"
echo "  If MODEL_DIR is not under /opt/upstream/models, set"
echo "  UPSTREAM_DEEPSTREAM_MODEL_STORE_DIR to its parent for the worker."
exit 0
