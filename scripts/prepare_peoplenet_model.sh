#!/usr/bin/env bash
# P5-K — operator-run NVIDIA PeopleNet artifact preparation (GPU host).
# IN-REPO PREP: PeopleNet reuses the BUILT-IN DetectNet_v2 parser (no custom
# parser to compile), but it has NOT been live-verified in Upstream — the
# package that references it is 'experimental'.
#
# LICENSE: NVIDIA PeopleNet ships under the NGC model EULA (commercial use is
# covered under the NVIDIA model license; re-review before any 'supported'
# promotion). Registered license_class=commercial_agreement / first_party=true.
#
# Populates the host model store dir for peoplenet_v1:
#   1. downloads the deployable ONNX from NGC (resnet34_peoplenet.onnx);
#   2. writes the labels file (person/bag/face, in model class order).
#
# The TensorRT engine is NOT built here — nvinfer builds it on the first run
# and caches it next to the ONNX (the model dir is bind-mounted rw).
#
# NEVER run by tests or the runtime worker; artifacts NEVER get committed
# (.gitignore covers *.onnx / *.so / *.engine / models/). No secrets here.
#
# Usage (on the GPU host, from the repo root):
#   CONFIRM_DOWNLOAD=1 scripts/prepare_peoplenet_model.sh
# Env:
#   MODEL_DIR   (default /opt/upstream/models/peoplenet_v1 — must be writable)
#   ONNX_URL    (default: NGC PeopleNet deployable resnet34 ONNX asset)
#   CONFIRM_DOWNLOAD=1 (required for the network download)
#
# Exit codes: 0 ready; 1 download failure; 2 bad usage/prereq.

set -uo pipefail

MODEL_DIR="${MODEL_DIR:-/opt/upstream/models/peoplenet_v1}"
# NGC PeopleNet deployable ONNX (pruned_quantized resnet34). Override ONNX_URL
# if NGC changes the asset path / version.
ONNX_URL="${ONNX_URL:-https://api.ngc.nvidia.com/v2/models/nvidia/tao/peoplenet/versions/deployable_quantized_onnx_v2.6.3/files/resnet34_peoplenet.onnx}"
ONNX_PATH="$MODEL_DIR/resnet34_peoplenet.onnx"
LABELS_PATH="$MODEL_DIR/labels.txt"

echo "== prepare peoplenet_v1 artifacts (NVIDIA PeopleNet, DetectNet_v2) =="
echo "  model_dir=$MODEL_DIR"

mkdir -p "$MODEL_DIR" 2>/dev/null \
  || { echo "FAIL cannot create $MODEL_DIR (need sudo? or override MODEL_DIR)" >&2; exit 2; }

# --- 1. ONNX download ------------------------------------------------------
if [ -f "$ONNX_PATH" ]; then
  echo "  onnx: already present, skipping download"
else
  if [ "${CONFIRM_DOWNLOAD:-0}" != "1" ]; then
    echo "FAIL $ONNX_PATH missing and CONFIRM_DOWNLOAD!=1." >&2
    echo "     Re-run with CONFIRM_DOWNLOAD=1 to fetch the NGC PeopleNet asset:" >&2
    echo "     $ONNX_URL" >&2
    exit 2
  fi
  echo "  downloading PeopleNet ONNX from NGC (NVIDIA model EULA) ..."
  curl -fL --retry 3 --max-time 600 -o "$ONNX_PATH.tmp" "$ONNX_URL" \
    || { echo "FAIL download failed (NGC may require auth; see docs/local-dev.md)" >&2; rm -f "$ONNX_PATH.tmp"; exit 1; }
  mv "$ONNX_PATH.tmp" "$ONNX_PATH"
fi

SHA256="$(sha256sum "$ONNX_PATH" | awk '{print $1}')"
SIZE="$(stat -c%s "$ONNX_PATH" 2>/dev/null || stat -f%z "$ONNX_PATH")"
echo "  onnx sha256=$SHA256 size=${SIZE}B"

# --- 2. labels (person/bag/face, model class order) ------------------------
printf 'person\nbag\nface\n' > "$LABELS_PATH"
echo "  labels: $LABELS_PATH"

echo "== ready (experimental — validate detections on a real stream first) =="
echo "  $ONNX_PATH"
echo "  $LABELS_PATH"
echo "  (TensorRT engine builds on first run and is cached here)"
exit 0
