#!/usr/bin/env bash
# P5-B — manual YOLOX pyds metadata probe driver (GPU host).
#
# Same shape as scripts/probe_deepstream_pyds_metadata.sh (the P2 TrafficCamNet
# probe) but runs YOLOX-S: official raw ONNX + the Upstream custom grid-decode
# bbox parser, FP16 TensorRT engine cached in the model store dir. Walks
# NvDsBatchMeta/NvDsFrameMeta/NvDsObjectMeta via pyds and prints counters +
# a bounded sample. Manual-only — no ObservationWriter, no UpStream API
# writes, no metadata adapter.
#
# Prereq (one-time): CONFIRM_DOWNLOAD=1 scripts/prepare_yolox_model.sh
#
# Run ON THE GPU HOST. The RTSP URL goes only into the container env +
# subprocess argv; all output is redacted. First run also BUILDS the FP16
# TensorRT engine (1-3 min) — use PROBE_SECONDS>=240 for the first run.
#
# Usage:
#   scripts/probe_deepstream_yolox_metadata.sh "rtsp://127.0.0.1:18554/sess_xxxx"
# Env:
#   MODEL_DIR              (default /opt/upstream/models/yolox_s_v1)
#   DEEPSTREAM_PYDS_IMAGE  (default upstream-deepstream:8.0-pyds-v1.2.2)
#   PROBE_SECONDS          (default 240 — engine build headroom; later runs can use 45)
#   KEEP_TEMP_FILES=1      (retain the per-run /tmp work dir)
#
# Exit codes: 0 probe saw frames; 1 pipeline/inference failure; 2 bad usage.

set -uo pipefail

URL="${1:-${RTSP_URL:-}}"
MODEL_DIR="${MODEL_DIR:-/opt/upstream/models/yolox_s_v1}"
DEEPSTREAM_PYDS_IMAGE="${DEEPSTREAM_PYDS_IMAGE:-upstream-deepstream:8.0-pyds-v1.2.2}"
PROBE_SECONDS="${PROBE_SECONDS:-240}"

redact() { sed -E 's#(rtsp://[^/ ]*/)[A-Za-z0-9_.:-]+#\1sess_<redacted>#g'; }

if [ -z "$URL" ]; then
  echo "usage: $0 <internal_rtsp_url>   (or set RTSP_URL)" >&2
  exit 2
fi
command -v docker >/dev/null 2>&1 || { echo "FAIL docker missing" >&2; exit 2; }
docker image inspect "$DEEPSTREAM_PYDS_IMAGE" >/dev/null 2>&1 \
  || { echo "FAIL image $DEEPSTREAM_PYDS_IMAGE not present" >&2; exit 2; }

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROBE_PY="$SCRIPT_DIR/deepstream_pyds_metadata_probe.py"
HELPERS_PY="$SCRIPT_DIR/_pyds_probe_helpers.py"
REGISTRY_JSON="$SCRIPT_DIR/../app/packages/model_registry.json"
for f in "$PROBE_PY" "$HELPERS_PY" "$REGISTRY_JSON"; do
  [ -f "$f" ] || { echo "FAIL missing artifact: $f" >&2; exit 2; }
done
for f in "$MODEL_DIR/yolox_s.onnx" "$MODEL_DIR/libnvdsparsebbox_yolox.so"; do
  [ -f "$f" ] || { echo "FAIL model artifact missing: $f" >&2; \
    echo "     Run: CONFIRM_DOWNLOAD=1 scripts/prepare_yolox_model.sh" >&2; exit 2; }
done

echo "== yolox pyds metadata probe =="
echo "  model_id=yolox_s_v1 family=yolox parser=NvDsInferParseYolox backend=tensorrt_fp16"
echo "  image=$DEEPSTREAM_PYDS_IMAGE"
echo "  model_dir=$MODEL_DIR"
echo "  url=$(printf '%s' "$URL" | redact)"
echo "  seconds=$PROBE_SECONDS"

WORK="$(mktemp -d /tmp/upstream-p5b-yolox.XXXXXX)"
cleanup() {
  if [ "${KEEP_TEMP_FILES:-0}" = "1" ]; then
    echo "  KEEP_TEMP_FILES=1; leaving $WORK"
    return
  fi
  find "$WORK" -type f -exec shred -u {} + 2>/dev/null || true
  rm -rf "$WORK" 2>/dev/null || true
}
trap cleanup EXIT
cp "$PROBE_PY" "$WORK/probe.py"
cp "$HELPERS_PY" "$WORK/_pyds_probe_helpers.py"

# Labels from the registry's supported_classes (COCO order) — same single
# source the runtime plan uses.
python3 - "$REGISTRY_JSON" > "$WORK/labels.txt" <<'PYEOF'
import json, sys
models = json.load(open(sys.argv[1]))["models"]
spec = next(m for m in models if m["model_id"] == "yolox_s_v1")
print("\n".join(spec["supported_classes"]))
PYEOF
NUM_CLASSES="$(wc -l < "$WORK/labels.txt" | tr -d ' ')"

cat > "$WORK/pgie.txt" <<EOF
[property]
gpu-id=0
net-scale-factor=1.0
model-color-format=1
onnx-file=$MODEL_DIR/yolox_s.onnx
model-engine-file=$MODEL_DIR/yolox_s_b1_gpu0_fp16.engine
labelfile-path=$WORK/labels.txt
batch-size=1
process-mode=1
network-mode=2
num-detected-classes=$NUM_CLASSES
interval=0
gie-unique-id=1
cluster-mode=2
maintain-aspect-ratio=1
symmetric-padding=0
parse-bbox-func-name=NvDsInferParseYolox
custom-lib-path=$MODEL_DIR/libnvdsparsebbox_yolox.so
[class-attrs-all]
topk=300
nms-iou-threshold=0.45
pre-cluster-threshold=0.25
EOF

echo "== running probe (timeout ${PROBE_SECONDS}s; first run builds the FP16 engine) =="
set +e
docker run --rm --gpus all --network host \
  -v "$WORK:$WORK" \
  -v "$MODEL_DIR:$MODEL_DIR" \
  -e RTSP_URL="$URL" \
  -e PYTHONDONTWRITEBYTECODE=1 \
  "$DEEPSTREAM_PYDS_IMAGE" \
  bash -lc "timeout -s INT $((PROBE_SECONDS + 10)) python3 -B $WORK/probe.py --pgie-config $WORK/pgie.txt --seconds $PROBE_SECONDS" \
  > "$WORK/run.log" 2>&1
DEXIT=$?
set -e 2>/dev/null || true

echo "-- key log lines (redacted) --"
grep -aiE 'pyds probe|playing;|PYDS_PROBE_SUMMARY=|GST_ERROR|ERROR|Unsupported SM|NvDsInferParseYolox|serialize cuda engine|deserialize' \
  "$WORK/run.log" 2>/dev/null | redact | tail -20 | sed 's/^/  /'
echo "  docker_exit=$DEXIT"

SUMMARY_LINE="$(grep -m1 -aE '^PYDS_PROBE_SUMMARY=' "$WORK/run.log" || true)"
if [ -z "$SUMMARY_LINE" ]; then
  if grep -aqiE 'Unsupported SM' "$WORK/run.log"; then
    echo "FAIL TensorRT engine build failed (Unsupported SM). Verify image/GPU." >&2
    exit 1
  fi
  echo "FAIL yolox probe produced no summary (see redacted log)." >&2
  exit 1
fi

echo "-- summary --"
python3 -c "
import json,sys
d=json.loads(sys.argv[1].split('PYDS_PROBE_SUMMARY=',1)[1])
print('frames_seen=%s' % d['frames_seen'])
print('frames_with_objects=%s' % d['frames_with_objects'])
print('objects_seen=%s' % d['objects_seen'])
print('has_bbox=%s' % d['has_bbox'])
print('has_confidence=%s' % d['has_confidence'])
print('has_tracker_id=%s' % d['has_tracker_id'])
print('class_counts=%s' % json.dumps(d.get('class_counts',{})))
print('label_counts=%s' % json.dumps(d.get('label_counts',{})))
print('sample_count=%s' % len(d.get('sample',[])))
" "$SUMMARY_LINE" 2>/dev/null | sed 's/^/  /'

echo "-- neutral metadata sample (bounded, redacted) --"
python3 -c "
import json,sys,re
d=json.loads(sys.argv[1].split('PYDS_PROBE_SUMMARY=',1)[1])
out=json.dumps(d.get('sample',[]), indent=2)
print(re.sub(r'rtsp://[^\\s\"]+', '<redacted-rtsp>', out))
" "$SUMMARY_LINE" 2>/dev/null | sed 's/^/  /' | head -80

FRAMES="$(python3 -c "
import json,sys; d=json.loads(sys.argv[1].split('PYDS_PROBE_SUMMARY=',1)[1]); print(d['frames_seen'])
" "$SUMMARY_LINE" 2>/dev/null)"
if [ "${FRAMES:-0}" -gt 0 ]; then
  echo "OK   yolox pyds metadata probe produced metadata"
  exit 0
fi
echo "FAIL probe attached but saw 0 frames (pipeline/RTSP issue?)." >&2
exit 1
