#!/usr/bin/env bash
# P2 — manual pyds metadata probe driver (GPU host).
#
# Builds (if needed) a thin DeepStream 8.0 + pyds image, generates a temp
# work dir holding a TrafficCamNet PGIE config + the Python probe, and runs
# scripts/deepstream_pyds_metadata_probe.py inside the container against the
# live RTSP stream. Walks NvDsBatchMeta/NvDsFrameMeta/NvDsObjectMeta via
# pyds; prints counters + a tiny bounded sample. Manual-only — no
# ObservationWriter, no UpStream API writes, no metadata adapter, no pyds-
# anything in app/.
#
# Run ON THE GPU HOST (crowdcountS2). The DeepStream 8.0 base image is
# ~33 GB; the +pyds layer is small but the BASE pull requires CONFIRM_PULL=1,
# and the +pyds build requires CONFIRM_BUILD=1. The container is --rm. The
# RTSP URL goes only into temp pgie config + subprocess argv; logs redact it.
#
# Usage:
#   scripts/probe_deepstream_pyds_metadata.sh "rtsp://127.0.0.1:18554/sess_xxxx"
# Env:
#   DEEPSTREAM_PYDS_IMAGE  (default upstream-deepstream:8.0-pyds-v1.2.2)
#   DEEPSTREAM_BASE_IMAGE  (default nvcr.io/nvidia/deepstream:8.0-samples-multiarch)
#   PROBE_SECONDS          (default 45)
#   CONFIRM_BUILD=1        (allow building the local pyds image if missing)
#   CONFIRM_PULL=1         (allow pulling the multi-GB base image if missing)
#   KEEP_TEMP_FILES=1      (retain the per-run /tmp work dir for inspection)
#
# Exit codes:
#   0  pyds probe ran and saw frames (summary printed)
#   1  pipeline / pyds / inference failure (see redacted logs)
#   2  bad usage / missing prerequisite
#   3  pyds image not built locally and CONFIRM_BUILD!=1

set -uo pipefail

URL="${1:-${RTSP_URL:-}}"
DEEPSTREAM_PYDS_IMAGE="${DEEPSTREAM_PYDS_IMAGE:-upstream-deepstream:8.0-pyds-v1.2.2}"
DEEPSTREAM_BASE_IMAGE="${DEEPSTREAM_BASE_IMAGE:-nvcr.io/nvidia/deepstream:8.0-samples-multiarch}"
PROBE_SECONDS="${PROBE_SECONDS:-45}"

redact() { sed -E 's#(rtsp://[^/ ]*/)[A-Za-z0-9_.:-]+#\1sess_<redacted>#g'; }

if [ -z "$URL" ]; then
  echo "usage: $0 <internal_rtsp_url>   (or set RTSP_URL)" >&2
  exit 2
fi
command -v docker >/dev/null 2>&1 || { echo "FAIL docker missing" >&2; exit 2; }

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROBE_PY="$SCRIPT_DIR/deepstream_pyds_metadata_probe.py"
HELPERS_PY="$SCRIPT_DIR/_pyds_probe_helpers.py"
DOCKERFILE="$(cd "$SCRIPT_DIR/../docker" && pwd)/Dockerfile.pyds"
for f in "$PROBE_PY" "$HELPERS_PY" "$DOCKERFILE"; do
  [ -f "$f" ] || { echo "FAIL missing artifact: $f" >&2; exit 2; }
done

echo "== pyds metadata probe =="
echo "  pyds_image=$DEEPSTREAM_PYDS_IMAGE"
echo "  base_image=$DEEPSTREAM_BASE_IMAGE"
echo "  url=$(printf '%s' "$URL" | redact)"
echo "  seconds=$PROBE_SECONDS"

# --- ensure the pyds image is present (build on demand) -------------------
if ! docker image inspect "$DEEPSTREAM_PYDS_IMAGE" >/dev/null 2>&1; then
  if [ "${CONFIRM_BUILD:-0}" != "1" ]; then
    echo "FAIL pyds image not built locally and CONFIRM_BUILD!=1." >&2
    echo "     Re-run with CONFIRM_BUILD=1 to build $DEEPSTREAM_PYDS_IMAGE." >&2
    echo "     The build also requires the base image to be present (or" >&2
    echo "     CONFIRM_PULL=1 to pull it; the base is ~33 GB)." >&2
    exit 3
  fi
  if ! docker image inspect "$DEEPSTREAM_BASE_IMAGE" >/dev/null 2>&1; then
    if [ "${CONFIRM_PULL:-0}" != "1" ]; then
      echo "FAIL base image $DEEPSTREAM_BASE_IMAGE not cached and CONFIRM_PULL!=1." >&2
      exit 3
    fi
    echo "  pulling $DEEPSTREAM_BASE_IMAGE (~33 GB) ..."
    docker pull "$DEEPSTREAM_BASE_IMAGE" || { echo "FAIL docker pull" >&2; exit 1; }
  fi
  echo "  building $DEEPSTREAM_PYDS_IMAGE from $DOCKERFILE ..."
  docker build \
    --build-arg "BASE_IMAGE=$DEEPSTREAM_BASE_IMAGE" \
    -t "$DEEPSTREAM_PYDS_IMAGE" \
    -f "$DOCKERFILE" \
    "$(dirname "$DOCKERFILE")" \
    || { echo "FAIL docker build" >&2; exit 1; }
fi

# --- per-run temp work dir (raw RTSP URL goes only into pgie config) ------
WORK="$(mktemp -d /tmp/upstream-p2-pyds.XXXXXX)"
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

M=/opt/nvidia/deepstream/deepstream/samples/models/Primary_Detector
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

# --- run the probe --------------------------------------------------------
echo "== running probe (timeout ${PROBE_SECONDS}s) =="
set +e
docker run --rm --gpus all --network host \
  -v "$WORK:$WORK" \
  -e RTSP_URL="$URL" \
  -e PYTHONDONTWRITEBYTECODE=1 \
  "$DEEPSTREAM_PYDS_IMAGE" \
  bash -lc "timeout -s INT $((PROBE_SECONDS + 10)) python3 -B $WORK/probe.py --pgie-config $WORK/pgie.txt --seconds $PROBE_SECONDS" \
  > "$WORK/run.log" 2>&1
DEXIT=$?
set -e 2>/dev/null || true

echo "-- key log lines (redacted) --"
grep -aiE 'pyds probe|playing;|PYDS_PROBE_SUMMARY=|GST_ERROR|ERROR|Unsupported SM|serialize cuda engine|deserialize' \
  "$WORK/run.log" 2>/dev/null | redact | tail -20 | sed 's/^/  /'
echo "  docker_exit=$DEXIT"

SUMMARY_LINE="$(grep -m1 -aE '^PYDS_PROBE_SUMMARY=' "$WORK/run.log" || true)"
if [ -z "$SUMMARY_LINE" ]; then
  if grep -aqiE 'Unsupported SM' "$WORK/run.log"; then
    echo "FAIL TensorRT engine build failed (Unsupported SM). Verify image/GPU." >&2
    exit 1
  fi
  echo "FAIL pyds probe produced no summary (see redacted log)." >&2
  exit 1
fi

# Parse summary with python3 (available on the host or skip). Use awk fallback.
parse() { python3 -c "
import json,sys
d=json.loads(sys.argv[1].split('PYDS_PROBE_SUMMARY=',1)[1])
print('frames_seen=%s' % d['frames_seen'])
print('frames_with_objects=%s' % d['frames_with_objects'])
print('objects_seen=%s' % d['objects_seen'])
print('has_bbox=%s' % d['has_bbox'])
print('has_confidence=%s' % d['has_confidence'])
print('has_tracker_id=%s' % d['has_tracker_id'])
print('label_counts=%s' % json.dumps(d.get('label_counts',{})))
print('sample_count=%s' % len(d.get('sample',[])))
" "$1" 2>/dev/null; }

echo "-- summary --"
parse "$SUMMARY_LINE" | sed 's/^/  /'

# Pretty-print the sample (small, bounded, redacted, no raw URL inside).
echo "-- neutral metadata sample (first frames with objects, redacted) --"
python3 -c "
import json,sys,re
d=json.loads(sys.argv[1].split('PYDS_PROBE_SUMMARY=',1)[1])
out=json.dumps(d.get('sample',[]), indent=2)
print(re.sub(r'rtsp://[^\\s\"]+', '<redacted-rtsp>', out))
" "$SUMMARY_LINE" 2>/dev/null | sed 's/^/  /' | head -80

# Exit-classify.
FRAMES="$(python3 -c "
import json,sys; d=json.loads(sys.argv[1].split('PYDS_PROBE_SUMMARY=',1)[1]); print(d['frames_seen'])
" "$SUMMARY_LINE" 2>/dev/null)"
if [ "${FRAMES:-0}" -gt 0 ]; then
  echo "OK   pyds pad probe walked NvDsBatchMeta and produced metadata"
  exit 0
fi
echo "FAIL pyds attached but saw 0 frames (pipeline/RTSP issue?)." >&2
exit 1
