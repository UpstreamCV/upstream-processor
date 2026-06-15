#!/usr/bin/env bash
# P5-E — hands-off IOU-vs-NvDCF tracker A/B on a RECORDED CLIP (GPU host).
#
# Runs DeepStream's stock `deepstream-app` against a recorded file (the bundled
# DS sample clip by default) twice — once with our rendered IOU tracker config,
# once with NvDCF — with `[tracker]` enabled and `kitti-track-output-dir` set,
# then diffs per-frame track-ID stability with compare_tracker_ab.py. Identical
# input both passes, no RTSP/session/streaming-server/person needed. This
# directly exercises OUR authored NvDCF ll-config on the GPU and measures
# fragmentation; it does NOT exercise the zone/line/dwell analytics events
# (those need the analytics pipeline + geometry — a separate live/seeded run).
#
# Inputs (rendered on a dev box with the repo, then copied here):
#   IOU_YML / NVDCF_YML — our build_tracker_config("iou"/"nvdcf") output.
# The sample clip + primary-gie config live INSIDE the DeepStream image.
#
# Usage (from the dir holding this script + the two YAMLs + the helper):
#   scripts/probe_tracker_ab_clip.sh
# Env:
#   IMAGE        (default upstream-deepstream:8.0-pyds-v1.2.2)
#   GPU_DEVICE   (default 1 — the near-idle GPU; avoids the live processor)
#   IOU_YML      (default <script_dir>/tracker_iou.yml)
#   NVDCF_YML    (default <script_dir>/tracker_nvdcf.yml)
#   CLIP         (in-container path; default the DS sample 1080p clip)
#   PRIMARY_GIE  (in-container path; default the DS sample primary infer config)
#   TRACKER_LIB  (in-container path; default libnvds_nvmultiobjecttracker.so)
#   PROBE_TIMEOUT (default 300s — first run also builds the TensorRT engine)
#   PYTHON       (host python for parsing/compare; default python3)
#   WORKDIR      (default a fresh /tmp dir)
#   KEEP_TEMP_FILES=1
#
# Exit codes: 0 both ran + compared; 1 a run produced no track output; 2 setup.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
IMAGE="${IMAGE:-upstream-deepstream:8.0-pyds-v1.2.2}"
GPU_DEVICE="${GPU_DEVICE:-1}"
IOU_YML="${IOU_YML:-$SCRIPT_DIR/tracker_iou.yml}"
NVDCF_YML="${NVDCF_YML:-$SCRIPT_DIR/tracker_nvdcf.yml}"
CLIP="${CLIP:-/opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h264.mp4}"
PRIMARY_GIE="${PRIMARY_GIE:-/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_infer_primary.txt}"
TRACKER_LIB="${TRACKER_LIB:-/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so}"
PROBE_TIMEOUT="${PROBE_TIMEOUT:-300}"
PYTHON="${PYTHON:-python3}"

# EXTRA_MOUNT: an additional host dir to bind-mount into the deepstream-app
# container (e.g. a YOLOX model store), so a custom PRIMARY_GIE can reference
# model artifacts by absolute path. Empty => no extra mount.
EXTRA_MOUNT_ARG=""
if [ -n "${EXTRA_MOUNT:-}" ]; then EXTRA_MOUNT_ARG="-v ${EXTRA_MOUNT}:${EXTRA_MOUNT}"; fi

# Defensive redaction helper (the clip is a local file, but keep the hygiene
# contract: never trace, never echo a stream key if one is ever passed).
redact() { sed -E 's#(rtsp://[^/ ]*/)[A-Za-z0-9_.:-]+#\1sess_<redacted>#g'; }

command -v docker >/dev/null 2>&1 || { echo "FAIL docker missing" >&2; exit 2; }
docker image inspect "$IMAGE" >/dev/null 2>&1 \
  || { echo "FAIL image $IMAGE not present" >&2; exit 2; }
for f in "$IOU_YML" "$NVDCF_YML" "$SCRIPT_DIR/_tracker_ab_helpers.py" "$SCRIPT_DIR/compare_tracker_ab.py"; do
  [ -f "$f" ] || { echo "FAIL missing required file: $f" >&2; exit 2; }
done

WORKDIR="${WORKDIR:-$(mktemp -d /tmp/upstream-p5e-clip.XXXXXX)}"
mkdir -p "$WORKDIR"
cleanup() {
  if [ "${KEEP_TEMP_FILES:-0}" = "1" ]; then echo "  KEEP_TEMP_FILES=1; leaving $WORKDIR"; return; fi
  rm -rf "$WORKDIR" 2>/dev/null || sudo rm -rf "$WORKDIR" 2>/dev/null || true
}
trap cleanup EXIT

cp "$IOU_YML" "$WORKDIR/tracker_iou.yml"
cp "$NVDCF_YML" "$WORKDIR/tracker_nvdcf.yml"

# CLIP_HOST: use a host-side recorded clip (e.g. a representative person clip)
# instead of the in-image sample. Copied into the mounted WORKDIR so the
# deepstream-app container sees it. Never echoed beyond its basename.
CLIP_IN_WORKDIR=0
if [ -n "${CLIP_HOST:-}" ]; then
  [ -f "$CLIP_HOST" ] || { echo "FAIL CLIP_HOST not found: $(basename "$CLIP_HOST")" >&2; exit 2; }
  cp "$CLIP_HOST" "$WORKDIR/clip.mp4"
  CLIP="$WORKDIR/clip.mp4"
  CLIP_IN_WORKDIR=1
fi

echo "== P5-E clip tracker A/B =="
echo "  image=$IMAGE gpu=$GPU_DEVICE clip=$(basename "$CLIP")"
echo "  primary_gie=$PRIMARY_GIE"

# Existence check (no GPU). A host clip lives in the mounted WORKDIR; the
# in-image sample is checked inside the image.
if [ "$CLIP_IN_WORKDIR" = "1" ]; then
  if ! docker run --rm -v "$WORKDIR:$WORKDIR" $EXTRA_MOUNT_ARG "$IMAGE" bash -lc "[ -f '$CLIP' ] && [ -f '$PRIMARY_GIE' ]"; then
    echo "FAIL host clip or primary-gie config not found; check CLIP_HOST / PRIMARY_GIE" >&2
    exit 2
  fi
elif ! docker run --rm "$IMAGE" bash -lc "[ -f '$CLIP' ] && [ -f '$PRIMARY_GIE' ]"; then
  echo "FAIL clip or primary-gie config not found inside the image; override CLIP / PRIMARY_GIE" >&2
  exit 2
fi

write_ds_config() {  # $1=algo
  local algo="$1"
  cat > "$WORKDIR/ds_${algo}.txt" <<EOF
[application]
enable-perf-measurement=1
perf-measurement-interval-sec=5
kitti-track-output-dir=$WORKDIR/kitti_${algo}

[tiled-display]
enable=0

[source0]
enable=1
type=3
uri=file://$CLIP
num-sources=1
gpu-id=0
cudadec-memtype=0

[streammux]
gpu-id=0
batch-size=1
batched-push-timeout=40000
width=1920
height=1080
live-source=0

[primary-gie]
enable=1
gpu-id=0
config-file=$PRIMARY_GIE

[tracker]
enable=1
tracker-width=960
tracker-height=544
ll-lib-file=$TRACKER_LIB
ll-config-file=$WORKDIR/tracker_${algo}.yml
gpu-id=0
enable-batch-process=1

[sink0]
enable=1
type=1
sync=0

[osd]
enable=0

[tests]
file-loop=0
EOF
}

run_algo() {  # $1=algo
  local algo="$1"
  echo "== deepstream-app run: $algo =="
  write_ds_config "$algo"
  mkdir -p "$WORKDIR/kitti_${algo}"
  # device=$GPU_DEVICE maps to in-container gpu-id 0; --rm; workdir lets the
  # sample primary-gie config resolve its relative model paths.
  docker run --rm --gpus "device=${GPU_DEVICE}" \
    -v "$WORKDIR:$WORKDIR" $EXTRA_MOUNT_ARG \
    -w /opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app \
    "$IMAGE" \
    bash -lc "timeout -s INT ${PROBE_TIMEOUT} deepstream-app -c $WORKDIR/ds_${algo}.txt" \
    > "$WORKDIR/run_${algo}.log" 2>&1
  local rc=$?
  local kitti_files
  kitti_files="$(find "$WORKDIR/kitti_${algo}" -name '*.txt' 2>/dev/null | wc -l | tr -d ' ')"
  echo "  docker_exit=$rc kitti_frames=$kitti_files"
  echo "  -- PERF (FPS) --"
  grep -aE '\*\*PERF' "$WORKDIR/run_${algo}.log" 2>/dev/null | redact | tail -3 | sed 's/^/    /'
  if [ "${kitti_files:-0}" -eq 0 ]; then
    echo "  -- last log lines (redacted) --"
    grep -aiE 'error|fail|unsupported|tracker|ll-config|onnx|engine' "$WORKDIR/run_${algo}.log" \
      2>/dev/null | redact | tail -15 | sed 's/^/    /'
    return 1
  fi
  echo "  -- sample kitti lines --"
  head -3 "$(find "$WORKDIR/kitti_${algo}" -name '*.txt' | sort | sed -n '10p')" 2>/dev/null | sed 's/^/    /'
  return 0
}

IOU_OK=0; NVDCF_OK=0
run_algo iou && IOU_OK=1
run_algo nvdcf && NVDCF_OK=1

if [ "$IOU_OK" != "1" ] || [ "$NVDCF_OK" != "1" ]; then
  echo "FAIL at least one run produced no track output (see logs above)." >&2
  exit 1
fi

# KITTI dirs -> snapshot JSON-lines, using the (unit-tested) pure helper.
PYTHONPATH="$SCRIPT_DIR" "$PYTHON" - "$WORKDIR" <<'PYEOF'
import json, sys
from _tracker_ab_helpers import kitti_dir_to_snapshots
work = sys.argv[1]
for algo in ("iou", "nvdcf"):
    snaps = kitti_dir_to_snapshots(f"{work}/kitti_{algo}")
    with open(f"{work}/{algo}.jsonl", "w", encoding="utf-8") as fh:
        for s in snaps:
            fh.write(json.dumps(s) + "\n")
PYEOF

echo "== comparison (track-ID stability; events N/A for clip A/B) =="
"$PYTHON" "$SCRIPT_DIR/compare_tracker_ab.py" iou "$WORKDIR/iou.jsonl" nvdcf "$WORKDIR/nvdcf.jsonl"
echo "NOTE: this clip A/B scores tracker fragmentation only (unique IDs / longest"
echo "      persistence / switch-proxy). zone/line/dwell event recall still needs"
echo "      the analytics pipeline — see the P5-E live runbook in docs/local-dev.md."
exit 0
