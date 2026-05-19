#!/usr/bin/env bash
# P1/M8.2A — smallest DeepStream/GStreamer RTSP *decode* probe (GPU).
#
# Opens the internal RTSP URL inside a GPU-enabled DeepStream container,
# NVDEC-decodes H264 for a short window, prints fps, exits. NO inference,
# NO detection, NO UpStream observation writing — decode proof only.
#
# Run on the GPU host (crowdcountS2). Uses --network host so a
# rtsp://127.0.0.1:18554/... URL resolves to the host's MediaMTX.
# Prints no secrets. Modifies nothing persistent (container is --rm).
#
# NOTE: the DeepStream image is large (~7-9 GB). To avoid a surprise pull
# during a probe, if the image is absent locally you must opt in with
# CONFIRM_PULL=1.
#
# Usage:
#   scripts/probe_deepstream_rtsp.sh "rtsp://127.0.0.1:18554/sess_xxxx"
# Env:
#   DEEPSTREAM_IMAGE  (default below — change per Blackwell note in docs)
#   PROBE_SECONDS     (default 20)
#   CONFIRM_PULL=1    (allow pulling the multi-GB image if not cached)

set -uo pipefail

URL="${1:-${RTSP_URL:-}}"
DEEPSTREAM_IMAGE="${DEEPSTREAM_IMAGE:-nvcr.io/nvidia/deepstream:7.1-samples-multiarch}"
PROBE_SECONDS="${PROBE_SECONDS:-20}"

if [ -z "$URL" ]; then
  echo "usage: $0 <internal_rtsp_url>   (or set RTSP_URL)" >&2
  exit 2
fi
command -v docker >/dev/null 2>&1 || { echo "FAIL docker missing" >&2; exit 1; }

echo "== probe =="
echo "  image=$DEEPSTREAM_IMAGE"
echo "  url=$URL"
echo "  seconds=$PROBE_SECONDS"

if ! docker image inspect "$DEEPSTREAM_IMAGE" >/dev/null 2>&1; then
  if [ "${CONFIRM_PULL:-0}" != "1" ]; then
    echo "FAIL image not cached locally and CONFIRM_PULL!=1." >&2
    echo "     Re-run with CONFIRM_PULL=1 to pull (~7-9 GB, several minutes)." >&2
    exit 3
  fi
  echo "  pulling $DEEPSTREAM_IMAGE (large, please wait) ..."
  docker pull "$DEEPSTREAM_IMAGE" || { echo "FAIL docker pull" >&2; exit 1; }
fi

# NVDEC decode -> fps meter -> fakesink. -e = send EOS on timeout for clean exit.
PIPELINE="rtspsrc location=\"$URL\" protocols=tcp latency=200 ! \
rtph264depay ! h264parse ! nvv4l2decoder ! nvvideoconvert ! \
fpsdisplaysink video-sink=fakesink text-overlay=false sync=false"

echo "== running (timeout ${PROBE_SECONDS}s) =="
set +e
LOG="$(docker run --rm --gpus all --network host "$DEEPSTREAM_IMAGE" \
  bash -lc "timeout -s INT ${PROBE_SECONDS} gst-launch-1.0 -e -v ${PIPELINE} 2>&1")"
set -e 2>/dev/null || true
echo "$LOG" | grep -Ei 'fps|PLAYING|ERROR|could not|Failed' | tail -20 | sed 's/^/  /'

if echo "$LOG" | grep -qi 'ERROR'; then
  echo "FAIL pipeline reported an ERROR (see above)"; exit 1
fi
if echo "$LOG" | grep -Eqi 'rendered: *[1-9][0-9]*|current: *[1-9]'; then
  echo "OK   frames decoded — RTSP read + NVDEC decode proven"
  exit 0
fi
echo "FAIL no decoded frames observed (no active publisher on the path, or"
echo "     image lacks NVDEC for this GPU — see Blackwell note in docs)"
exit 1
