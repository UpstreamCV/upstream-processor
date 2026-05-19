#!/usr/bin/env bash
# P1/M8.2A — RTSP reachability probe (no GPU, no DeepStream).
#
# Verifies the internal MediaMTX RTSP URL from a M8.0 eligible session is
# reachable + decodable BEFORE involving DeepStream. Run on the host that
# will run the processor (crowdcountS2). The RTSP URL is NOT a secret, but
# do not commit it. Safe to run repeatedly; modifies nothing.
#
# Usage:  scripts/probe_rtsp_url.sh "rtsp://127.0.0.1:18554/sess_xxxx"
#   or:   RTSP_URL="rtsp://..." scripts/probe_rtsp_url.sh

set -uo pipefail

URL="${1:-${RTSP_URL:-}}"
if [ -z "$URL" ]; then
  echo "usage: $0 <internal_rtsp_url>   (or set RTSP_URL)" >&2
  exit 2
fi
case "$URL" in
  rtsp://*) : ;;
  *) echo "FAIL: not an rtsp:// URL: $URL" >&2; exit 2 ;;
esac

# host:port out of rtsp://host:port/path
hostport="${URL#rtsp://}"; hostport="${hostport%%/*}"
host="${hostport%%:*}"
port="${hostport##*:}"; [ "$port" = "$host" ] && port=554
echo "== target =="
echo "  url=$URL"
echo "  host=$host port=$port"

echo "== TCP reachability =="
if command -v nc >/dev/null 2>&1; then
  if nc -z -w5 "$host" "$port" 2>/dev/null; then echo "OK   tcp $host:$port open"
  else echo "FAIL tcp $host:$port closed/unreachable"; exit 1; fi
else
  echo "  nc missing — skipping raw TCP check"
fi

echo "== ffprobe (RTSP over TCP, video stream 0) =="
if ! command -v ffprobe >/dev/null 2>&1; then
  echo "  ffprobe MISSING — install ffmpeg or use the DeepStream probe instead"
  exit 1
fi
out="$(ffprobe -v error -rtsp_transport tcp -timeout 10000000 \
        -select_streams v:0 \
        -show_entries stream=codec_name,width,height,r_frame_rate,pix_fmt \
        -of default=nw=1 "$URL" 2>&1)"
rc=$?
echo "$out" | sed 's/^/  /'
if [ $rc -eq 0 ] && echo "$out" | grep -q codec_name=; then
  echo "OK   RTSP opened and a video stream was described"
  echo "$out" | grep -q 'codec_name=h264' \
    && echo "OK   codec is h264 (expected)" \
    || echo "WARN codec is not h264 — check approved config"
  exit 0
else
  echo "FAIL ffprobe could not open/describe the RTSP stream"
  echo "     (common when no publisher is on the path yet — needs an active stream)"
  exit 1
fi
