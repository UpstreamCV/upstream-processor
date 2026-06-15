#!/usr/bin/env bash
# P5-E — manual IOU-vs-NvDCF A/B comparison driver (GPU host, real worker path).
#
# Runs the REAL upstream-processor supervisor twice against the SAME active
# validated session — once forced to the IOU YOLOX analytics package, once to
# the NvDCF variant — capturing each run's redacted processor log, then diffs
# the two with scripts/compare_tracker_ab.py. The standalone pyds probe is NOT
# used (it has no nvtracker); this exercises the same worker / process-manager
# path the packages actually run, so the comparison reflects production behavior.
# See the P5-E A/B runbook in docs/local-dev.md.
#
# Run ON THE GPU/streaming host with an active validated session, AFTER stopping
# the systemd processor so two supervisors don't grab the same session. Source
# your processor secret env first (UPSTREAM_PROCESSOR_WEBHOOK_SECRET). The RTSP
# URL is never an argument here (the worker polls the API for the session); the
# captured logs are redacted regardless.
#
# Usage:
#   CONFIRM_RUN=1 scripts/probe_tracker_ab.sh
# Env:
#   UPSTREAM_API_BASE_URL             (default https://api.upstreamcv.com)
#   UPSTREAM_PROCESSOR_ID             (default proc_p5e_ab_local)
#   UPSTREAM_PROCESSOR_WEBHOOK_SECRET (required)
#   IOU_PKG    (default deepstream.yolox_analytics_probe.v0)
#   NVDCF_PKG  (default deepstream.yolox_analytics_nvdcf_probe.v0)
#   PROBE_SECONDS (default 180)
#   PYTHON     (default python3)
#   KEEP_TEMP_FILES=1   (retain the per-run capture dir)
#
# Exit codes: 0 both runs captured + compared; 1 a run failed; 2 bad env;
#             3 not confirmed (CONFIRM_RUN unset).

set -uo pipefail

API_BASE="${UPSTREAM_API_BASE_URL:-https://api.upstreamcv.com}"
PROCESSOR_ID="${UPSTREAM_PROCESSOR_ID:-proc_p5e_ab_local}"
IOU_PKG="${IOU_PKG:-deepstream.yolox_analytics_probe.v0}"
NVDCF_PKG="${NVDCF_PKG:-deepstream.yolox_analytics_nvdcf_probe.v0}"
PROBE_SECONDS="${PROBE_SECONDS:-180}"
PYTHON="${PYTHON:-python3}"

redact() { sed -E 's#(rtsp://[^/ ]*/)[A-Za-z0-9_.:-]+#\1sess_<redacted>#g'; }

if [ "${CONFIRM_RUN:-0}" != "1" ]; then
  echo "refusing to run: this starts the REAL supervisor against a live session and posts to the API." >&2
  echo "  re-run with CONFIRM_RUN=1 once the systemd processor is stopped and a validated session is active." >&2
  exit 3
fi
if [ -z "${UPSTREAM_PROCESSOR_WEBHOOK_SECRET:-}" ]; then
  echo "FAIL UPSTREAM_PROCESSOR_WEBHOOK_SECRET is required (source your processor secret env first)" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
COMPARE_PY="$SCRIPT_DIR/compare_tracker_ab.py"
HELPERS_PY="$SCRIPT_DIR/_tracker_ab_helpers.py"
for f in "$COMPARE_PY" "$HELPERS_PY"; do
  [ -f "$f" ] || { echo "FAIL missing artifact: $f" >&2; exit 2; }
done

WORK="$(mktemp -d /tmp/upstream-p5e-ab.XXXXXX)"
cleanup() {
  if [ "${KEEP_TEMP_FILES:-0}" = "1" ]; then
    echo "  KEEP_TEMP_FILES=1; leaving $WORK"
    return
  fi
  rm -rf "$WORK" 2>/dev/null || true
}
trap cleanup EXIT

run_pkg() {  # $1=label $2=package_id $3=capture_path
  local label="$1" pkg="$2" out="$3"
  echo "== A/B run: $label ($pkg) for ${PROBE_SECONDS}s =="
  # Bounded run: the worker keeps processing the eligible session until the
  # timeout sends SIGINT for a clean shutdown.
  UPSTREAM_API_BASE_URL="$API_BASE" \
  UPSTREAM_PROCESSOR_ID="${PROCESSOR_ID}_${label}" \
  UPSTREAM_FORCE_PACKAGE_ID="$pkg" \
    timeout -s INT "$((PROBE_SECONDS + 10))" "$PYTHON" -m app.main > "$out.raw" 2>&1
  local rc=$?
  # Redact before anything is read back, so no raw stream key survives.
  redact < "$out.raw" > "$out"
  rm -f "$out.raw"
  echo "  exit=$rc captured_lines=$(wc -l < "$out" | tr -d ' ')"
  return 0
}

IOU_CAP="$WORK/iou.log"
NVDCF_CAP="$WORK/nvdcf.log"
run_pkg "iou" "$IOU_PKG" "$IOU_CAP"
run_pkg "nvdcf" "$NVDCF_PKG" "$NVDCF_CAP"

echo "== comparison (redacted) =="
"$PYTHON" "$COMPARE_PY" iou "$IOU_CAP" nvdcf "$NVDCF_CAP" | redact
echo "NOTE: event-level counts come from the observations the worker POSTs to the API."
echo "      For a full event-level A/B, append each run's observation rows (one"
echo "      JSON object per line) to the capture files before comparing — see the"
echo "      P5-E runbook in docs/local-dev.md."
exit 0
