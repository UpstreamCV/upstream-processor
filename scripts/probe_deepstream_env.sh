#!/usr/bin/env bash
# P1/M8.2A — DeepStream environment probe (read-only).
#
# Run ON THE GPU HOST (crowdcountS2), not the Mac. Checks whether the box can
# run NVIDIA/DeepStream and whether Docker can see the GPU. Modifies nothing
# except (optionally) pulling a small CUDA *base* image for the --gpus test.
# Safe to run repeatedly. Prints no secrets.
#
# Usage:  scripts/probe_deepstream_env.sh
# Env:    CUDA_BASE_IMAGE  (default nvidia/cuda:12.6.3-base-ubuntu24.04)
#         SKIP_DOCKER_GPU=1  to skip the image pull / container run

set -uo pipefail

CUDA_BASE_IMAGE="${CUDA_BASE_IMAGE:-nvidia/cuda:12.6.3-base-ubuntu24.04}"
fail=0
note() { printf '  %s\n' "$*"; }
ok()   { printf 'OK   %s\n' "$*"; }
bad()  { printf 'FAIL %s\n' "$*"; fail=$((fail + 1)); }

echo "== host =="
uname -a
( . /etc/os-release 2>/dev/null && echo "OS=${PRETTY_NAME:-unknown}" ) || true

echo "== NVIDIA driver / GPU =="
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || bad "nvidia-smi query failed"
  nvidia-smi 2>/dev/null | grep -i "CUDA Version" | head -1
  ok "nvidia-smi present"
else
  bad "nvidia-smi MISSING — no usable NVIDIA driver"
fi

echo "== Docker =="
if command -v docker >/dev/null 2>&1; then
  docker --version
  if docker ps >/dev/null 2>&1; then ok "docker usable without sudo"
  else bad "cannot run docker as this user (group/daemon?)"; fi
else
  bad "docker MISSING"
fi

echo "== NVIDIA Container Toolkit =="
if command -v nvidia-ctk >/dev/null 2>&1; then
  nvidia-ctk --version | head -1
  ok "nvidia-ctk present"
else
  bad "nvidia-ctk MISSING — NVIDIA Container Toolkit not installed"
fi
docker info 2>/dev/null | grep -i "Runtimes:" | grep -q nvidia \
  && ok "docker 'nvidia' runtime registered" \
  || note "docker 'nvidia' runtime not listed (—gpus may still work via CDI)"

echo "== Docker -> GPU passthrough =="
if [ "${SKIP_DOCKER_GPU:-0}" = "1" ]; then
  note "skipped (SKIP_DOCKER_GPU=1)"
elif command -v docker >/dev/null 2>&1; then
  if docker run --rm --gpus all "$CUDA_BASE_IMAGE" \
        nvidia-smi --query-gpu=name --format=csv,noheader >/tmp/_ds_gpu.$$  2>&1; then
    ok "container saw GPU(s):"; sed 's/^/    /' /tmp/_ds_gpu.$$
  else
    bad "docker --gpus all could not reach the GPU"; sed 's/^/    /' /tmp/_ds_gpu.$$
  fi
  rm -f /tmp/_ds_gpu.$$
fi

echo "== Host GStreamer (DeepStream NVDEC ships in the DS image, not host) =="
for p in rtspsrc avdec_h264 nvv4l2decoder nvcodec; do
  if command -v gst-inspect-1.0 >/dev/null 2>&1 && gst-inspect-1.0 "$p" >/dev/null 2>&1; then
    note "$p: present"
  else
    note "$p: absent"
  fi
done

echo "== result =="
if [ "$fail" -eq 0 ]; then echo "PASS: environment can run NVIDIA/DeepStream containers"; exit 0
else echo "FAIL: $fail blocking issue(s) above"; exit 1; fi
