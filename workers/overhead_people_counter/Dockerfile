# Overhead people-counter worker image (crowdcount-counter:dev)
#
# Self-contained container-per-session counting worker for Upstream's processor
# supervisor: one PeopleCounterProcessor (YOLO v6 + BoT-SORT + OSNet ReID +
# doorway line-crossing) per session, inference in-process, counts posted to
# crowdcount-server via the service credential. Ported into upstream-processor
# so it no longer depends on crowdcount-counter-server.
#
# Build (context = this dir):
#   docker build -t crowdcount-counter:dev workers/overhead_people_counter

FROM nvidia/cuda:12.8.0-runtime-ubuntu24.04

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1 \
    ffmpeg curl gcc g++ \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/bin/python

# PyTorch STABLE cu128 (RTX 50 Blackwell sm_120), then torchreid from source.
# Stable (not nightly) is pinned-and-aligned and never pruned from the index,
# so the build is reproducible — the original nightly this was ported from has
# since been pruned.
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --break-system-packages torch torchvision --index-url https://download.pytorch.org/whl/cu128 \
    && pip install --no-cache-dir --break-system-packages cython numpy \
    && pip install --no-cache-dir --break-system-packages --no-binary torchreid torchreid \
    && pip install --no-cache-dir --break-system-packages -r /tmp/requirements.txt

# torchreid's runtime deps that its setup doesn't pull in (scipy/tensorboard/
# h5py/yacs for module imports, gdown for the pretrained-weight download).
# Matches the set present in the original orchestrator-based image.
RUN pip install --no-cache-dir --break-system-packages \
    scipy gdown tensorboard h5py yacs matplotlib imageio future six chardet Pillow

# Bake OSNet ReID weights so first run needs no network (was gdrive-fetched).
RUN python3 -c "import torchreid; torchreid.models.build_model(name='osnet_x0_25', num_classes=1000, pretrained=True)"

COPY src /app/src
COPY scripts /app/scripts
COPY models /app/models

# One PeopleCounterProcessor per container; config via CC_* env (--env-file).
ENTRYPOINT ["python3", "/app/scripts/run_counter_worker.py"]
