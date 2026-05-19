"""M8.2C1-B DeepStream nvinfer inference subprocess manager.

Runs the *exact* pipeline proven in M8.2C1-A
(``scripts/probe_deepstream_inference.sh``) as a managed asyncio subprocess:
a bounded ``deepstream-app`` running RTSP → ``nvstreammux`` → ``nvinfer``
(bundled TrafficCamNet sample model) → ``fakesink``, with no display, the
TensorRT engine built fresh from the sample ONNX, and detection evidence
written to a KITTI dump (``gie-kitti-output-dir``).

    docker run --rm --name <uniq> --gpus all --network <net> \
      -v <workdir>:<workdir> <ds8-image> \
      bash -lc "timeout -s INT <t> deepstream-app -c <workdir>/ds_app.txt"

Inference only — **no pyds, no NvDsBatchMeta, no metadata adapter, no
observations**. The RTSP URL is written ONLY into the per-run temp
``ds_app.txt`` under the system tempdir (never committed, cleaned up by
default) and into the subprocess argv; it is never logged (logs use the
redacted form). KITTI counts are status/debug evidence only — *not* metadata
extraction.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import shutil
import tempfile
from pathlib import Path
from uuid import uuid4

from app.redact import redact_rtsp

_log = logging.getLogger("upstream_processor.deepstream.infer")

# Symlinked sample-model path; valid in both DeepStream 7.1 and 8.0 images
# (the M8.2C1-A probe built the engine from exactly this path).
_MODEL_DIR = "/opt/nvidia/deepstream/deepstream/samples/models/Primary_Detector"

# v0 wires only the bundled TrafficCamNet primary detector (Primary_Detector).
_KNOWN_MODELS = {"trafficcamnet", "sample_trafficcamnet"}


def resolve_model_dir(model: str) -> str:
    """Map the configured model name to its in-container sample dir.

    Raises ``ValueError`` for an unknown model so the worker posts ``failed``
    with a clear message rather than launching a misconfigured pipeline."""
    if model.strip().lower() not in _KNOWN_MODELS:
        raise ValueError(
            f"unknown deepstream inference model {model!r}; v0 supports: {sorted(_KNOWN_MODELS)}"
        )
    return _MODEL_DIR


def build_pgie_config(*, model_dir: str, engine_path: str) -> str:
    """PGIE config: bundled TrafficCamNet ONNX, INT8, batch 1. The
    ``model-engine-file`` points at a not-yet-existing path so nvinfer must
    BUILD the TensorRT engine (the SM-arch test). Mirrors M8.2C1-A."""
    return (
        "[property]\n"
        "gpu-id=0\n"
        "net-scale-factor=0.00392156862745098\n"
        f"onnx-file={model_dir}/resnet18_trafficcamnet_pruned.onnx\n"
        f"model-engine-file={engine_path}\n"
        f"labelfile-path={model_dir}/labels.txt\n"
        f"int8-calib-file={model_dir}/cal_trt.bin\n"
        "batch-size=1\n"
        "process-mode=1\n"
        "model-color-format=0\n"
        "network-mode=1\n"
        "num-detected-classes=4\n"
        "interval=0\n"
        "gie-unique-id=1\n"
        "cluster-mode=2\n"
        "[class-attrs-all]\n"
        "topk=20\n"
        "nms-iou-threshold=0.5\n"
        "pre-cluster-threshold=0.2\n"
        "[class-attrs-0]\n"
        "topk=20\n"
        "nms-iou-threshold=0.5\n"
        "pre-cluster-threshold=0.4\n"
    )


def build_app_config(
    *,
    rtsp_url: str,
    pgie_path: str,
    kitti_dir: str,
    width: int,
    height: int,
    rtsp_transport: str,
) -> str:
    """deepstream-app config: RTSP source → nvinfer → fakesink. No
    tiled-display, no OSD, no tracker (no X11). KITTI dump = detection
    evidence. The raw RTSP URL appears only in this temp file. Mirrors
    M8.2C1-A."""
    # deepstream-app source `select-rtp-protocol`: 4 = TCP, 0 = UDP/any.
    rtp_proto = 4 if rtsp_transport.strip().lower() == "tcp" else 0
    return (
        "[application]\n"
        "enable-perf-measurement=1\n"
        "perf-measurement-interval-sec=5\n"
        f"gie-kitti-output-dir={kitti_dir}\n"
        "[tiled-display]\n"
        "enable=0\n"
        "[source0]\n"
        "enable=1\n"
        "type=4\n"
        f"uri={rtsp_url}\n"
        "num-sources=1\n"
        "gpu-id=0\n"
        "cudadec-memtype=0\n"
        f"select-rtp-protocol={rtp_proto}\n"
        "latency=200\n"
        "[streammux]\n"
        "gpu-id=0\n"
        "batch-size=1\n"
        "batched-push-timeout=40000\n"
        f"width={width}\n"
        f"height={height}\n"
        "live-source=1\n"
        "nvbuf-memory-type=0\n"
        "[primary-gie]\n"
        "enable=1\n"
        "gpu-id=0\n"
        "gie-unique-id=1\n"
        "batch-size=1\n"
        f"config-file={pgie_path}\n"
        "[sink0]\n"
        "enable=1\n"
        "type=1\n"
        "sync=0\n"
        "[osd]\n"
        "enable=0\n"
        "[tracker]\n"
        "enable=0\n"
        "[tests]\n"
        "file-loop=0\n"
    )


def build_inference_docker_command(
    *,
    work_dir: str,
    image: str,
    network: str,
    probe_timeout_seconds: float,
    container_name: str,
) -> list[str]:
    """The docker argv for the bounded inference run. Pure/side-effect free
    so it is unit-testable without Docker. Mirrors the proven M8.2C1-A
    invocation (``--gpus all``, ``--network``, work dir bind-mounted at the
    same path so the configs' absolute paths resolve inside the container)."""
    inner = (
        f"timeout -s INT {int(probe_timeout_seconds)} deepstream-app -c {work_dir}/ds_app.txt 2>&1"
    )
    return [
        "docker", "run", "--rm", "--name", container_name,
        "--gpus", "all", "--network", network,
        "-v", f"{work_dir}:{work_dir}", image,
        "bash", "-lc", inner,
    ]  # fmt: skip


class DeepStreamInferenceProcessManager:
    """Real Docker-backed ``ProcessManager`` for the inference probe.

    Owns a per-run temp work dir (deepstream-app + PGIE config + KITTI output
    dir), bind-mounted into the DeepStream 8.0 container. Streams stdout,
    supports bounded graceful stop + SIGKILL-after-timeout, exposes KITTI
    detection counts, and removes the temp dir on stop unless
    ``keep_temp_files`` is set. The RTSP URL is only in the temp config +
    subprocess argv; logs use the redacted form."""

    def __init__(
        self,
        *,
        rtsp_url: str,
        image: str,
        network: str,
        rtsp_transport: str,
        probe_timeout_seconds: float,
        stop_timeout_seconds: float,
        model: str,
        output_base_dir: str,
        keep_temp_files: bool,
        width: int,
        height: int,
    ) -> None:
        self._rtsp_url = rtsp_url
        self._image = image
        self._network = network
        self._rtsp_transport = rtsp_transport
        self._probe_timeout = probe_timeout_seconds
        self._stop_timeout = stop_timeout_seconds
        self._model = model
        self._output_base_dir = output_base_dir
        self._keep_temp_files = keep_temp_files
        self._width = width
        self._height = height
        self._container = f"upstream-dsi-{uuid4().hex[:12]}"
        self._work_dir: Path | None = None
        self._proc: asyncio.subprocess.Process | None = None
        self._returncode: int | None = None

    # --- temp work dir ----------------------------------------------------

    def _prepare_work_dir(self) -> Path:
        model_dir = resolve_model_dir(self._model)  # may raise ValueError
        base = self._output_base_dir.strip() or None
        if base is not None:
            Path(base).mkdir(parents=True, exist_ok=True)
        work = Path(tempfile.mkdtemp(prefix="upstream-m82c1b.", dir=base))
        kitti = work / "kitti"
        kitti.mkdir(parents=True, exist_ok=True)
        pgie_path = work / "pgie.txt"
        app_path = work / "ds_app.txt"
        engine_path = work / "trafficcamnet_b1_gpu0_int8.engine"
        pgie_path.write_text(
            build_pgie_config(model_dir=model_dir, engine_path=str(engine_path)),
            encoding="utf-8",
        )
        # The raw RTSP URL goes ONLY into this temp file (never logged, never
        # committed; removed on stop unless keep_temp_files).
        app_path.write_text(
            build_app_config(
                rtsp_url=self._rtsp_url,
                pgie_path=str(pgie_path),
                kitti_dir=str(kitti),
                width=self._width,
                height=self._height,
                rtsp_transport=self._rtsp_transport,
            ),
            encoding="utf-8",
        )
        return work

    def _cleanup_work_dir(self) -> None:
        if self._work_dir is None or self._keep_temp_files:
            return
        with contextlib.suppress(OSError):
            shutil.rmtree(self._work_dir, ignore_errors=True)

    def kitti_stats(self) -> tuple[int, int]:
        """(frame_file_count, total_detection_lines) from the KITTI dump.

        Status/debug evidence only — NOT metadata extraction. Best-effort:
        returns ``(0, 0)`` if the dir is gone (e.g. already cleaned)."""
        if self._work_dir is None:
            return (0, 0)
        kitti = self._work_dir / "kitti"
        if not kitti.is_dir():
            return (0, 0)
        frames = 0
        detections = 0
        try:
            for f in kitti.glob("*.txt"):
                frames += 1
                with contextlib.suppress(OSError):
                    text = f.read_text(encoding="utf-8", errors="replace")
                    detections += sum(1 for line in text.splitlines() if line.strip())
        except OSError:
            return (frames, detections)
        return (frames, detections)

    # --- ProcessManager protocol -----------------------------------------

    def command(self) -> list[str]:
        assert self._work_dir is not None, "command() called before start()"
        return build_inference_docker_command(
            work_dir=str(self._work_dir),
            image=self._image,
            network=self._network,
            probe_timeout_seconds=self._probe_timeout,
            container_name=self._container,
        )

    async def start(self) -> None:
        self._work_dir = self._prepare_work_dir()
        argv = self.command()
        _log.info(
            "deepstream inference start: image=%s network=%s model=%s url=%s",
            self._image,
            self._network,
            self._model,
            redact_rtsp(self._rtsp_url),  # never log the real URL
        )
        self._proc = await asyncio.create_subprocess_exec(
            *argv,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )

    async def readline(self) -> str | None:
        if self._proc is None or self._proc.stdout is None:
            return None
        data = await self._proc.stdout.readline()
        if not data:
            self._returncode = await self._proc.wait()
            return None
        return data.decode("utf-8", "replace").rstrip("\n")

    async def stop(self) -> None:
        try:
            if self._proc is not None:
                try:
                    stopper = await asyncio.create_subprocess_exec(
                        "docker", "stop", "--time", str(int(self._stop_timeout)),
                        self._container,
                        stdout=asyncio.subprocess.DEVNULL,
                        stderr=asyncio.subprocess.DEVNULL,
                    )  # fmt: skip
                    await asyncio.wait_for(stopper.wait(), timeout=self._stop_timeout + 5)
                except (OSError, TimeoutError) as exc:
                    _log.warning("docker stop best-effort failed: %s", exc)

                if self._proc.returncode is None:
                    with contextlib.suppress(ProcessLookupError):
                        self._proc.terminate()
                    try:
                        await asyncio.wait_for(self._proc.wait(), timeout=self._stop_timeout)
                    except TimeoutError:
                        _log.warning(
                            "inference subprocess did not exit in %.0fs; killing",
                            self._stop_timeout,
                        )
                        with contextlib.suppress(ProcessLookupError):
                            self._proc.kill()
                        await self._proc.wait()
                self._returncode = self._proc.returncode
        finally:
            self._cleanup_work_dir()

    @property
    def returncode(self) -> int | None:
        return self._returncode
