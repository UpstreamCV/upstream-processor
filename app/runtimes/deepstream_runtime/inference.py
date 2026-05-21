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
from dataclasses import dataclass
from pathlib import Path
from typing import Any
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


# --- P3 tracker-probe builders -----------------------------------------

# v0 supports only the no-ReID IOU tracker (simplest stable-id proof; no
# model file beyond what's bundled in libnvds_nvmultiobjecttracker.so).
_KNOWN_TRACKER_ALGORITHMS = {"iou"}


def build_tracker_config(algorithm: str = "iou") -> str:
    """Low-level tracker YAML config for ``nvtracker``.

    The DS 8.0 sample IOU config (``config_tracker_IOU.yml``) is the upstream
    template; we ship our own minimal copy so the worker is self-contained
    and tunables live in this repo. Required by ``nvtracker`` via the
    ``ll-config-file`` property. Pure / side-effect free."""
    algo = algorithm.strip().lower()
    if algo not in _KNOWN_TRACKER_ALGORITHMS:
        raise ValueError(
            f"unknown deepstream tracker algorithm {algorithm!r}; "
            f"v0 supports: {sorted(_KNOWN_TRACKER_ALGORITHMS)}"
        )
    # DS 8.0 nvtracker parses %YAML:1.0 (note the colon, not a space) — keep
    # the prefix exactly. Sections / keys mirror the SDK sample.
    return (
        "%YAML:1.0\n"
        "BaseConfig:\n"
        "  minDetectorConfidence: 0\n"
        "TargetManagement:\n"
        "  preserveStreamUpdateOrder: 0\n"
        "  maxTargetsPerStream: 150\n"
        "  minIouDiff4NewTarget: 0.5\n"
        "  probationAge: 4\n"
        "  maxShadowTrackingAge: 38\n"
        "  earlyTerminationAge: 1\n"
        "TrajectoryManagement:\n"
        "  useUniqueID: 0\n"
        "DataAssociator:\n"
        "  dataAssociatorType: 0\n"
        "  associationMatcherType: 0\n"
        "  checkClassMatch: 1\n"
        "  minMatchingScore4Overall: 0.0\n"
        "  minMatchingScore4SizeSimilarity: 0.0\n"
        "  minMatchingScore4Iou: 0.0\n"
        "  matchingScoreWeight4SizeSimilarity: 0.4\n"
        "  matchingScoreWeight4Iou: 0.6\n"
    )


# --- P3.5 analytics-probe config builder -------------------------------

# Stream id (per-stream sections key off this). v0 only batches a single
# RTSP source so 0 is the only stream index ever used; making it a named
# constant keeps the section names tidy and lets future multi-stream
# work parameterize it without breaking tests.
_ANALYTICS_STREAM_ID = 0

# v0 ships one ROI ("entry-roi") and one line ("entry-line"). Both are
# parameterized but the public knob surface stays tiny: width / height of
# the frame the geometry was authored against. The dashboard-driven /
# operator-authored config generator is a later milestone.
_DEFAULT_ROI_LABEL = "entry-roi"
_DEFAULT_LINE_LABEL = "entry-line"


def normalize_analytics_class_ids(class_ids: str) -> str:
    """Validate + normalize the operator-facing
    ``UPSTREAM_DEEPSTREAM_ANALYTICS_CLASS_IDS`` value into the
    integer string nvdsanalytics' ``class-id=`` line expects.

    Accepted inputs:
      - ``"all"`` / ``""`` / ``"-1"`` → ``"-1"`` (any class counted)
      - Single integer like ``"0"`` / ``"2"`` → unchanged
      - Anything else → ``ValueError`` (multi-class lists are NOT
        supported — nvdsanalytics' parser takes a single int per
        section, so a comma-list would require multiple sections,
        which is a later milestone)."""
    raw = (class_ids or "").strip().lower()
    if raw in ("", "all", "-1"):
        return "-1"
    # Reject comma-lists explicitly so a typo produces a clear error
    # rather than a silently-mistuned filter.
    if "," in raw:
        raise ValueError(
            f"deepstream_analytics_class_ids: comma-lists are not supported "
            f"by nvdsanalytics' single-section class-id key; got {class_ids!r}. "
            "Use 'all' for every class or a single integer for one class."
        )
    try:
        _ = int(raw)
    except ValueError as exc:
        raise ValueError(
            f"deepstream_analytics_class_ids must be 'all', '-1', or a single integer; "
            f"got {class_ids!r}"
        ) from exc
    return raw


@dataclass(frozen=True)
class GeometryPoint:
    """One pixel-coordinate vertex (origin top-left)."""

    x: int
    y: int

    def to_json(self) -> dict[str, int]:
        return {"x": self.x, "y": self.y}


@dataclass(frozen=True)
class RoiGeometry:
    """One analytics ROI as an ordered polygon (clockwise from top-left)."""

    id: str
    label: str
    points: list[GeometryPoint]

    def to_json(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "points": [p.to_json() for p in self.points],
        }


@dataclass(frozen=True)
class LineGeometry:
    """One crossing line as its two segment endpoints. The nvdsanalytics
    direction-reference points are an LC implementation detail and are
    deliberately excluded — a dashboard draws only the segment."""

    id: str
    label: str
    points: list[GeometryPoint]

    def to_json(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "points": [p.to_json() for p in self.points],
        }


@dataclass(frozen=True)
class AnalyticsGeometry:
    """The ROI/line geometry the analytics probe is configured with — the
    single source shared by ``build_analytics_config`` (the nvdsanalytics
    INI) and the P5-C perception-snapshot builder."""

    rois: list[RoiGeometry]
    lines: list[LineGeometry]

    def to_json(self) -> dict[str, Any]:
        return {
            "rois": [r.to_json() for r in self.rois],
            "lines": [line.to_json() for line in self.lines],
        }


def compute_analytics_geometry(
    *,
    frame_width: int = 1280,
    frame_height: int = 720,
    roi_margin_pct: float = 0.05,
    roi_label: str = _DEFAULT_ROI_LABEL,
    line_label: str = _DEFAULT_LINE_LABEL,
) -> AnalyticsGeometry:
    """The v0 analytics geometry: one near-full-frame ROI + one horizontal
    crossing line. Pure / side-effect free. Single source of the geometry
    numbers — ``build_analytics_config`` renders them into the nvdsanalytics
    INI, and the P5-C perception-snapshot builder serializes them as JSON for
    the dashboard overlay."""
    if frame_width <= 0 or frame_height <= 0:
        raise ValueError(
            f"frame_width and frame_height must be > 0; got {frame_width}x{frame_height}"
        )
    if not roi_label or not line_label:
        raise ValueError("roi_label and line_label must be non-empty")
    if not 0.0 <= roi_margin_pct < 0.5:
        raise ValueError(f"roi_margin_pct must be in [0.0, 0.5); got {roi_margin_pct}")
    # Near-full-frame ROI: default ``margin=0.05`` → 5%..95% on both axes.
    # Polygon points clockwise from top-left as nvdsanalytics expects.
    rx0 = int(frame_width * roi_margin_pct)
    rx1 = int(frame_width * (1.0 - roi_margin_pct))
    ry0 = int(frame_height * roi_margin_pct)
    ry1 = int(frame_height * (1.0 - roi_margin_pct))
    roi = RoiGeometry(
        id=roi_label,
        label=roi_label,
        points=[
            GeometryPoint(rx0, ry0),
            GeometryPoint(rx1, ry0),
            GeometryPoint(rx1, ry1),
            GeometryPoint(rx0, ry1),
        ],
    )
    # Horizontal line across the middle of the ROI — the two segment
    # endpoints only (the LC direction-reference points stay INI-side).
    line_y = frame_height // 2
    lx0 = int(frame_width * 0.25)
    lx1 = int(frame_width * 0.75)
    line = LineGeometry(
        id=line_label,
        label=line_label,
        points=[GeometryPoint(lx0, line_y), GeometryPoint(lx1, line_y)],
    )
    return AnalyticsGeometry(rois=[roi], lines=[line])


def build_analytics_config(
    *,
    frame_width: int = 1280,
    frame_height: int = 720,
    roi_label: str = _DEFAULT_ROI_LABEL,
    line_label: str = _DEFAULT_LINE_LABEL,
    roi_margin_pct: float = 0.05,
    class_ids: str = "all",
) -> str:
    """``nvdsanalytics`` config (INI-style) covering one ROI + one line
    crossing. Modelled on the DS 8.0 sample
    ``config_nvdsanalytics.txt`` but stripped to the two rules we
    actually surface in the status payload (ROI + LC). Pure / side-effect
    free so it is unit-testable without DeepStream.

    The ROI is a near-full-frame rectangle (covers ~90% x 90% by
    default after the 5% margin on each side) — wide enough that an
    operator's bbox bottom-center (nvdsanalytics' reference point
    for ROI containment) is inside even when the operator stands at
    the bottom edge of the frame. The line is horizontal across the
    middle of the ROI, so walking through the ROI also crosses the
    line. Coordinates are in pixels of the ``config-width`` /
    ``config-height`` (set to the nvstreammux output).

    ``roi_margin_pct`` controls the inset on each side of the frame
    in [0, 0.49]; ``class_ids`` controls the per-section class
    filter (see ``normalize_analytics_class_ids``).

    Geometry tuning for specific scenes is out of scope for v0; this
    config exists to prove the analytics extraction path, not to ship
    production zones."""
    geometry = compute_analytics_geometry(
        frame_width=frame_width,
        frame_height=frame_height,
        roi_margin_pct=roi_margin_pct,
        roi_label=roi_label,
        line_label=line_label,
    )
    class_id_value = normalize_analytics_class_ids(class_ids)
    # ROI polygon — clockwise from top-left (see compute_analytics_geometry).
    roi_points = ";".join(f"{p.x};{p.y}" for p in geometry.rois[0].points)
    # nvdsanalytics LC requires the 2 line-segment endpoints PLUS 2 direction-
    # reference point pairs: line_x0;line_y0;line_x1;line_y1;dir_x0;dir_y0;dir_x1;dir_y1.
    # Direction reference points (from above the line to below, so "downwards"
    # trips the crossing) are an LC detail kept here, not in the geometry.
    line_points = geometry.lines[0].points
    line_y = line_points[0].y
    dir_x = frame_width // 2
    dir_y0 = max(0, line_y - 80)
    dir_y1 = min(frame_height - 1, line_y + 80)
    lc_points = (
        ";".join(f"{p.x};{p.y}" for p in line_points) + f";{dir_x};{dir_y0};{dir_x};{dir_y1}"
    )
    return (
        "[property]\n"
        "enable=1\n"
        f"config-width={int(frame_width)}\n"
        f"config-height={int(frame_height)}\n"
        # osd-mode=0: don't draw anything (we render to fakesink anyway).
        "osd-mode=0\n"
        "display-font-size=12\n"
        f"[roi-filtering-stream-{_ANALYTICS_STREAM_ID}]\n"
        "enable=1\n"
        # The DS 8.0 nvdsanalytics property parser keys ROIs off "roi-"-
        # prefixed properties: the label is the suffix after "roi-" in the
        # KEY name (NOT in the value); the value is the polygon point list.
        # See sources/gst-plugins/gst-nvdsanalytics/nvdsanalytics_property_parser.cpp
        # (DSANALYTICS_PROPERTY_ROI = "roi-"). Verified against the DS 8.0
        # sample at sample_apps/deepstream-nvdsanalytics-test/config_nvdsanalytics.txt
        # which uses ``roi-RF=`` with "RF" as the label.
        f"roi-{roi_label}={roi_points}\n"
        # inverse-roi=0: count objects INSIDE the polygon (vs outside).
        "inverse-roi=0\n"
        # P4-D follow-up: class filter is env-driven via
        # ``UPSTREAM_DEEPSTREAM_ANALYTICS_CLASS_IDS``. Default ``-1``
        # (all classes) — the live run showed TrafficCamNet sometimes
        # misclassifies people as cars, so probe-default-all is the
        # safe choice. Production scenes can narrow per-deployment.
        f"class-id={class_id_value}\n"
        f"[line-crossing-stream-{_ANALYTICS_STREAM_ID}]\n"
        "enable=1\n"
        # line-crossing-<Label>=<line-coords>;<direction-coords> — label
        # is part of the key (parser strips the "line-crossing-" prefix).
        f"line-crossing-{line_label}={lc_points}\n"
        # Same class filter as the ROI section.
        f"class-id={class_id_value}\n"
        "extended=0\n"
        # mode=loose: count any crossing without strict direction
        # adherence; we want to prove events flow, not enforce direction.
        "mode=loose\n"
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
