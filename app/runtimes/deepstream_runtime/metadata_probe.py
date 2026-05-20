"""P2 — supervisor-managed pyds metadata probe subprocess manager.

Manages a bounded DeepStream 8.0 + pyds container as an asyncio
subprocess. The in-container app
(:mod:`app.runtimes.deepstream_runtime.pyds_metadata_app`) is bind-mounted
into the work dir and executed under ``timeout -s INT <t>``. The
container's stdout carries the proven structured metadata lines
(``UPSTREAM_METADATA=<json>``, ``UPSTREAM_METADATA_SUMMARY=<json>``); the
outer worker validates each line against
``NeutralFrameMetadata`` and pushes through the
``BoundedBridgeQueue`` / ``BridgeQueueMetadataSource`` seam. The container
never calls UpStream APIs — the **webhook secret never enters the
container**.

This file is pyds-free / gi-free: it only orchestrates the container and
the temp work dir. Importable on the Mac without DeepStream installed.
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
from app.runtimes.deepstream_runtime.inference import (
    build_pgie_config,
    build_tracker_config,
    resolve_model_dir,
)

_log = logging.getLogger("upstream_processor.deepstream.meta")

# Path to the in-container app file. ``Path(__file__).parent`` resolves
# both in editable install (uv sync) and in a wheel — the runtime app file
# is shipped as a regular package file.
_PYDS_APP_PATH = Path(__file__).resolve().parent / "pyds_metadata_app.py"


def build_metadata_docker_command(
    *,
    work_dir: str,
    image: str,
    network: str,
    probe_timeout_seconds: float,
    container_name: str,
    sample_limit: int,
    stream_session_id: str | None,
    rtsp_url: str,
    tracker_enabled: bool = False,
    tracker_lib_path: str = "",
    tracker_width: int = 960,
    tracker_height: int = 544,
) -> list[str]:
    """Docker argv for one bounded metadata- or tracker-probe run. Pure /
    side-effect free so it is unit-testable without Docker. Mirrors the
    proven manual P2 invocation (``--gpus all``, ``--network host``, work
    dir bind-mounted at the same path so the app's absolute config paths
    resolve inside the container) and runs the in-container app with
    ``python3 -B`` so no ``__pycache__`` files land in the host work dir.

    When ``tracker_enabled`` is set, the in-container app gets the extra
    ``--tracker-config-path`` / ``--tracker-lib-path`` / ``--tracker-width``
    / ``--tracker-height`` args so it splices ``nvtracker`` into the
    pipeline and attaches the pad probe at the tracker src pad."""
    # Note: ``RTSP_URL`` env passes the URL to the in-container app; the
    # in-container app redacts it before any printing.
    tracker_args = ""
    if tracker_enabled:
        tracker_args = (
            f"--tracker-config-path {work_dir}/tracker_config.yml "
            f"--tracker-lib-path {tracker_lib_path} "
            f"--tracker-width {int(tracker_width)} "
            f"--tracker-height {int(tracker_height)} "
        )
    inner = (
        f"timeout -s INT {int(probe_timeout_seconds)} "
        f"python3 -B {work_dir}/pyds_metadata_app.py "
        f"--pgie-config {work_dir}/pgie.txt "
        f"--seconds {int(probe_timeout_seconds)} "
        f"--sample-limit {int(sample_limit)} "
        f"--stream-session-id {stream_session_id or ''} "
        f"{tracker_args}"
        "2>&1"
    )
    argv: list[str] = [
        "docker", "run", "--rm", "--name", container_name,
        "--gpus", "all", "--network", network,
        "-v", f"{work_dir}:{work_dir}",
        "-e", "PYTHONDONTWRITEBYTECODE=1",
        "-e", f"RTSP_URL={rtsp_url}",
        image, "bash", "-lc", inner,
    ]  # fmt: skip
    return argv


class DeepStreamMetadataProcessManager:
    """Docker-backed ``ProcessManager`` for the pyds metadata probe.

    Owns a per-run temp work dir holding the pgie config + a copy of the
    in-container pyds app, both bind-mounted into the DS 8.0 + pyds
    container. Streams stdout, supports bounded graceful stop +
    SIGKILL-after-timeout. The raw RTSP URL is only in the subprocess
    argv / temp pgie config; logs use the redacted form. The temp dir is
    removed on stop unless ``keep_temp_files``."""

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
        sample_limit: int,
        stream_session_id: str | None,
        width: int,
        height: int,
        tracker_enabled: bool = False,
        tracker_algorithm: str = "iou",
        tracker_lib_path: str = "",
        tracker_width: int = 960,
        tracker_height: int = 544,
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
        self._sample_limit = sample_limit
        self._stream_session_id = stream_session_id
        self._width = width
        self._height = height
        self._tracker_enabled = tracker_enabled
        self._tracker_algorithm = tracker_algorithm
        self._tracker_lib_path = tracker_lib_path
        self._tracker_width = tracker_width
        self._tracker_height = tracker_height
        # ``upstream-dst-`` distinguishes tracker-probe containers from
        # metadata-probe ones (``upstream-dsm-``) in `docker ps`.
        prefix = "upstream-dst" if tracker_enabled else "upstream-dsm"
        self._container = f"{prefix}-{uuid4().hex[:12]}"
        self._work_dir: Path | None = None
        self._proc: asyncio.subprocess.Process | None = None
        self._returncode: int | None = None
        # Idempotent-stop guard. Set on the first ``stop()`` so a second
        # call (e.g., from the worker's outer ``except CancelledError``
        # handler running after our shielded ``finally``) short-circuits.
        self._stopped = False

    # --- temp work dir ---------------------------------------------------

    def _prepare_work_dir(self) -> Path:
        model_dir = resolve_model_dir(self._model)  # may raise ValueError
        if self._tracker_enabled:
            # Validate the tracker algorithm up-front so a typo posts
            # ``failed`` cleanly rather than failing inside the container.
            build_tracker_config(self._tracker_algorithm)
        base = self._output_base_dir.strip() or None
        if base is not None:
            Path(base).mkdir(parents=True, exist_ok=True)
        prefix = "upstream-p3-tracker." if self._tracker_enabled else "upstream-p2-meta."
        work = Path(tempfile.mkdtemp(prefix=prefix, dir=base))
        pgie_path = work / "pgie.txt"
        engine_path = work / "trafficcamnet_b1_gpu0_int8.engine"
        pgie_path.write_text(
            build_pgie_config(model_dir=model_dir, engine_path=str(engine_path)),
            encoding="utf-8",
        )
        if self._tracker_enabled:
            (work / "tracker_config.yml").write_text(
                build_tracker_config(self._tracker_algorithm),
                encoding="utf-8",
            )
        # Copy (not symlink) the pyds app into the work dir so the
        # container, which bind-mounts only the work dir, can read it.
        shutil.copy(_PYDS_APP_PATH, work / "pyds_metadata_app.py")
        return work

    def _cleanup_work_dir(self) -> None:
        """Idempotent: removes the temp work dir if one is present and
        ``keep_temp_files`` is not set. Safe to call multiple times — the
        second call is a no-op because we null ``_work_dir`` after rmtree.
        Never references the RTSP URL."""
        if self._work_dir is None or self._keep_temp_files:
            return
        with contextlib.suppress(OSError):
            shutil.rmtree(self._work_dir, ignore_errors=True)
        # Mark cleaned so a second call is a no-op (idempotency guard).
        self._work_dir = None

    # --- ProcessManager protocol ---------------------------------------

    def command(self) -> list[str]:
        assert self._work_dir is not None, "command() called before start()"
        return build_metadata_docker_command(
            work_dir=str(self._work_dir),
            image=self._image,
            network=self._network,
            probe_timeout_seconds=self._probe_timeout,
            container_name=self._container,
            sample_limit=self._sample_limit,
            stream_session_id=self._stream_session_id,
            rtsp_url=self._rtsp_url,
            tracker_enabled=self._tracker_enabled,
            tracker_lib_path=self._tracker_lib_path,
            tracker_width=self._tracker_width,
            tracker_height=self._tracker_height,
        )

    async def start(self) -> None:
        self._work_dir = self._prepare_work_dir()
        argv = self.command()
        _log.info(
            "deepstream metadata start: image=%s network=%s model=%s tracker=%s url=%s",
            self._image,
            self._network,
            self._model,
            "on" if self._tracker_enabled else "off",
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
        """Idempotent: ``docker stop`` then SIGTERM/SIGKILL the subprocess
        if still alive, then clean the temp work dir. A second call is a
        no-op (cleanup ran on the first call; the container is gone)."""
        if self._stopped:
            # Defensive: still clean the work dir in case the first call
            # was cancelled before reaching the ``finally``. ``_cleanup_work_dir``
            # itself is idempotent.
            self._cleanup_work_dir()
            return
        self._stopped = True
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
                            "metadata subprocess did not exit in %.0fs; killing",
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
