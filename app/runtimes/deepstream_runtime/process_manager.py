"""DeepStream/GStreamer decode subprocess manager.

Runs the *exact* pipeline proven in M8.2A
(``scripts/probe_deepstream_rtsp.sh``) as a managed asyncio subprocess:

    docker run --rm --name <uniq> --gpus all --network <net> <image> \
      bash -lc "timeout -s INT <probe_timeout> gst-launch-1.0 -e -v \
        rtspsrc location='<URL>' protocols=<transport> latency=200 ! \
        rtph264depay ! h264parse ! nvv4l2decoder ! nvvideoconvert ! \
        fpsdisplaysink video-sink=fakesink text-overlay=false sync=false"

Decode only — no inference, no observations. The RTSP URL is passed ONLY
inside the subprocess argv and is never logged (logs use the redacted form).
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from typing import Protocol
from uuid import uuid4

from app.redact import redact_rtsp

_log = logging.getLogger("upstream_processor.deepstream.proc")


class ProcessManager(Protocol):
    """What the DeepStream worker needs from a decode subprocess.

    A fake implementing this is used in unit tests (no Docker/GPU)."""

    async def start(self) -> None: ...
    async def readline(self) -> str | None: ...
    async def stop(self) -> None: ...
    @property
    def returncode(self) -> int | None: ...


def build_docker_command(
    *,
    rtsp_url: str,
    image: str,
    network: str,
    rtsp_transport: str,
    probe_timeout_seconds: float,
    container_name: str,
    gpus: str = "all",
) -> list[str]:
    """The docker argv. The RTSP URL appears only here (subprocess), never
    in logs. Mirrors the proven M8.2A pipeline. ``gpus`` is the literal
    ``docker run --gpus`` value ("all" or e.g. "device=1" to pin a card)."""
    pipeline = (
        f'rtspsrc location="{rtsp_url}" protocols={rtsp_transport} latency=200 ! '
        "rtph264depay ! h264parse ! nvv4l2decoder ! nvvideoconvert ! "
        "fpsdisplaysink video-sink=fakesink text-overlay=false sync=false"
    )
    inner = f"timeout -s INT {int(probe_timeout_seconds)} gst-launch-1.0 -e -v {pipeline} 2>&1"
    return [
        "docker", "run", "--rm", "--name", container_name,
        "--gpus", gpus, "--network", network, image,
        "bash", "-lc", inner,
    ]  # fmt: skip


class DeepStreamProcessManager:
    """Real Docker-backed implementation of ``ProcessManager``."""

    def __init__(
        self,
        *,
        rtsp_url: str,
        image: str,
        network: str,
        rtsp_transport: str,
        probe_timeout_seconds: float,
        stop_timeout_seconds: float,
        gpus: str = "all",
    ) -> None:
        self._rtsp_url = rtsp_url
        self._image = image
        self._network = network
        self._rtsp_transport = rtsp_transport
        self._probe_timeout = probe_timeout_seconds
        self._stop_timeout = stop_timeout_seconds
        self._gpus = gpus
        self._container = f"upstream-ds-{uuid4().hex[:12]}"
        self._proc: asyncio.subprocess.Process | None = None
        self._returncode: int | None = None

    def command(self) -> list[str]:
        return build_docker_command(
            rtsp_url=self._rtsp_url,
            image=self._image,
            network=self._network,
            rtsp_transport=self._rtsp_transport,
            probe_timeout_seconds=self._probe_timeout,
            container_name=self._container,
            gpus=self._gpus,
        )

    async def start(self) -> None:
        argv = self.command()
        _log.info(
            "deepstream decode start: image=%s network=%s transport=%s url=%s",
            self._image,
            self._network,
            self._rtsp_transport,
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
        if self._proc is None:
            return
        # Ask the container to stop (PID1 SIGTERM, then SIGKILL after grace).
        try:
            stopper = await asyncio.create_subprocess_exec(
                "docker",
                "stop",
                "--time",
                str(int(self._stop_timeout)),
                self._container,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
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
                    "decode subprocess did not exit in %.0fs; killing",
                    self._stop_timeout,
                )
                with contextlib.suppress(ProcessLookupError):
                    self._proc.kill()
                # Bound the post-kill reap; an unbounded wait() can hang
                # teardown (and the supervisor that awaits it) if the killed
                # docker-run client stalls on container/GPU cleanup. Same
                # rationale as metadata_probe.stop().
                try:
                    await asyncio.wait_for(self._proc.wait(), timeout=self._stop_timeout)
                except TimeoutError:
                    _log.warning(
                        "decode subprocess did not reap after kill; "
                        "abandoning wait so teardown can proceed",
                    )
        self._returncode = self._proc.returncode

    @property
    def returncode(self) -> int | None:
        return self._returncode
