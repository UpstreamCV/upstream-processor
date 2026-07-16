"""CrowdCount counting-container subprocess manager.

Runs the CrowdCount single-session counting container as a managed asyncio
subprocess, mirroring ``DeepStreamProcessManager`` with two deliberate
differences:

- **No ``timeout`` wrapper.** The counter is a long-running worker that ends
  only on stream loss or ``docker stop`` — not a bounded probe.
- **Session inputs travel in a per-run ``--env-file``** (0600 temp file,
  removed on stop) instead of argv, so the internal RTSP URL and the
  credentials never appear in host ``ps`` output. Note the limit of this:
  the env-file keeps them out of argv only — the RESOLVED container env is
  still visible to anyone who can run ``docker inspect`` on the box.

The container reuses the existing CrowdCount prod counting code: its
entrypoint reads the ``CC_*`` variables below, builds a
``PeopleCounterProcessor`` against ``CC_RTSP_URL``, and posts
counts-of-record to ``CC_API_BASE_URL`` via its own ``CrowdCountAPIClient``.

Credential delivery (decided 2026-07-06): Upstream does not hold per-device
CrowdCount passwords, so ``CC_DEVICE_PASSWORD`` is never forwarded. Instead
``CC_SERVICE_SECRET`` (from ``UPSTREAM_CROWDCOUNT_SERVICE_SECRET``) lets the
container post on behalf of ``CC_DEVICE_ID`` — the camera's external_id —
using trusted-backend service auth. With neither, the client disables itself
and counts without posting (probe mode).
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import tempfile
from uuid import uuid4

from app.redact import redact_rtsp

_log = logging.getLogger("upstream_processor.crowdcount.proc")

# Env-var contract with the CrowdCount single-session container entrypoint.
# Keep in lock-step with the CrowdCount repo's worker entrypoint milestone.
ENV_RTSP_URL = "CC_RTSP_URL"
ENV_DEVICE_ID = "CC_DEVICE_ID"
ENV_API_BASE_URL = "CC_API_BASE_URL"
ENV_FRAME_WIDTH = "CC_FRAME_WIDTH"
ENV_FRAME_HEIGHT = "CC_FRAME_HEIGHT"
ENV_FRAME_FPS = "CC_FRAME_FPS"
ENV_SERVICE_SECRET = "CC_SERVICE_SECRET"
ENV_CAMERA_ID = "CC_CAMERA_ID"
ENV_SESSION_ID = "CC_SESSION_ID"
ENV_FANOUT_URL = "CC_FANOUT_URL"
ENV_FANOUT_SECRET = "CC_FANOUT_SECRET"
ENV_PROCESSOR_ID = "CC_PROCESSOR_ID"
ENV_PACKAGE_ID = "CC_PACKAGE_ID"


def build_docker_command(
    *,
    image: str,
    network: str,
    env_file: str,
    container_name: str,
    gpus: str = "all",
) -> list[str]:
    """The docker argv. Session inputs (including the RTSP URL) are in the
    env-file, never here — argv is visible in host ``ps``/``docker inspect``.
    The image's own entrypoint/CMD runs the single-session counter."""
    return [
        "docker", "run", "--rm", "--name", container_name,
        "--gpus", gpus, "--network", network,
        "--env-file", env_file, image,
    ]  # fmt: skip


class CrowdCountProcessManager:
    """Real Docker-backed implementation of ``ProcessManager``
    (``app.runtimes.deepstream_runtime.process_manager.ProcessManager``)."""

    def __init__(
        self,
        *,
        rtsp_url: str,
        image: str,
        network: str,
        stop_timeout_seconds: float,
        api_base_url: str,
        device_id: str = "",
        width: int | None = None,
        height: int | None = None,
        fps: int | None = None,
        gpus: str = "all",
        service_secret: str = "",
        camera_id: str = "",
        session_id: str = "",
        fanout_url: str = "",
        fanout_secret: str = "",
        processor_id: str = "",
        package_id: str = "",
    ) -> None:
        self._rtsp_url = rtsp_url
        self._image = image
        self._network = network
        self._stop_timeout = stop_timeout_seconds
        self._api_base_url = api_base_url
        self._device_id = device_id
        self._width = width
        self._height = height
        self._fps = fps
        self._gpus = gpus
        self._service_secret = service_secret
        self._camera_id = camera_id
        self._session_id = session_id
        self._fanout_url = fanout_url
        self._fanout_secret = fanout_secret
        self._processor_id = processor_id
        self._package_id = package_id
        self._container = f"upstream-cc-{uuid4().hex[:12]}"
        self._env_file: str | None = None
        self._proc: asyncio.subprocess.Process | None = None
        self._returncode: int | None = None

    # --- env-file lifecycle -----------------------------------------------

    def _ensure_env_file(self) -> str:
        """Write the per-run env-file (0600 via mkstemp) if not yet written."""
        if self._env_file is not None:
            return self._env_file
        lines = [
            f"{ENV_RTSP_URL}={self._rtsp_url}",
            f"{ENV_DEVICE_ID}={self._device_id}",
            f"{ENV_API_BASE_URL}={self._api_base_url}",
        ]
        if self._width is not None:
            lines.append(f"{ENV_FRAME_WIDTH}={self._width}")
        if self._height is not None:
            lines.append(f"{ENV_FRAME_HEIGHT}={self._height}")
        if self._fps is not None:
            lines.append(f"{ENV_FRAME_FPS}={self._fps}")
        if self._service_secret:
            lines.append(f"{ENV_SERVICE_SECRET}={self._service_secret}")
        if self._camera_id:
            lines.append(f"{ENV_CAMERA_ID}={self._camera_id}")
        if self._session_id:
            lines.append(f"{ENV_SESSION_ID}={self._session_id}")
        if self._fanout_url:
            lines.append(f"{ENV_FANOUT_URL}={self._fanout_url}")
        if self._fanout_secret:
            lines.append(f"{ENV_FANOUT_SECRET}={self._fanout_secret}")
        # Stamped onto the container's fanout envelopes so an overlay frame
        # says which processor and package produced it.
        if self._processor_id:
            lines.append(f"{ENV_PROCESSOR_ID}={self._processor_id}")
        if self._package_id:
            lines.append(f"{ENV_PACKAGE_ID}={self._package_id}")
        fd, path = tempfile.mkstemp(prefix="upstream-cc-env-", text=True)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        self._env_file = path
        return path

    def _cleanup_env_file(self) -> None:
        if self._env_file is None:
            return
        with contextlib.suppress(OSError):
            os.unlink(self._env_file)
        self._env_file = None

    # --- ProcessManager surface ---------------------------------------------

    def command(self) -> list[str]:
        return build_docker_command(
            image=self._image,
            network=self._network,
            env_file=self._ensure_env_file(),
            container_name=self._container,
            gpus=self._gpus,
        )

    async def start(self) -> None:
        argv = self.command()
        _log.info(
            "crowdcount counter start: image=%s network=%s gpus=%s url=%s",
            self._image,
            self._network,
            self._gpus,
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
            if self._proc is None:
                return
            # Ask the container to stop. PID1 (the CrowdCount single-session
            # entrypoint) gets SIGTERM and must flush pending count posts +
            # post device offline within the grace, then docker SIGKILLs.
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
                        "counter subprocess did not exit in %.0fs; killing",
                        self._stop_timeout,
                    )
                    with contextlib.suppress(ProcessLookupError):
                        self._proc.kill()
                    # Bound the post-kill reap; an unbounded wait() can hang
                    # teardown if the killed docker-run client stalls on
                    # container/GPU cleanup (same rationale as the DeepStream
                    # process managers).
                    try:
                        await asyncio.wait_for(self._proc.wait(), timeout=self._stop_timeout)
                    except TimeoutError:
                        _log.warning(
                            "counter subprocess did not reap after kill; "
                            "abandoning wait so teardown can proceed",
                        )
            self._returncode = self._proc.returncode
        finally:
            self._cleanup_env_file()

    @property
    def returncode(self) -> int | None:
        return self._returncode
