"""M8.2B DeepStream decode-probe worker.

Implements the existing ``Runtime`` interface so the supervisor treats it
exactly like the mock worker. It runs the M8.2A-proven GStreamer decode
pipeline as a managed subprocess and maps it onto the M8.0 status lifecycle:

    starting -> attached (first frames decoded) -> processing (fps updates)
             -> stopped (graceful) | failed (process error)

Decode only: **no inference, no tracking, no observations**. The RTSP URL is
never logged (only its redacted form); the secret is never handled here.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from collections.abc import Callable
from typing import Any
from uuid import UUID

from app.api_client.base import ProcessorApi
from app.backoff import Backoff, wait_or_event
from app.clock import now_utc
from app.config import Settings
from app.errors import (
    TRANSIENT_API_ERRORS,
    ApiSessionGoneError,
    ApiValidationError,
)
from app.observations.schemas import EligibleSession
from app.packages.resolver import PackageSpec
from app.redact import redact_rtsp, scrub_rtsp_in_text
from app.runtimes.base import Runtime, WorkerState
from app.runtimes.deepstream_runtime.process_manager import (
    DeepStreamProcessManager,
    ProcessManager,
)
from app.supervisor.processor_status import build_status_request

_log = logging.getLogger("upstream_processor.deepstream.worker")

_TERMINAL_STATUS_ATTEMPTS = 3
_READLINE_POLL_SECONDS = 0.5
# gst -e on SIGINT, or `timeout` self-terminating the bounded probe, are
# graceful endings (124 = timeout, 130 = SIGINT, 137 = SIGKILL after stop).
_CLEAN_EXIT_CODES = frozenset({0, 124, 130, 137})

_FPS_RE = re.compile(
    r"rendered:\s*(\d+),\s*dropped:\s*(\d+),.*?average:\s*([\d.]+)",
    re.IGNORECASE,
)


def _parse_fps(line: str) -> tuple[int, int, float] | None:
    m = _FPS_RE.search(line)
    if m is None:
        return None
    return int(m.group(1)), int(m.group(2)), float(m.group(3))


class DeepStreamWorker(Runtime):
    def __init__(
        self,
        *,
        session: EligibleSession,
        settings: Settings,
        api: ProcessorApi,
        package: PackageSpec,
        pm_factory: Callable[[], ProcessManager] | None = None,
    ) -> None:
        self._session = session
        self._settings = settings
        self._api = api
        self._package = package
        self._pm_factory = pm_factory or self._default_pm_factory
        self._pm: ProcessManager | None = None
        self._stop = asyncio.Event()
        self._state = WorkerState.INIT
        self._backoff = Backoff(
            initial=settings.backoff_initial_seconds,
            factor=settings.backoff_factor,
            maximum=settings.backoff_max_seconds,
            jitter=settings.backoff_jitter,
        )

    # --- Runtime interface ------------------------------------------------

    @property
    def session_id(self) -> UUID:
        return self._session.session_id

    @property
    def state(self) -> WorkerState:
        return self._state

    def request_stop(self) -> None:
        self._stop.set()

    def _default_pm_factory(self) -> ProcessManager:
        return DeepStreamProcessManager(
            rtsp_url=self._session.internal_rtsp_url,
            image=self._settings.deepstream_image,
            network=self._settings.deepstream_docker_network,
            rtsp_transport=self._settings.deepstream_rtsp_transport,
            probe_timeout_seconds=self._settings.deepstream_probe_timeout_seconds,
            stop_timeout_seconds=self._settings.deepstream_process_stop_timeout_seconds,
        )

    async def run(self) -> None:
        try:
            if not self._session.internal_rtsp_url:
                self._state = WorkerState.FAILED
                await self._post_best_effort(
                    "failed",
                    error_code="no_rtsp_url",
                    error_message="eligible session missing internal_rtsp_url",
                )
                return

            self._state = WorkerState.STARTING
            if not await self._post_gating("starting"):
                return
            _log.info(
                "deepstream worker session=%s package=%s url=%s",
                self.session_id,
                self._package.package_id,
                redact_rtsp(self._session.internal_rtsp_url),
            )
            self._pm = self._pm_factory()
            await self._pm.start()

            attached = False
            last_proc_post = 0.0
            last_fps: float | None = None
            last_err: str | None = None

            while not self._stop.is_set():
                try:
                    line = await asyncio.wait_for(
                        self._pm.readline(), timeout=_READLINE_POLL_SECONDS
                    )
                except TimeoutError:
                    continue
                if line is None:
                    break  # decode process ended
                if "ERROR" in line:
                    last_err = scrub_rtsp_in_text(line)[:480]
                parsed = _parse_fps(line)
                if parsed is None:
                    continue
                frames, dropped, fps = parsed
                last_fps = fps
                if not attached and frames > 0:
                    attached = True
                    self._state = WorkerState.ATTACHED
                    if not await self._post_gating("attached", fps_processed=fps):
                        break
                    self._state = WorkerState.PROCESSING
                if attached:
                    now = time.monotonic()
                    if now - last_proc_post >= self._settings.status_interval_seconds:
                        last_proc_post = now
                        await self._post(
                            "processing",
                            transient_ok=True,
                            fps_processed=fps,
                            last_frame_at=now_utc(),
                            payload={
                                "runtime": "deepstream",
                                "frames": frames,
                                "dropped": dropped,
                                "decoder": "nvv4l2decoder",
                            },
                        )

            await self._safe_stop_pm()
            rc = self._pm.returncode if self._pm else None
            graceful = self._stop.is_set() or rc in _CLEAN_EXIT_CODES
            if graceful:
                self._state = WorkerState.STOPPED
                await self._post_best_effort("stopped", fps_processed=last_fps)
            else:
                self._state = WorkerState.FAILED
                await self._post_best_effort(
                    "failed",
                    error_code="deepstream_exit",
                    error_message=(
                        f"decode process exited rc={rc}; last_error={last_err or 'n/a'}"
                    ),
                )
        except ApiSessionGoneError:
            self._state = WorkerState.FAILED
            _log.info("session %s gone; stopping quietly", self.session_id)
            await self._safe_stop_pm()
        except ApiValidationError as exc:
            self._state = WorkerState.FAILED
            _log.error("contract error for session %s: %s", self.session_id, exc)
            await self._safe_stop_pm()
            await self._post_best_effort(
                "failed", error_code="validation_error", error_message=str(exc)
            )
        except asyncio.CancelledError:
            await self._safe_stop_pm()
            raise
        except Exception as exc:  # never crash the supervisor
            self._state = WorkerState.FAILED
            _log.exception("deepstream worker session=%s crashed", self.session_id)
            await self._safe_stop_pm()
            await self._post_best_effort(
                "failed", error_code="worker_error", error_message=repr(exc)
            )

    # --- helpers ----------------------------------------------------------

    async def _safe_stop_pm(self) -> None:
        if self._pm is None:
            return
        try:
            await self._pm.stop()
        except Exception as exc:  # stop must never raise out of teardown
            _log.warning("error stopping decode process: %s", exc)

    def _build(self, status: str, **fields: Any) -> Any:
        return build_status_request(
            session=self._session,
            processor_id=self._settings.processor_id,
            status=status,  # type: ignore[arg-type]  # caller passes a valid literal
            runtime=self._package.runtime,
            package_id=self._package.package_id,
            **fields,
        )

    async def _post(self, status: str, *, transient_ok: bool = False, **fields: Any) -> None:
        try:
            await self._api.post_processor_status(self._build(status, **fields))
            self._backoff.reset()
        except TRANSIENT_API_ERRORS as exc:
            if not transient_ok:
                raise
            _log.warning(
                "transient posting status=%s for %s: %s",
                status,
                self.session_id,
                exc,
            )

    async def _post_gating(self, status: str, **fields: Any) -> bool:
        while not self._stop.is_set():
            try:
                await self._api.post_processor_status(self._build(status, **fields))
                self._backoff.reset()
                return True
            except TRANSIENT_API_ERRORS as exc:
                _log.warning(
                    "transient posting status=%s for %s: %s",
                    status,
                    self.session_id,
                    exc,
                )
                if await wait_or_event(self._stop, self._backoff.next_delay()):
                    return False
        return False

    async def _post_best_effort(
        self,
        status: str,
        *,
        fps_processed: float | None = None,
        error_code: str | None = None,
        error_message: str | None = None,
    ) -> None:
        for _ in range(_TERMINAL_STATUS_ATTEMPTS):
            try:
                await self._api.post_processor_status(
                    self._build(
                        status,
                        fps_processed=fps_processed,
                        error_code=error_code,
                        error_message=error_message,
                    )
                )
                return
            except ApiSessionGoneError:
                return
            except TRANSIENT_API_ERRORS as exc:
                _log.info(
                    "transient terminal status=%s for %s: %s",
                    status,
                    self.session_id,
                    exc,
                )
                await asyncio.sleep(0.1)
            except Exception as exc:  # terminal post must not raise
                _log.warning(
                    "giving up terminal status=%s for %s: %s",
                    status,
                    self.session_id,
                    exc,
                )
                return
