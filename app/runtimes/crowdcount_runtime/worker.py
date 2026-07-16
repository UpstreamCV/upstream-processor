"""The CrowdCount per-session worker (status-only v1).

Supervises one CrowdCount counting container per eligible session,
mirroring ``DeepStreamWorker``'s docker-backed lifecycle: guard the RTSP
URL, post ``starting``, launch the container, attach on the container's
ready marker, heartbeat ``processing``, and map the container's exit to a
terminal ``stopped``/``failed``.

Division of labor (migration decision 2026-07-05/06): the container owns
counting, tracking, the counts-of-record posts to crowdcount-server, and
the device online/offline posts — all existing CrowdCount prod code.
Upstream owns only the container lifecycle and processor status here.
``processor.line_crossing`` observation emission alongside the count path
is a later milestone; the ``ObservationWriter`` is accepted now so the
``build_worker`` seam doesn't change when it lands.

Status order: starting -> attached -> processing (repeated) -> stopped, or
-> failed on a contract/internal error or non-graceful container exit.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from typing import Any
from uuid import UUID

from app.api_client.base import ProcessorApi
from app.backoff import Backoff, wait_or_event
from app.config import Settings
from app.errors import (
    TRANSIENT_API_ERRORS,
    ApiSessionGoneError,
    ApiValidationError,
)
from app.observations.schemas import EligibleSession
from app.observations.writer import ObservationWriter
from app.packages.resolver import PackageSpec
from app.redact import redact_rtsp, scrub_rtsp_in_text
from app.runtimes.base import Runtime, WorkerState
from app.runtimes.crowdcount_runtime.process_manager import CrowdCountProcessManager
from app.runtimes.deepstream_runtime.process_manager import ProcessManager
from app.supervisor.processor_status import build_status_request

_log = logging.getLogger("upstream_processor.crowdcount.worker")

_TERMINAL_STATUS_ATTEMPTS = 3
_READLINE_POLL_SECONDS = 0.5
# Attach evidence on the container's stdout. ``CROWDCOUNT_READY`` is the
# canonical marker for the single-session entrypoint; ``Worker ready`` is
# what the existing CrowdCount prod processor already prints when its
# counting loop is up (``people_counter_processor.py``), accepted so the
# unmodified prod code attaches without a CrowdCount-side change.
_READY_MARKERS = ("CROWDCOUNT_READY", "Worker ready")
# 143 = SIGTERM (docker stop grace), 137 = SIGKILL after the grace — both
# are ordinary endings for a stop we requested.
_CLEAN_EXIT_CODES = frozenset({0, 137, 143})


class CrowdCountWorker(Runtime):
    def __init__(
        self,
        *,
        session: EligibleSession,
        settings: Settings,
        api: ProcessorApi,
        package: PackageSpec,
        writer: ObservationWriter | None = None,
        pm_factory: Callable[[], ProcessManager] | None = None,
    ) -> None:
        self._session = session
        self._settings = settings
        self._api = api
        self._package = package
        self._writer = writer  # unused in v1 (observation emission is a later milestone)
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
        return CrowdCountProcessManager(
            rtsp_url=self._session.internal_rtsp_url,
            image=self._package.container_image or self._settings.crowdcount_image,
            network=self._settings.crowdcount_docker_network,
            stop_timeout_seconds=self._settings.crowdcount_process_stop_timeout_seconds,
            api_base_url=self._settings.crowdcount_api_base_url,
            # The camera's external_id IS the CrowdCount device_id (set at
            # registration). With a service secret configured, the container
            # posts counts on behalf of this device; missing either the id or
            # the secret leaves the CrowdCountAPIClient disabled (probe:
            # counts without posting).
            device_id=self._session.external_id or "",
            service_secret=(
                self._settings.crowdcount_service_secret.get_secret_value()
                if self._settings.crowdcount_service_secret
                else ""
            ),
            width=self._session.approved.width,
            height=self._session.approved.height,
            fps=self._session.approved.fps,
            gpus=self._settings.crowdcount_gpus,
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
                "crowdcount worker session=%s package=%s url=%s",
                self.session_id,
                self._package.package_id,
                redact_rtsp(self._session.internal_rtsp_url),
            )
            self._pm = self._pm_factory()
            await self._pm.start()
            await self._drive_counter()
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
            _log.exception("crowdcount worker session=%s crashed", self.session_id)
            await self._safe_stop_pm()
            await self._post_best_effort(
                "failed", error_code="worker_error", error_message=repr(exc)
            )

    # --- driver -------------------------------------------------------------

    async def _drive_counter(self) -> None:
        assert self._pm is not None
        attached = False
        last_proc_post = 0.0
        last_err: str | None = None

        while not self._stop.is_set():
            try:
                line = await asyncio.wait_for(self._pm.readline(), timeout=_READLINE_POLL_SECONDS)
            except TimeoutError:
                line = ""  # quiet container — fall through to the heartbeat check
            else:
                if line is None:
                    break  # counter process ended
                if "ERROR" in line:
                    last_err = scrub_rtsp_in_text(line)[:480]
                if not attached and any(marker in line for marker in _READY_MARKERS):
                    attached = True
                    self._state = WorkerState.ATTACHED
                    if not await self._post_gating("attached"):
                        break
                    self._state = WorkerState.PROCESSING
            if attached:
                now = time.monotonic()
                if now - last_proc_post >= self._settings.status_interval_seconds:
                    last_proc_post = now
                    await self._post(
                        "processing",
                        transient_ok=True,
                        payload={"runtime": "crowdcount"},
                    )

        await self._safe_stop_pm()
        rc = self._pm.returncode if self._pm else None
        graceful = self._stop.is_set() or rc in _CLEAN_EXIT_CODES
        if graceful:
            self._state = WorkerState.STOPPED
            await self._post_best_effort("stopped")
        else:
            self._state = WorkerState.FAILED
            await self._post_best_effort(
                "failed",
                error_code="crowdcount_exit",
                error_message=(f"counter process exited rc={rc}; last_error={last_err or 'n/a'}"),
            )

    # --- helpers ----------------------------------------------------------

    async def _safe_stop_pm(self) -> None:
        if self._pm is None:
            return
        try:
            await self._pm.stop()
        except Exception as exc:  # stop must never raise out of teardown
            _log.warning("error stopping counter process: %s", exc)

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
        error_code: str | None = None,
        error_message: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> None:
        for _ in range(_TERMINAL_STATUS_ATTEMPTS):
            try:
                await self._api.post_processor_status(
                    self._build(
                        status,
                        error_code=error_code,
                        error_message=error_message,
                        payload=payload,
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
