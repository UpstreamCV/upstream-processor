"""The mock per-session worker.

Proves the handoff lifecycle with zero video work. It treats
``session.internal_rtsp_url`` as opaque metadata and never reads it (M8.1 is
HTTP-only; a dedicated test enforces this).

Status order: starting -> attached -> processing (repeated) -> stopped, or
-> failed on a contract/internal error. A 404 (session gone) ends the worker
quietly with no terminal post (the session no longer exists to post against).
Transient API errors are absorbed with backoff inside the pulses so a blip
never tears the worker down.
"""

from __future__ import annotations

import asyncio
import logging
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
from app.observations.schemas import EligibleSession, ObservationType
from app.observations.writer import ObservationWriter
from app.packages.resolver import PackageSpec
from app.runtimes.base import Runtime, WorkerState
from app.runtimes.mock_runtime import fake_observations as fake
from app.supervisor.processor_status import build_status_request

_log = logging.getLogger("upstream_processor.worker")

_TERMINAL_STATUS_ATTEMPTS = 3


class MockWorker(Runtime):
    def __init__(
        self,
        *,
        session: EligibleSession,
        settings: Settings,
        api: ProcessorApi,
        writer: ObservationWriter,
        package: PackageSpec,
    ) -> None:
        self._session = session
        self._settings = settings
        self._api = api
        self._writer = writer
        self._package = package
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

    async def run(self) -> None:
        try:
            self._state = WorkerState.STARTING
            if not await self._post_status_gating("starting"):
                return  # stop requested before we got going; nothing to tear down
            self._state = WorkerState.ATTACHED
            if not await self._post_status_gating("attached"):
                return
            self._state = WorkerState.PROCESSING
            await self._run_pulses()
            self._state = WorkerState.STOPPED
            await self._post_status_best_effort("stopped")
        except ApiSessionGoneError:
            self._state = WorkerState.FAILED
            _log.info("session %s gone; stopping quietly", self.session_id)
        except ApiValidationError as exc:
            self._state = WorkerState.FAILED
            _log.error("contract error for session %s: %s", self.session_id, exc)
            await self._post_status_best_effort(
                "failed", error_code="validation_error", error_message=str(exc)
            )
        except asyncio.CancelledError:
            raise  # hard-shutdown escape hatch — do not swallow
        except Exception as exc:  # never crash the supervisor
            self._state = WorkerState.FAILED
            _log.exception("worker for session %s crashed", self.session_id)
            await self._post_status_best_effort(
                "failed", error_code="worker_error", error_message=repr(exc)
            )

    # --- pulses -----------------------------------------------------------

    async def _run_pulses(self) -> None:
        tasks = [
            asyncio.create_task(self._status_pulse()),
            asyncio.create_task(
                self._observation_pulse(
                    "processor.mock_frame_seen",
                    self._settings.mock_observation_interval_seconds,
                    lambda n: fake.frame_seen_payload(self._processor_id, n),
                )
            ),
            asyncio.create_task(
                self._observation_pulse(
                    "processor.mock_heartbeat",
                    self._settings.heartbeat_interval_seconds,
                    lambda n: fake.heartbeat_payload(self._processor_id, n),
                )
            ),
            asyncio.create_task(
                self._observation_pulse(
                    "processor.mock_detection",
                    self._settings.detection_interval_seconds,
                    lambda n: fake.detection_payload(self._processor_id, n),
                )
            ),
        ]
        try:
            done, _pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)
        finally:
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
        for t in done:
            if not t.cancelled() and t.exception() is not None:
                raise t.exception()  # type: ignore[misc]  # propagate session-gone/validation

    async def _status_pulse(self) -> None:
        while not self._stop.is_set():
            now = now_utc()
            await self._post_status(
                "processing",
                fps_input=float(self._session.approved.fps),
                fps_processed=float(self._session.approved.fps),
                last_frame_at=now,
                last_observation_at=now,
                payload={"runtime": "mock"},
                transient_ok=True,
            )
            if await wait_or_event(self._stop, self._settings.status_interval_seconds):
                return

    async def _observation_pulse(
        self,
        obs_type: ObservationType,
        interval: float,
        payload_fn: Callable[[int], dict[str, Any]],
    ) -> None:
        seq = 0
        while not self._stop.is_set():
            seq += 1
            try:
                await self._writer.emit(
                    stream_session_id=self.session_id,
                    obs_type=obs_type,
                    payload=payload_fn(seq),
                    package_id=self._session.package_id,
                )
                self._backoff.reset()
            except TRANSIENT_API_ERRORS as exc:
                _log.warning(
                    "transient emitting %s for %s: %s",
                    obs_type,
                    self.session_id,
                    exc,
                )
                if await wait_or_event(self._stop, self._backoff.next_delay()):
                    return
                continue
            # ApiSessionGoneError / ApiValidationError propagate to run()
            if await wait_or_event(self._stop, interval):
                return

    # --- status helpers ---------------------------------------------------

    @property
    def _processor_id(self) -> str:
        return self._settings.processor_id

    def _build(self, status: str, **fields: Any) -> Any:
        return build_status_request(
            session=self._session,
            processor_id=self._processor_id,
            status=status,  # type: ignore[arg-type]  # caller passes a valid literal
            **fields,
        )

    async def _post_status(self, status: str, *, transient_ok: bool = False, **fields: Any) -> None:
        req = self._build(status, **fields)
        try:
            await self._api.post_processor_status(req)
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

    async def _post_status_gating(self, status: str) -> bool:
        """Post a gating status (starting/attached), retrying transient errors
        until it lands. Returns False if stop was requested first."""
        while not self._stop.is_set():
            try:
                await self._api.post_processor_status(self._build(status))
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

    async def _post_status_best_effort(
        self,
        status: str,
        *,
        error_code: str | None = None,
        error_message: str | None = None,
    ) -> None:
        """Terminal post (stopped/failed): a few bounded attempts, never
        raises, never loops forever (shutdown must not hang)."""
        for _ in range(_TERMINAL_STATUS_ATTEMPTS):
            try:
                await self._api.post_processor_status(
                    self._build(
                        status,
                        error_code=error_code,
                        error_message=error_message,
                    )
                )
                return
            except ApiSessionGoneError:
                return  # session gone — nothing to post against
            except TRANSIENT_API_ERRORS as exc:
                _log.info(
                    "transient posting terminal status=%s for %s: %s",
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
