"""Reconcile one mock worker per eligible session.

Single-writer model: only ``run()`` mutates ``_workers`` (asyncio is
single-threaded and cooperative), so no locks are needed. Each tick:
reap finished tasks, start workers for newly-eligible sessions, stop workers
for sessions that dropped out, never duplicate, quarantine crash-loopers.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from uuid import UUID

from app.api_client.base import ProcessorApi
from app.backoff import Backoff, wait_or_event
from app.config import Settings
from app.errors import (
    ApiAuthError,
    ApiError,
    ApiValidationError,
    PackageResolutionError,
)
from app.observations.schemas import EligibleSession
from app.observations.writer import ObservationWriter
from app.packages.resolver import PackageSpec, resolve
from app.runtimes.base import Runtime
from app.runtimes.deepstream_runtime.worker import DeepStreamWorker
from app.runtimes.mock_runtime.worker import MockWorker
from app.supervisor.processor_status import build_status_request
from app.supervisor.session_poller import SessionPoller

_log = logging.getLogger("upstream_processor.manager")


def effective_package_id(settings: Settings, session: EligibleSession) -> str:
    """Dev/test override: ``UPSTREAM_FORCE_PACKAGE_ID`` wins over the session's
    own ``package_id`` (so a mock-defaulted session can be routed to the
    DeepStream decode probe without mutating production camera config)."""
    return settings.force_package_id or session.package_id


def build_worker(
    *,
    session: EligibleSession,
    spec: PackageSpec,
    settings: Settings,
    api: ProcessorApi,
    writer: ObservationWriter,
) -> Runtime:
    """Pick the runtime worker from the resolved package's runtime. Pure +
    side-effect free (constructing a DeepStreamWorker does NOT touch Docker —
    the subprocess starts in ``run()``), so this is unit-testable."""
    if spec.runtime == "deepstream":
        # P4-B threads the same ObservationWriter used by MockWorker so the
        # analytics mode can emit ``processor.line_crossing`` observations.
        # Non-analytics DeepStream modes ignore the writer.
        return DeepStreamWorker(
            session=session,
            settings=settings,
            api=api,
            package=spec,
            writer=writer,
        )
    return MockWorker(
        session=session,
        settings=settings,
        api=api,
        writer=writer,
        package=spec,
    )


@dataclass
class _Handle:
    task: asyncio.Task[None]
    worker: Runtime


@dataclass
class _RestartBook:
    starts: list[float] = field(default_factory=list)


class WorkerManager:
    def __init__(
        self,
        *,
        poller: SessionPoller,
        api: ProcessorApi,
        settings: Settings,
        shutdown: asyncio.Event,
    ) -> None:
        self._poller = poller
        self._api = api
        self._settings = settings
        self._shutdown = shutdown
        self._writer = ObservationWriter(api, settings.processor_id)
        self._workers: dict[UUID, _Handle] = {}
        self._quarantine: dict[UUID, float] = {}
        self._restarts: dict[UUID, _RestartBook] = {}
        self._bg: set[asyncio.Task[None]] = set()
        self._poll_backoff = Backoff(
            initial=settings.backoff_initial_seconds,
            factor=settings.backoff_factor,
            maximum=settings.backoff_max_seconds,
            jitter=settings.backoff_jitter,
        )

    def active_count(self) -> int:
        return len(self._workers)

    # --- main loop --------------------------------------------------------

    async def run(self) -> None:
        while not self._shutdown.is_set():
            try:
                sessions = await self._poller.fetch_eligible()
            except ApiAuthError:
                raise  # fatal config — bubble to main, exit non-zero
            except ApiValidationError:
                raise  # a 422 on a GET means our client is wrong — fatal
            except ApiError as exc:
                # Everything else (session-gone / 5xx / timeout / contract /
                # rate-limit) is transient for the poll loop — survive it.
                _log.warning("eligible-sessions fetch failed (transient): %s", exc)
                if await wait_or_event(self._shutdown, self._poll_backoff.next_delay()):
                    break
                continue

            self._poll_backoff.reset()
            self._reap()
            desired = {s.session_id: s for s in sessions}
            self._start_new(desired)
            await self._stop_removed(desired)
            _log.info(
                "reconcile: eligible=%d active=%d",
                len(desired),
                self.active_count(),
            )
            if await wait_or_event(self._shutdown, self._settings.poll_interval_seconds):
                break

        await self._shutdown_all()

    # --- reconcile steps --------------------------------------------------

    def _reap(self) -> None:
        for sid, handle in list(self._workers.items()):
            if not handle.task.done():
                continue
            if not handle.task.cancelled():
                exc = handle.task.exception()
                if exc is not None:
                    _log.error("worker for session %s ended in error: %r", sid, exc)
            self._workers.pop(sid, None)

    def _start_new(self, desired: dict[UUID, EligibleSession]) -> None:
        now = time.monotonic()
        window = self._settings.restart_window_seconds
        for sid, session in desired.items():
            if sid in self._workers:
                continue  # never duplicate
            deadline = self._quarantine.get(sid)
            if deadline is not None and now < deadline:
                continue
            book = self._restarts.setdefault(sid, _RestartBook())
            book.starts = [t for t in book.starts if now - t < window]
            if len(book.starts) >= self._settings.max_restarts:
                self._quarantine[sid] = now + window
                book.starts.clear()
                _log.error(
                    "quarantining session %s: >=%d restarts within %.0fs",
                    sid,
                    self._settings.max_restarts,
                    window,
                )
                continue
            pkg_id = effective_package_id(self._settings, session)
            if self._settings.force_package_id and pkg_id != session.package_id:
                _log.info(
                    "package override active for %s: forced=%s (session offered %s)",
                    sid,
                    pkg_id,
                    session.package_id,
                )
            try:
                spec = resolve(pkg_id)
            except PackageResolutionError as exc:
                _log.error("package resolution failed for %s: %s", sid, exc)
                self._spawn_bg(self._post_failed(session, "package_resolution", str(exc)))
                self._quarantine[sid] = now + window
                continue
            self._spawn(sid, session, spec)
            book.starts.append(now)
            self._quarantine.pop(sid, None)

    def _spawn(self, sid: UUID, session: EligibleSession, spec: PackageSpec) -> None:
        worker = build_worker(
            session=session,
            spec=spec,
            settings=self._settings,
            api=self._api,
            writer=self._writer,
        )
        task: asyncio.Task[None] = asyncio.create_task(worker.run())
        self._workers[sid] = _Handle(task=task, worker=worker)

    async def _stop_removed(self, desired: dict[UUID, EligibleSession]) -> None:
        for sid in list(self._workers):
            if sid not in desired:
                await self._graceful_stop(sid)

    async def _graceful_stop(self, sid: UUID) -> None:
        handle = self._workers.get(sid)
        if handle is None:
            return
        handle.worker.request_stop()
        budget = self._settings.final_status_timeout_seconds + 1.0
        try:
            await asyncio.wait_for(handle.task, timeout=budget)
        except TimeoutError:
            _log.warning("worker %s did not stop in %.1fs; cancelling", sid, budget)
            handle.task.cancel()
            await asyncio.gather(handle.task, return_exceptions=True)
        finally:
            self._workers.pop(sid, None)

    async def _shutdown_all(self) -> None:
        if not self._workers:
            return
        _log.info("shutting down %d worker(s)", len(self._workers))
        for handle in self._workers.values():
            handle.worker.request_stop()

        async def stop_one(h: _Handle) -> None:
            try:
                await asyncio.wait_for(h.task, timeout=self._settings.final_status_timeout_seconds)
            except (TimeoutError, asyncio.CancelledError):
                h.task.cancel()
                await asyncio.gather(h.task, return_exceptions=True)

        handles = list(self._workers.values())
        try:
            await asyncio.wait_for(
                asyncio.gather(*(stop_one(h) for h in handles), return_exceptions=True),
                timeout=self._settings.overall_shutdown_timeout_seconds,
            )
        except TimeoutError:
            _log.warning("overall shutdown timed out; cancelling remaining workers")
            for h in handles:
                h.task.cancel()
            await asyncio.gather(*(h.task for h in handles), return_exceptions=True)
        self._workers.clear()

    # --- helpers ----------------------------------------------------------

    def _spawn_bg(self, coro: object) -> None:
        task: asyncio.Task[None] = asyncio.create_task(coro)  # type: ignore[arg-type]
        self._bg.add(task)
        task.add_done_callback(self._bg.discard)

    async def _post_failed(self, session: EligibleSession, code: str, message: str) -> None:
        try:
            await self._api.post_processor_status(
                build_status_request(
                    session=session,
                    processor_id=self._settings.processor_id,
                    status="failed",
                    error_code=code,
                    error_message=message,
                )
            )
        except Exception as exc:  # best-effort one-shot
            _log.info("could not post failed for %s: %s", session.session_id, exc)
