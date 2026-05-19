from __future__ import annotations

import asyncio
from collections.abc import Callable

import pytest
from app.errors import ApiAuthError
from app.observations.schemas import EligibleSession
from app.supervisor.session_poller import SessionPoller
from app.supervisor.worker_manager import WorkerManager

from tests._fakes import FakeApi
from tests.conftest import eligible_session_dict


def _session(**ov: object) -> EligibleSession:
    return EligibleSession.model_validate(eligible_session_dict(**ov))


def _manager(api: FakeApi, settings):  # type: ignore[no-untyped-def]
    shutdown = asyncio.Event()
    poller = SessionPoller(api, settings.eligible_session_limit)
    mgr = WorkerManager(poller=poller, api=api, settings=settings, shutdown=shutdown)
    return mgr, shutdown


async def _wait_until(pred: Callable[[], bool], limit: float = 2.0) -> bool:
    deadline = asyncio.get_event_loop().time() + limit
    while asyncio.get_event_loop().time() < deadline:
        if pred():
            return True
        await asyncio.sleep(0.01)
    return pred()


async def test_starts_one_worker_per_session_no_duplicate(fast_settings) -> None:  # type: ignore[no-untyped-def]
    api = FakeApi()
    sess = _session()
    api.queue_eligible([sess])  # single batch repeats forever
    mgr, shutdown = _manager(api, fast_settings)
    task = asyncio.create_task(mgr.run())

    assert await _wait_until(lambda: mgr.active_count() == 1)
    # Stays at exactly one even after many reconcile ticks.
    await asyncio.sleep(0.1)
    assert mgr.active_count() == 1
    assert api.status_sequence().count("starting") == 1

    shutdown.set()
    await asyncio.wait_for(task, timeout=3)
    assert mgr.active_count() == 0
    assert "stopped" in api.status_sequence()


async def test_stops_worker_when_session_no_longer_eligible(fast_settings) -> None:  # type: ignore[no-untyped-def]
    api = FakeApi()
    api.queue_eligible([_session()])  # tick 1: one session
    api.queue_eligible([])  # tick 2+: none
    mgr, shutdown = _manager(api, fast_settings)
    task = asyncio.create_task(mgr.run())

    assert await _wait_until(lambda: mgr.active_count() == 1)
    assert await _wait_until(lambda: mgr.active_count() == 0)
    assert "stopped" in api.status_sequence()

    shutdown.set()
    await asyncio.wait_for(task, timeout=3)


async def test_auth_error_is_fatal_and_bubbles(fast_settings) -> None:  # type: ignore[no-untyped-def]
    api = FakeApi()
    api.eligible_error = ApiAuthError("nope", status=401, code="invalid_webhook_secret")
    mgr, _shutdown = _manager(api, fast_settings)
    with pytest.raises(ApiAuthError):
        await asyncio.wait_for(mgr.run(), timeout=2)
    assert mgr.active_count() == 0


async def test_unresolvable_package_is_quarantined_not_spawned(
    fast_settings,  # type: ignore[no-untyped-def]
) -> None:
    api = FakeApi()
    api.queue_eligible([_session(runtime="deepstream")])  # not supported in M8.1
    mgr, shutdown = _manager(api, fast_settings)
    task = asyncio.create_task(mgr.run())

    assert await _wait_until(lambda: any(s.status == "failed" for s in api.statuses))
    await asyncio.sleep(0.05)
    assert mgr.active_count() == 0
    assert "starting" not in api.status_sequence()

    shutdown.set()
    await asyncio.wait_for(task, timeout=3)


async def test_shutdown_stops_all_workers(fast_settings) -> None:  # type: ignore[no-untyped-def]
    api = FakeApi()
    api.queue_eligible([_session(), _session()])
    mgr, shutdown = _manager(api, fast_settings)
    task = asyncio.create_task(mgr.run())

    assert await _wait_until(lambda: mgr.active_count() == 2)
    shutdown.set()
    await asyncio.wait_for(task, timeout=3)
    assert mgr.active_count() == 0
    assert api.status_sequence().count("stopped") == 2
