from __future__ import annotations

import asyncio
from collections.abc import Callable

import pytest
from app.errors import ApiAuthError
from app.observations.schemas import EligibleSession
from app.observations.writer import ObservationWriter
from app.packages.resolver import resolve
from app.runtimes.deepstream_runtime.worker import DeepStreamWorker
from app.runtimes.mock_runtime.worker import MockWorker
from app.supervisor.session_poller import SessionPoller
from app.supervisor.worker_manager import (
    WorkerManager,
    build_worker,
    effective_package_id,
)

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
    api.queue_eligible([_session(package_id="does.not.exist")])  # unknown package
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


# --- M8.2B: package override + runtime dispatch (pure, no docker) ---------


def test_effective_package_id_defaults_to_session(fast_settings) -> None:  # type: ignore[no-untyped-def]
    sess = _session(package_id="mock.detect.v0")
    assert effective_package_id(fast_settings, sess) == "mock.detect.v0"


def test_effective_package_id_force_override(fast_settings) -> None:  # type: ignore[no-untyped-def]
    forced = fast_settings.model_copy(update={"force_package_id": "deepstream.decode_probe.v0"})
    sess = _session(package_id="mock.detect.v0")
    assert effective_package_id(forced, sess) == "deepstream.decode_probe.v0"


def test_build_worker_dispatches_on_runtime(fast_settings) -> None:  # type: ignore[no-untyped-def]
    api = FakeApi()
    writer = ObservationWriter(api, fast_settings.processor_id)
    sess = _session()
    mock_w = build_worker(
        session=sess,
        spec=resolve("mock.detect.v0"),
        settings=fast_settings,
        api=api,
        writer=writer,
    )
    ds_w = build_worker(
        session=sess,
        spec=resolve("deepstream.decode_probe.v0"),
        settings=fast_settings,
        api=api,
        writer=writer,
    )
    assert isinstance(mock_w, MockWorker)
    assert isinstance(ds_w, DeepStreamWorker)  # constructed, not run -> no docker
