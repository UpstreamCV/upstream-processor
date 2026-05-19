from __future__ import annotations

import asyncio
from typing import cast
from uuid import uuid4

from app.runtimes.mock_runtime.worker import MockWorker
from app.supervisor.session_poller import SessionPoller
from app.supervisor.worker_manager import WorkerManager, _Handle

from tests._fakes import FakeApi


class _StuckWorker:
    def request_stop(self) -> None:  # ignores the request — simulates a hang
        return None


async def test_shutdown_is_bounded_even_with_a_hung_worker(
    fast_settings,  # type: ignore[no-untyped-def]
) -> None:
    api = FakeApi()
    mgr = WorkerManager(
        poller=SessionPoller(api, 50),
        api=api,
        settings=fast_settings,
        shutdown=asyncio.Event(),
    )
    stuck = asyncio.create_task(asyncio.sleep(1000))
    mgr._workers[uuid4()] = _Handle(task=stuck, worker=cast(MockWorker, _StuckWorker()))

    loop = asyncio.get_event_loop()
    start = loop.time()
    await mgr._shutdown_all()
    elapsed = loop.time() - start

    # final_status_timeout (0.5) then cancel; well under overall (1.0) + slack.
    assert elapsed < fast_settings.overall_shutdown_timeout_seconds + 0.5
    assert stuck.cancelled()
    assert mgr.active_count() == 0
