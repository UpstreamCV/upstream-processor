from __future__ import annotations

import ast
import asyncio
import inspect

from app.errors import ApiSessionGoneError, ApiValidationError
from app.observations.schemas import EligibleSession
from app.observations.writer import ObservationWriter
from app.packages.resolver import resolve
from app.runtimes.base import WorkerState
from app.runtimes.mock_runtime import worker as worker_module
from app.runtimes.mock_runtime.worker import MockWorker

from tests._fakes import FakeApi
from tests.conftest import eligible_session_dict

_TRIPWIRE = "rtsp://TRIPWIRE.invalid:8554/should-never-be-read"


def _worker(api: FakeApi, settings):  # type: ignore[no-untyped-def]
    session = EligibleSession.model_validate(eligible_session_dict(internal_rtsp_url=_TRIPWIRE))
    return MockWorker(
        session=session,
        settings=settings,
        api=api,
        writer=ObservationWriter(api, settings.processor_id),
        package=resolve("mock.detect.v0"),
    )


async def test_lifecycle_order_and_observation_types(fast_settings) -> None:  # type: ignore[no-untyped-def]
    api = FakeApi()
    w = _worker(api, fast_settings)
    task = asyncio.create_task(w.run())
    await asyncio.sleep(0.1)
    w.request_stop()
    await asyncio.wait_for(task, timeout=2)

    seq = api.status_sequence()
    assert seq[0] == "starting"
    assert seq[1] == "attached"
    assert "processing" in seq
    assert seq[-1] == "stopped"
    assert w.state is WorkerState.STOPPED

    types = {o.type for o in api.observations}
    assert types == {
        "processor.mock_frame_seen",
        "processor.mock_heartbeat",
        "processor.mock_detection",
    }


async def test_worker_never_touches_internal_rtsp_url(fast_settings) -> None:  # type: ignore[no-untyped-def]
    # Static: no code path reads the attribute (AST-level, so the docstring
    # that *mentions* it for documentation does not trip the check).
    tree = ast.parse(inspect.getsource(worker_module))
    attrs = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    assert "internal_rtsp_url" not in attrs
    assert "internal_rtsp_url" not in names

    api = FakeApi()
    w = _worker(api, fast_settings)
    task = asyncio.create_task(w.run())
    await asyncio.sleep(0.08)
    w.request_stop()
    await asyncio.wait_for(task, timeout=2)

    # Behavioural: the RTSP URL never reached the wire or any payload.
    blob = repr(api.statuses) + repr(api.observations)
    assert "TRIPWIRE" not in blob


async def test_session_gone_stops_quietly(fast_settings) -> None:  # type: ignore[no-untyped-def]
    api = FakeApi()
    api.observation_error = ApiSessionGoneError("gone", status=404, code="session_not_found")
    w = _worker(api, fast_settings)
    await asyncio.wait_for(w.run(), timeout=2)

    seq = api.status_sequence()
    assert "starting" in seq
    assert "stopped" not in seq  # session gone -> no terminal post
    assert "failed" not in seq
    assert w.state is WorkerState.FAILED


async def test_validation_error_posts_failed(fast_settings) -> None:  # type: ignore[no-untyped-def]
    api = FakeApi()
    api.observation_error = ApiValidationError("bad", status=422, code="validation_error")
    w = _worker(api, fast_settings)
    await asyncio.wait_for(w.run(), timeout=2)

    seq = api.status_sequence()
    assert "failed" in seq
    assert "stopped" not in seq
    assert w.state is WorkerState.FAILED
