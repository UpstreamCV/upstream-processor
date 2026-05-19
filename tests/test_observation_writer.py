from __future__ import annotations

from datetime import datetime
from uuid import uuid4

import pytest
from app.errors import ApiServerError, ProcessorError
from app.observations.writer import ObservationWriter

from tests._fakes import FakeApi


async def test_emit_builds_correct_payload() -> None:
    api = FakeApi()
    writer = ObservationWriter(api, "proc_test_001")
    sid = uuid4()
    await writer.emit(
        stream_session_id=sid,
        obs_type="processor.mock_detection",
        payload={"class": "person"},
        package_id="mock.detect.v0",
    )
    assert len(api.observations) == 1
    obs = api.observations[0]
    assert obs.stream_session_id == sid
    assert obs.type == "processor.mock_detection"
    assert obs.processor_id == "proc_test_001"
    assert obs.package_id == "mock.detect.v0"
    assert obs.payload == {"class": "person"}
    # observed_at defaulted to a tz-aware datetime.
    assert isinstance(obs.observed_at, datetime)
    assert obs.observed_at.tzinfo is not None


async def test_emit_rejects_non_mock_type() -> None:
    writer = ObservationWriter(FakeApi(), "p")
    with pytest.raises(ProcessorError):
        await writer.emit(
            stream_session_id=uuid4(),
            obs_type="processor.real_detection",  # type: ignore[arg-type]
            payload={},
        )


async def test_emit_propagates_api_error() -> None:
    api = FakeApi()
    api.observation_error = ApiServerError("5xx", status=500)
    writer = ObservationWriter(api, "p")
    with pytest.raises(ApiServerError):
        await writer.emit(
            stream_session_id=uuid4(),
            obs_type="processor.mock_heartbeat",
            payload={},
        )
