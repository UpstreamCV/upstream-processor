from __future__ import annotations

import pytest
from app.errors import ApiServerError
from app.observations.schemas import EligibleSession
from app.supervisor.session_poller import SessionPoller

from tests._fakes import FakeApi
from tests.conftest import eligible_session_dict


async def test_fetch_returns_parsed_sessions() -> None:
    api = FakeApi()
    api.queue_eligible([EligibleSession.model_validate(eligible_session_dict())])
    poller = SessionPoller(api, limit=10)
    sessions = await poller.fetch_eligible()
    assert len(sessions) == 1
    assert sessions[0].runtime == "mock"


async def test_fetch_empty_returns_empty_list() -> None:
    poller = SessionPoller(FakeApi(), limit=10)
    assert await poller.fetch_eligible() == []


async def test_transient_error_propagates_for_manager_to_handle() -> None:
    api = FakeApi()
    api.eligible_error = ApiServerError("boom", status=503)
    poller = SessionPoller(api, limit=10)
    with pytest.raises(ApiServerError):
        await poller.fetch_eligible()
