from __future__ import annotations

from uuid import uuid4

import httpx
import pytest
import respx
from app.api_client.upstream_api import UpstreamApiClient
from app.config import get_settings
from app.errors import (
    ApiAuthError,
    ApiConnectionError,
    ApiContractError,
    ApiServerError,
    ApiSessionGoneError,
    ApiValidationError,
)
from app.observations.schemas import ObservationPayload, ProcessorStatusPayload

from tests.conftest import eligible_session_dict

_BASE = "https://api.test"
_ELIGIBLE = f"{_BASE}/v1/internal/processor/eligible-sessions"
_STATUS = f"{_BASE}/v1/internal/processor/status"
_OBS = f"{_BASE}/v1/internal/processor/observations"


def _client() -> UpstreamApiClient:
    return UpstreamApiClient(get_settings())


@respx.mock
async def test_sends_webhook_secret_header_and_no_authorization() -> None:
    route = respx.get(_ELIGIBLE).mock(return_value=httpx.Response(200, json={"sessions": []}))
    api = _client()
    await api.get_eligible_sessions(limit=7)
    await api.aclose()

    req = route.calls.last.request
    secret = get_settings().processor_webhook_secret.get_secret_value()
    assert req.headers["X-Upstream-Webhook-Secret"] == secret
    assert "authorization" not in {k.lower() for k in req.headers}
    assert req.url.params["limit"] == "7"


@respx.mock
async def test_parses_eligible_sessions_envelope() -> None:
    respx.get(_ELIGIBLE).mock(
        return_value=httpx.Response(200, json={"sessions": [eligible_session_dict()]})
    )
    api = _client()
    resp = await api.get_eligible_sessions()
    await api.aclose()
    assert len(resp.sessions) == 1
    assert resp.sessions[0].runtime == "mock"


@respx.mock
async def test_status_post_excludes_none_and_has_no_tenant_fields() -> None:
    route = respx.post(_STATUS).mock(
        return_value=httpx.Response(
            200,
            json={
                "id": str(uuid4()),
                "account_id": str(uuid4()),
                "camera_id": str(uuid4()),
                "stream_session_id": str(uuid4()),
                "processor_id": "p",
                "runtime": "mock",
                "package_id": "mock.detect.v0",
                "status": "processing",
                "created_at": "2026-05-18T00:00:00+00:00",
                "updated_at": "2026-05-18T00:00:00+00:00",
            },
        )
    )
    api = _client()
    await api.post_processor_status(
        ProcessorStatusPayload(
            processor_id="p",
            stream_session_id=uuid4(),
            runtime="mock",
            package_id="mock.detect.v0",
            status="processing",
        )
    )
    await api.aclose()
    body = route.calls.last.request.content.decode()
    assert "account_id" not in body
    assert "camera_id" not in body
    assert "fps_input" not in body  # None excluded


@respx.mock
async def test_observation_post_parses_response() -> None:
    oid = uuid4()
    respx.post(_OBS).mock(return_value=httpx.Response(200, json={"id": str(oid), "count": 1}))
    api = _client()
    resp = await api.post_observation(
        ObservationPayload(
            stream_session_id=uuid4(),
            type="processor.mock_detection",
            payload={"x": 1},
            observed_at="2026-05-18T00:00:00+00:00",  # type: ignore[arg-type]
        )
    )
    await api.aclose()
    assert resp.id == oid
    assert resp.count == 1


@pytest.mark.parametrize(
    ("status", "exc"),
    [
        (401, ApiAuthError),
        (404, ApiSessionGoneError),
        (422, ApiValidationError),
        (500, ApiServerError),
        (503, ApiServerError),
    ],
)
@respx.mock
async def test_status_codes_map_to_typed_errors(status, exc) -> None:  # type: ignore[no-untyped-def]
    respx.get(_ELIGIBLE).mock(
        return_value=httpx.Response(status, json={"error": {"code": "x", "request_id": "r1"}})
    )
    api = _client()
    with pytest.raises(exc):
        await api.get_eligible_sessions()
    await api.aclose()


@respx.mock
async def test_timeout_and_transport_errors_are_connection_errors() -> None:
    respx.get(_ELIGIBLE).mock(side_effect=httpx.ConnectTimeout("slow"))
    api = _client()
    with pytest.raises(ApiConnectionError):
        await api.get_eligible_sessions()
    await api.aclose()


@respx.mock
async def test_2xx_unparsable_body_is_contract_error() -> None:
    respx.get(_ELIGIBLE).mock(return_value=httpx.Response(200, text="definitely not json"))
    api = _client()
    with pytest.raises(ApiContractError):
        await api.get_eligible_sessions()
    await api.aclose()


@respx.mock
async def test_secret_never_appears_in_exception_text() -> None:
    respx.get(_ELIGIBLE).mock(return_value=httpx.Response(401, json={}))
    api = _client()
    secret = get_settings().processor_webhook_secret.get_secret_value()
    try:
        await api.get_eligible_sessions()
    except ApiAuthError as exc:
        assert secret not in str(exc)
        assert secret not in repr(exc)
    await api.aclose()
