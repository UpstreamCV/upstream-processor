from __future__ import annotations

import json
from datetime import UTC, datetime
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
from app.observations.schemas import (
    EligibleSession,
    ObservationPayload,
    ProcessorStatusPayload,
)
from app.perception.schemas import PerceptionSnapshotPayload

from tests.conftest import eligible_session_dict

_BASE = "https://api.test"
_ELIGIBLE = f"{_BASE}/v1/internal/processor/eligible-sessions"
_STATUS = f"{_BASE}/v1/internal/processor/status"
_OBS = f"{_BASE}/v1/internal/processor/observations"
_PERCEPTION = f"{_BASE}/v1/internal/processor/perception-snapshots"


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


def test_eligible_session_geometry_config_defaults_to_none() -> None:
    """P6-A: ``geometry_config`` defaults to None when absent from the
    payload — a camera with no saved config (and the renamed-away
    ``zones_lines`` is no longer a field)."""
    payload = eligible_session_dict()
    del payload["geometry_config"]
    session = EligibleSession.model_validate(payload)
    assert session.geometry_config is None


@respx.mock
async def test_eligible_session_parses_geometry_config_envelope() -> None:
    """P6-A: a saved geometry config arrives as a ``{revision, config}``
    envelope and is wire-carried as an opaque dict (not consumed until P6-C)."""
    envelope = {
        "revision": 3,
        "config": {
            "version": 1,
            "frame": {"width": 1280, "height": 720, "coordinate_space": "normalized"},
            "rois": [],
            "lines": [],
        },
    }
    respx.get(_ELIGIBLE).mock(
        return_value=httpx.Response(
            200,
            json={"sessions": [eligible_session_dict(geometry_config=envelope)]},
        )
    )
    api = _client()
    resp = await api.get_eligible_sessions()
    await api.aclose()
    assert resp.sessions[0].geometry_config == envelope


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


@respx.mock
async def test_post_perception_snapshot_path_and_envelope() -> None:
    sid = uuid4()
    now = "2026-05-21T12:00:00Z"
    record = {
        "id": str(uuid4()),
        "camera_id": str(uuid4()),
        "stream_session_id": str(sid),
        "processor_id": "proc-1",
        "package_id": "deepstream.analytics_probe.v0",
        "snapshot": {"objects": [], "geometry": {}},
        "observed_at": now,
        "updated_at": now,
    }
    route = respx.post(_PERCEPTION).mock(
        return_value=httpx.Response(200, json={"snapshot": record})
    )
    api = _client()
    resp = await api.post_perception_snapshot(
        PerceptionSnapshotPayload(
            stream_session_id=sid,
            processor_id="proc-1",
            package_id="deepstream.analytics_probe.v0",
            observed_at=datetime(2026, 5, 21, 12, 0, 0, tzinfo=UTC),
            snapshot={"objects": []},
        )
    )
    await api.aclose()
    assert resp.snapshot.snapshot == {"objects": [], "geometry": {}}
    body = json.loads(route.calls.last.request.content)
    assert body["stream_session_id"] == str(sid)
    assert body["observed_at"].startswith("2026-05-21T12:00:00")


@respx.mock
async def test_post_perception_snapshot_omits_unset_optional_ids() -> None:
    """``exclude_none`` drops unset processor_id / package_id from the wire."""
    route = respx.post(_PERCEPTION).mock(
        return_value=httpx.Response(
            200,
            json={
                "snapshot": {
                    "id": str(uuid4()),
                    "camera_id": str(uuid4()),
                    "stream_session_id": str(uuid4()),
                    "processor_id": None,
                    "package_id": None,
                    "snapshot": {},
                    "observed_at": "2026-05-21T12:00:00Z",
                    "updated_at": "2026-05-21T12:00:00Z",
                }
            },
        )
    )
    api = _client()
    await api.post_perception_snapshot(
        PerceptionSnapshotPayload(
            stream_session_id=uuid4(),
            observed_at=datetime(2026, 5, 21, 12, 0, 0, tzinfo=UTC),
            snapshot={},
        )
    )
    await api.aclose()
    body = json.loads(route.calls.last.request.content)
    assert "processor_id" not in body
    assert "package_id" not in body
