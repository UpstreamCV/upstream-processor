"""Counting-container API client: idempotency keying + the status auth verdict.

``workers/overhead_people_counter`` is the counting container's own tree — it
ships with the image and its deps (``requests``, cv2, torch) are installed
there, NOT in the supervisor's environment. Adding them here just to test would
put them on the live supervisor's ``uv run`` sync path, so ``requests`` is
faked: these tests exercise our retry/keying logic, which is all that the
transport is needed for. The CV-dependent worker code stays untested here.
"""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path
from typing import Any

import pytest

_WORKER_SRC = Path(__file__).resolve().parent.parent / "workers/overhead_people_counter/src"


class FakeResponse:
    def __init__(self, status_code: int, payload: dict[str, Any] | None = None) -> None:
        self.status_code = status_code
        self._payload = payload or {}

    def json(self) -> dict[str, Any]:
        return self._payload


def _fake_requests() -> types.ModuleType:
    """Minimal stand-in for the parts of ``requests`` the client touches."""
    mod = types.ModuleType("requests")

    # Names mirror requests' own; the client catches them by attribute.
    class RequestException(Exception):  # noqa: N818
        pass

    class Timeout(RequestException):
        pass

    class ConnectionError(RequestException):
        pass

    mod.RequestException = RequestException  # type: ignore[attr-defined]
    mod.Timeout = Timeout  # type: ignore[attr-defined]
    mod.ConnectionError = ConnectionError  # type: ignore[attr-defined]
    mod.Session = object  # type: ignore[attr-defined]
    mod.Response = FakeResponse  # type: ignore[attr-defined]  # evaluated annotation

    def _unstubbed(*_a: Any, **_kw: Any) -> None:
        raise AssertionError("test did not stub requests.post")

    mod.post = _unstubbed  # type: ignore[attr-defined]
    return mod


@pytest.fixture
def client_mod(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Import the container's API client against the fake transport."""
    fake = _fake_requests()
    monkeypatch.setitem(sys.modules, "requests", fake)
    monkeypatch.syspath_prepend(str(_WORKER_SRC))
    for name in ("api", "api.crowdcount_api_client"):
        sys.modules.pop(name, None)
    mod = importlib.import_module("api.crowdcount_api_client")
    yield mod
    # Never leave the container tree (or the fake requests) importable for
    # the rest of the suite.
    for name in ("api", "api.crowdcount_api_client"):
        sys.modules.pop(name, None)


def _client(client_mod: Any, **ov: Any) -> Any:
    kwargs: dict[str, Any] = {
        "device_id": "C4804",
        "base_url": "https://api.crowdcount.tech",
        "service_secret": "s3cret",
        "timeout": 3,
    }
    kwargs.update(ov)
    return client_mod.CrowdCountAPIClient(**kwargs)


class _Recorder:
    """Records each POST and replays a scripted sequence of outcomes."""

    def __init__(self, outcomes: list[Any]) -> None:
        self._outcomes = list(outcomes)
        self.keys: list[str | None] = []
        self.calls = 0

    def __call__(
        self,
        url: str,
        data: bytes = b"",
        headers: dict[str, str] | None = None,
        timeout: float | None = None,
    ) -> FakeResponse:
        self.calls += 1
        self.keys.append((headers or {}).get("Idempotency-Key"))
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


@pytest.mark.parametrize(
    ("method", "kwargs"),
    [
        ("add_to_count", {"count": 1}),
        ("subtract_from_count", {"count": 1}),
        ("update_status", {"status": "online"}),
    ],
)
def test_retry_reuses_one_idempotency_key(
    client_mod: Any, monkeypatch: pytest.MonkeyPatch, method: str, kwargs: dict[str, Any]
) -> None:
    """A ReadTimeout retry must carry the SAME key: the server may have already
    committed the first attempt, and a fresh key would double-count it."""
    rec = _Recorder([client_mod.requests.Timeout("read timeout"), FakeResponse(200)])
    monkeypatch.setattr(client_mod.requests, "post", rec)
    monkeypatch.setattr(client_mod.time, "sleep", lambda _s: None)

    c = _client(client_mod)
    assert getattr(c, method)(**kwargs) is True
    assert rec.calls == 2
    assert rec.keys[0] is not None
    assert rec.keys[0] == rec.keys[1]


def test_separate_operations_get_distinct_keys(
    client_mod: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One key per logical operation — two crossings must NOT dedupe together."""
    rec = _Recorder([FakeResponse(200), FakeResponse(200)])
    monkeypatch.setattr(client_mod.requests, "post", rec)

    c = _client(client_mod)
    assert c.add_to_count(1) is True
    assert c.add_to_count(1) is True
    assert rec.keys[0] != rec.keys[1]


def test_update_status_records_auth_rejection(
    client_mod: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A wrong/rotated service secret is a verdict, not a blip: it must be
    distinguishable so the worker can fail fast instead of silently dropping
    every count."""
    monkeypatch.setattr(
        client_mod.requests,
        "post",
        _Recorder([
            FakeResponse(401, {"error": "Invalid service signature",
                               "code": "service_auth_invalid_signature"})
        ]),
    )
    c = _client(client_mod)
    assert c.update_status("online", max_retries=0) is False
    assert c.last_status_auth_ok is False
    assert "Invalid service signature" in c.last_status_auth_error


@pytest.mark.parametrize(
    "payload",
    [
        pytest.param(
            {"error": "Timestamp expired (drift: 400s, max: 300s)",
             "code": "service_auth_timestamp_invalid"},
            id="clock-skew",
        ),
        pytest.param(
            {"error": "Nonce already used (possible replay attack)",
             "code": "service_auth_nonce_rejected"},
            id="nonce-replay",
        ),
        pytest.param({"error": "Unauthorized"}, id="no-code-from-older-server"),
    ],
)
def test_update_status_non_credential_401s_are_not_a_verdict(
    client_mod: Any, monkeypatch: pytest.MonkeyPatch, payload: dict[str, Any]
) -> None:
    """Only rejections re-signing cannot fix are fatal. Clock skew on the
    processor host 401s every worker at once — treating that as a bad secret
    would turn an NTP blip into a fleet-wide outage. A server with no code
    yields no verdict, so the codes can deploy in either order."""
    monkeypatch.setattr(
        client_mod.requests, "post", _Recorder([FakeResponse(401, payload)])
    )
    c = _client(client_mod)
    assert c.update_status("online", max_retries=0) is False
    assert c.last_status_auth_ok is None


def test_update_status_transient_failure_is_not_an_auth_verdict(
    client_mod: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A 5xx/timeout must leave the verdict unknown (None) — killing the worker
    on an API blip would be worse than the bug being fixed."""
    c = _client(client_mod)
    for outcome in (FakeResponse(500), client_mod.requests.Timeout("boom")):
        monkeypatch.setattr(client_mod.requests, "post", _Recorder([outcome]))
        assert c.update_status("online", max_retries=0) is False
        assert c.last_status_auth_ok is None


def test_update_status_success_records_authorized(
    client_mod: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(client_mod.requests, "post", _Recorder([FakeResponse(200)]))
    c = _client(client_mod)
    assert c.update_status("online", max_retries=0) is True
    assert c.last_status_auth_ok is True
