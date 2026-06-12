"""P5-F — perception-fanout hot-path publisher tests.

No network, no sidecar. Publisher unit tests inject an httpx client
double; worker integration tests monkeypatch the publisher class with a
capturing fake (content/cadence) or point a REAL publisher at an
unreachable URL (best-effort error tolerance — the run must complete
cleanly with errors only counted)."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

import httpx
import pytest
from app.config import get_settings
from app.observations.schemas import EligibleSession
from app.observations.writer import ObservationWriter
from app.packages.resolver import resolve
from app.perception.fanout_publisher import FanoutPublisher, build_fanout_envelope
from app.runtimes.base import WorkerState
from app.runtimes.deepstream_runtime import worker as worker_mod
from app.runtimes.deepstream_runtime.worker import DeepStreamWorker

from tests._fakes import FakeApi
from tests.conftest import eligible_session_dict
from tests.test_deepstream_metadata_probe import FakeMetadataPM

_URL = "rtsp://127.0.0.1:18554/sess_abcdef0123456789"
_PKG = "deepstream.yolox_analytics_probe.v0"
_SECRET = "fanout-publish-secret-for-tests"


def _session(**ov: object) -> EligibleSession:
    return EligibleSession.model_validate(eligible_session_dict(**ov))


class _FakeHttpClient:
    """httpx.AsyncClient double: scripted responses or exceptions."""

    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        self.responses: list[httpx.Response | Exception] = []
        self.closed = False

    def queue(self, *items: httpx.Response | Exception) -> None:
        self.responses.extend(items)

    async def post(self, url: str, *, json: Any, headers: dict[str, str]) -> httpx.Response:
        self.requests.append({"url": url, "json": json, "headers": headers})
        item = self.responses.pop(0) if self.responses else _resp(202, {"subscribers": 0})
        if isinstance(item, Exception):
            raise item
        return item

    async def aclose(self) -> None:
        self.closed = True


def _resp(status: int, body: dict[str, Any] | None = None) -> httpx.Response:
    return httpx.Response(status, json=body if body is not None else {})


def _publisher(client: _FakeHttpClient, **ov: float) -> FanoutPublisher:
    return FanoutPublisher(
        url="http://127.0.0.1:18990/v1/publish",
        secret=_SECRET,
        hot_interval_seconds=ov.get("hot", 1 / 12),
        idle_interval_seconds=ov.get("idle", 2.0),
        client=client,  # type: ignore[arg-type]
    )


# --- publisher unit -----------------------------------------------------------


def test_publisher_requires_url_and_secret() -> None:
    with pytest.raises(ValueError):
        FanoutPublisher(url="", secret="x", hot_interval_seconds=0.1, idle_interval_seconds=1)
    with pytest.raises(ValueError):
        FanoutPublisher(
            url="http://x", secret="", hot_interval_seconds=0.1, idle_interval_seconds=1
        )


async def test_publish_success_tracks_subscribers_and_cadence() -> None:
    client = _FakeHttpClient()
    pub = _publisher(client)
    # Nothing published yet + 0 subscribers -> idle cadence.
    assert pub.interval_seconds == 2.0
    client.queue(_resp(202, {"accepted": True, "subscribers": 3}))
    assert await pub.publish({"type": "snapshot"}) is True
    assert pub.published == 1
    assert pub.subscribers == 3
    assert pub.interval_seconds == pytest.approx(1 / 12)  # hot while watched
    client.queue(_resp(202, {"accepted": True, "subscribers": 0}))
    await pub.publish({"type": "snapshot"})
    assert pub.interval_seconds == 2.0  # nobody watching -> idle again
    # Auth header rides every request; the secret appears nowhere else.
    assert client.requests[0]["headers"] == {"X-Upstream-Webhook-Secret": _SECRET}


async def test_publish_errors_are_counted_never_raised(
    caplog: pytest.LogCaptureFixture,
) -> None:
    client = _FakeHttpClient()
    pub = _publisher(client)
    client.queue(
        httpx.ConnectError("refused"),
        _resp(401, {"error": {"code": "invalid_publish_secret"}}),
        _resp(202, {"subscribers": 2}),
    )
    with caplog.at_level(logging.WARNING):
        assert await pub.publish({}) is False  # transport error
        assert await pub.publish({}) is False  # non-202
        assert await pub.publish({}) is True  # recovery
    assert pub.errors == 2
    assert pub.published == 1
    assert pub.subscribers == 2
    # Errors collapse cadence to idle until a success lands.
    # (after the final success with subscribers=2 we are hot again)
    assert pub.interval_seconds == pytest.approx(1 / 12)
    # The secret never reaches the log.
    assert _SECRET not in caplog.text


async def test_publish_malformed_202_body_still_counts() -> None:
    client = _FakeHttpClient()
    pub = _publisher(client)
    client.queue(httpx.Response(202, content=b"not json"))
    assert await pub.publish({}) is True
    assert pub.published == 1
    assert pub.subscribers == 0  # unchanged


async def test_aclose_closes_client() -> None:
    client = _FakeHttpClient()
    pub = _publisher(client)
    await pub.aclose()
    assert client.closed is True


def test_envelope_shape_matches_sidecar_contract() -> None:
    env = build_fanout_envelope(
        camera_id="cam-1",
        stream_session_id="sess-1",
        processor_id="proc-1",
        package_id=_PKG,
        observed_at_iso="2026-06-11T17:03:22+00:00",
        snapshot={"frame": {}, "objects": [], "geometry": {"rois": [], "lines": []}},
    )
    assert env["type"] == "snapshot"
    assert set(env) == {
        "type",
        "camera_id",
        "stream_session_id",
        "processor_id",
        "package_id",
        "observed_at",
        "snapshot",
    }


def test_fanout_disabled_by_default() -> None:
    settings = get_settings()
    assert settings.fanout_publish_url == ""
    assert settings.fanout_publish_secret is None


# --- worker integration -------------------------------------------------------


def _yolox_frame(frame_num: int) -> str:
    return "UPSTREAM_METADATA=" + json.dumps(
        {
            "runtime": "deepstream",
            "metadata_source": "pyds_pad_probe",
            "stream_session_id": None,
            "frame": {"frame_num": frame_num, "width": 1280, "height": 720},
            "objects": [
                {
                    "class_id": 0,
                    "label": "person",
                    "confidence": 0.9,
                    "bbox_pixels": {"left": 10.0, "top": 10.0, "width": 50.0, "height": 100.0},
                    "bbox_normalized": {"x": 0.01, "y": 0.01, "w": 0.04, "h": 0.14},
                    "tracker_id": "1",
                    "debug": {"analytics": {"roi_labels": ["entry-roi"]}},
                }
            ],
            "source": {"analytics": {"roi_counts": {"entry-roi": 1}}},
        }
    )


def _fanout_settings(fast_settings):  # type: ignore[no-untyped-def]
    from pydantic import SecretStr

    return fast_settings.model_copy(
        update={
            "fanout_publish_url": "http://127.0.0.1:18990/v1/publish",
            "fanout_publish_secret": SecretStr(_SECRET),
            "fanout_publish_hz": 1000.0,  # effectively every frame in tests
            "fanout_publish_idle_hz": 1000.0,
        }
    )


class _CapturingPublisher:
    """worker_mod.FanoutPublisher stand-in: records envelopes, no I/O."""

    instances: list[_CapturingPublisher] = []

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.envelopes: list[dict[str, Any]] = []
        self.published = 0
        self.errors = 0
        self.subscribers = 5
        self.closed = False
        _CapturingPublisher.instances.append(self)

    @property
    def interval_seconds(self) -> float:
        return 0.0  # publish on every frame

    async def publish(self, envelope: dict[str, Any]) -> bool:
        self.envelopes.append(envelope)
        self.published += 1
        return True

    async def aclose(self) -> None:
        self.closed = True


async def test_worker_publishes_snapshots_with_geometry(
    fast_settings,  # type: ignore[no-untyped-def]
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _CapturingPublisher.instances.clear()
    monkeypatch.setattr(worker_mod, "FanoutPublisher", _CapturingPublisher)
    api = FakeApi()
    worker = DeepStreamWorker(
        session=_session(internal_rtsp_url=_URL),
        settings=_fanout_settings(fast_settings),
        api=api,
        package=resolve(_PKG),
        pm_factory=lambda: FakeMetadataPM([_yolox_frame(1), _yolox_frame(2)]),
    )
    await asyncio.wait_for(worker.run(), timeout=5)
    assert worker.state is WorkerState.STOPPED
    assert len(_CapturingPublisher.instances) == 1
    pub = _CapturingPublisher.instances[0]
    assert pub.closed is True  # client closed at teardown
    assert len(pub.envelopes) >= 1
    env = pub.envelopes[0]
    assert env["type"] == "snapshot"
    assert env["camera_id"] == str(worker._session.camera_id)
    assert env["package_id"] == _PKG
    # The snapshot is the SAME builder output as the cold path: objects +
    # geometry present on EVERY hot frame (this is what kills the
    # geometry-flashing artifact seen with the synthetic test publisher).
    assert env["snapshot"]["objects"][0]["label"] == "person"
    assert "geometry" in env["snapshot"]
    # No secret anywhere in the envelope.
    assert _SECRET not in json.dumps(env)
    # Status payload carries the evidence fields.
    final = api.statuses[-1].payload
    assert final["fanout_publish_enabled"] is True
    assert final["fanout_subscribers"] == 5
    assert final["fanout_published_count"] >= 1
    assert _SECRET not in json.dumps([s.model_dump(mode="json") for s in api.statuses])


async def test_worker_fanout_disabled_without_config(fast_settings) -> None:  # type: ignore[no-untyped-def]
    api = FakeApi()
    worker = DeepStreamWorker(
        session=_session(internal_rtsp_url=_URL),
        settings=fast_settings,  # no fanout url/secret
        api=api,
        package=resolve(_PKG),
        pm_factory=lambda: FakeMetadataPM([_yolox_frame(1)]),
    )
    await asyncio.wait_for(worker.run(), timeout=5)
    final = api.statuses[-1].payload
    assert final["fanout_publish_enabled"] is False
    assert "fanout_published_count" not in final


async def test_worker_survives_unreachable_fanout_and_still_observes(
    fast_settings,  # type: ignore[no-untyped-def]
) -> None:
    """REAL publisher pointed at a closed port: every publish errors, but
    the run completes cleanly, observations still emit, and the errors are
    only counted (best-effort contract)."""
    from pydantic import SecretStr

    settings = fast_settings.model_copy(
        update={
            "fanout_publish_url": "http://127.0.0.1:9/v1/publish",  # closed port
            "fanout_publish_secret": SecretStr(_SECRET),
            "fanout_publish_hz": 1000.0,
            "fanout_publish_idle_hz": 1000.0,
            "fanout_publish_timeout_seconds": 0.2,
        }
    )
    api = FakeApi()
    worker = DeepStreamWorker(
        session=_session(internal_rtsp_url=_URL),
        settings=settings,
        api=api,
        package=resolve(_PKG),
        writer=ObservationWriter(api, "proc_test_001"),
        pm_factory=lambda: FakeMetadataPM([_yolox_frame(1), _yolox_frame(2)]),
    )
    await asyncio.wait_for(worker.run(), timeout=10)
    assert worker.state is WorkerState.STOPPED  # fanout failure is non-fatal
    assert "processor.zone_entered" in [o.type for o in api.observations]
    final = api.statuses[-1].payload
    assert final["fanout_publish_enabled"] is True
    assert final["fanout_publish_errors_count"] >= 1
    assert final["fanout_published_count"] == 0


async def test_non_analytics_modes_never_publish(
    fast_settings,  # type: ignore[no-untyped-def]
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _CapturingPublisher.instances.clear()
    monkeypatch.setattr(worker_mod, "FanoutPublisher", _CapturingPublisher)
    worker = DeepStreamWorker(
        session=_session(internal_rtsp_url=_URL),
        settings=_fanout_settings(fast_settings),
        api=FakeApi(),
        package=resolve("deepstream.yolox_probe.v0"),  # metadata mode, no analytics
        pm_factory=lambda: FakeMetadataPM([_yolox_frame(1)]),
    )
    await asyncio.wait_for(worker.run(), timeout=5)
    assert _CapturingPublisher.instances == []  # never constructed
