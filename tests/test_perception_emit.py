"""P5-C — DeepStream worker latest-perception-snapshot emit path.

In-process tests of the analytics worker posting perception snapshots via
``ProcessorApi.post_perception_snapshot``. No Docker / GPU / pyds —
``FakeMetadataPM`` feeds scripted frames, ``FakeApi`` records the posts, the
worker runs end-to-end through ``_drive_metadata_probe``.

Perception is a dedicated path: it does NOT go through ``ObservationWriter``,
and a perception failure must never crash the worker or disturb observations.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from app.errors import ApiServerError
from app.observations.schemas import EligibleSession
from app.observations.writer import ObservationWriter
from app.packages.resolver import resolve
from app.runtimes.base import WorkerState
from app.runtimes.deepstream_runtime.worker import DeepStreamWorker

from tests._fakes import FakeApi
from tests.conftest import eligible_session_dict
from tests.test_deepstream_metadata_probe import FakeMetadataPM

_URL = "rtsp://127.0.0.1:18554/sess_abcdef0123456789"
_PROCESSOR_ID = "proc_test_001"


def _session(**ov: object) -> EligibleSession:
    return EligibleSession.model_validate(eligible_session_dict(**ov))


def _person(*, tracker_id: str = "0", class_id: int = 2) -> dict[str, Any]:
    return {
        "class_id": class_id,
        "label": "person",
        "confidence": 0.8,
        "tracker_id": tracker_id,
        "bbox_pixels": {"left": 10, "top": 20, "width": 30, "height": 40},
    }


def _frame_line(
    *,
    frame_num: int,
    objects: list[dict[str, Any]] | None = None,
    line_crossing_cumulative: dict[str, int] | None = None,
) -> str:
    payload: dict[str, Any] = {
        "runtime": "deepstream",
        "metadata_source": "pyds_pad_probe",
        "stream_session_id": None,
        "frame": {"frame_num": frame_num, "width": 1280, "height": 720},
        "objects": objects or [],
    }
    if line_crossing_cumulative is not None:
        payload["source"] = {"analytics": {"line_crossing_cumulative": line_crossing_cumulative}}
    return "UPSTREAM_METADATA=" + json.dumps(payload)


def _analytics_worker(api: FakeApi, settings: Any, pm: FakeMetadataPM) -> DeepStreamWorker:
    return DeepStreamWorker(
        session=_session(internal_rtsp_url=_URL),
        settings=settings,
        api=api,
        package=resolve("deepstream.analytics_probe.v0"),
        writer=ObservationWriter(api, _PROCESSOR_ID),
        pm_factory=lambda: pm,
    )


def _last_metadata_payload(api: FakeApi) -> dict[str, Any]:
    payloads = [s.payload for s in api.statuses if "perception_snapshots_enabled" in s.payload]
    assert payloads, "no status carried the perception keys"
    return payloads[-1]


# --- emit happy path ---------------------------------------------------


async def test_analytics_mode_posts_perception_snapshot(fast_settings: Any) -> None:
    api = FakeApi()
    pm = FakeMetadataPM([_frame_line(frame_num=0, objects=[_person()])], exit_code=124)
    w = _analytics_worker(api, fast_settings, pm)
    await asyncio.wait_for(w.run(), timeout=3)

    assert w.state is WorkerState.STOPPED
    assert len(api.perception_snapshots) == 1
    snap = api.perception_snapshots[0]
    assert snap.processor_id == _PROCESSOR_ID
    assert snap.package_id == "deepstream.analytics_probe.v0"
    body = snap.snapshot
    assert body["frame"]["width"] == 1280
    assert body["frame"]["frame_num"] == 0
    assert len(body["objects"]) == 1
    assert body["objects"][0]["track_id"] == "0"
    assert body["geometry"]["rois"][0]["id"] == "entry-roi"
    assert body["geometry"]["lines"][0]["id"] == "entry-line"


async def test_low_rate_posts_at_most_once_per_interval(fast_settings: Any) -> None:
    """Many frames inside one interval window → exactly one post (the first)."""
    settings = fast_settings.model_copy(update={"perception_snapshot_interval_seconds": 100.0})
    api = FakeApi()
    pm = FakeMetadataPM(
        [_frame_line(frame_num=i, objects=[_person()]) for i in range(8)],
        exit_code=124,
    )
    w = _analytics_worker(api, settings, pm)
    await asyncio.wait_for(w.run(), timeout=3)
    assert len(api.perception_snapshots) == 1


async def test_disabled_does_not_post(fast_settings: Any) -> None:
    settings = fast_settings.model_copy(update={"perception_snapshot_enabled": False})
    api = FakeApi()
    pm = FakeMetadataPM(
        [_frame_line(frame_num=i, objects=[_person()]) for i in range(5)],
        exit_code=124,
    )
    w = _analytics_worker(api, settings, pm)
    await asyncio.wait_for(w.run(), timeout=3)
    assert api.perception_snapshots == []


async def test_non_analytics_mode_does_not_post(fast_settings: Any) -> None:
    api = FakeApi()
    pm = FakeMetadataPM([_frame_line(frame_num=0, objects=[_person()])], exit_code=124)
    w = DeepStreamWorker(
        session=_session(internal_rtsp_url=_URL),
        settings=fast_settings,
        api=api,
        package=resolve("deepstream.metadata_probe.v0"),
        writer=None,
        pm_factory=lambda: pm,
    )
    await asyncio.wait_for(w.run(), timeout=3)
    assert api.perception_snapshots == []


# --- isolation: perception failure must not affect observations --------


async def test_post_failure_does_not_crash_worker(fast_settings: Any) -> None:
    api = FakeApi()
    api.perception_snapshot_error = ApiServerError("simulated 500", status=500)
    pm = FakeMetadataPM(
        [
            _frame_line(frame_num=0, line_crossing_cumulative={"entry-line": 0}),
            _frame_line(
                frame_num=1,
                objects=[_person()],
                line_crossing_cumulative={"entry-line": 1},
            ),
        ],
        exit_code=124,
    )
    w = _analytics_worker(api, fast_settings, pm)
    await asyncio.wait_for(w.run(), timeout=3)

    # Worker still ends cleanly despite every perception post failing.
    assert w.state is WorkerState.STOPPED
    # Observations are unaffected — the line crossing still emitted.
    line_obs = [o for o in api.observations if o.type == "processor.line_crossing"]
    assert len(line_obs) == 1
    # The error was counted in the status payload.
    assert _last_metadata_payload(api)["perception_snapshot_write_errors_count"] > 0


async def test_observations_still_emit_with_perception_enabled(
    fast_settings: Any,
) -> None:
    api = FakeApi()
    pm = FakeMetadataPM(
        [
            _frame_line(frame_num=0, line_crossing_cumulative={"entry-line": 0}),
            _frame_line(
                frame_num=1,
                objects=[_person()],
                line_crossing_cumulative={"entry-line": 1},
            ),
        ],
        exit_code=124,
    )
    w = _analytics_worker(api, fast_settings, pm)
    await asyncio.wait_for(w.run(), timeout=3)

    line_obs = [o for o in api.observations if o.type == "processor.line_crossing"]
    assert len(line_obs) == 1
    assert len(api.perception_snapshots) >= 1


# --- payload shaping ---------------------------------------------------


async def test_max_objects_capped_in_posted_snapshot(fast_settings: Any) -> None:
    settings = fast_settings.model_copy(update={"perception_snapshot_max_objects": 2})
    api = FakeApi()
    pm = FakeMetadataPM(
        [
            _frame_line(
                frame_num=0,
                objects=[_person(tracker_id=str(i)) for i in range(5)],
            )
        ],
        exit_code=124,
    )
    w = _analytics_worker(api, settings, pm)
    await asyncio.wait_for(w.run(), timeout=3)

    assert len(api.perception_snapshots) == 1
    body = api.perception_snapshots[0].snapshot
    assert len(body["objects"]) == 2
    assert body["debug"]["objects_seen"] == 5
    assert body["debug"]["objects_truncated"] is True


async def test_perception_counters_in_status_payload(fast_settings: Any) -> None:
    api = FakeApi()
    pm = FakeMetadataPM([_frame_line(frame_num=0, objects=[_person()])], exit_code=124)
    w = _analytics_worker(api, fast_settings, pm)
    await asyncio.wait_for(w.run(), timeout=3)

    payload = _last_metadata_payload(api)
    assert payload["perception_snapshots_enabled"] is True
    assert payload["perception_snapshots_posted"] >= 1
    assert payload["perception_snapshot_write_errors_count"] == 0
    assert "perception_snapshot_interval_seconds" in payload
    assert payload["last_perception_snapshot_at"] is not None
