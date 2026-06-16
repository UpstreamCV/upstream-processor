"""Side task — Per-Object Line Crossing Producer.

Worker-level (in-process, no Docker/GPU/pyds) tests for
``processor.line_crossing_object``: per-OBJECT crossing evidence emitted
ALONGSIDE the aggregate cumulative-counter ``processor.line_crossing``
events, which must keep their exact pre-existing behavior.

Dedupe contract under test (``LineCrossingObjectState``): emit exactly
once on the absent → present rising edge of a (track, line) lcStatus
label; no re-emission while the label persists across sightings; the
label clearing re-arms the pair so a genuine re-crossing emits again
with a bumped sequence. Fields are omitted — never guessed — when the
producer doesn't have them.
"""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import uuid4

from app.observations.observation_adapter import LineCrossingObjectState
from app.observations.schemas import OBSERVATION_TYPES, EligibleSession, ObservationPayload
from app.observations.writer import ObservationWriter
from app.packages.resolver import resolve
from app.runtimes.deepstream_runtime.worker import DeepStreamWorker

from tests._fakes import FakeApi
from tests.conftest import eligible_session_dict
from tests.test_deepstream_metadata_probe import FakeMetadataPM

_URL = "rtsp://127.0.0.1:18554/sess_abcdef0123456789"
_PROCESSOR_ID = "proc_test_001"
_BASE_TIME = datetime(2026, 6, 10, 16, 0, 0, tzinfo=UTC)


def test_line_crossing_object_in_observation_types() -> None:
    assert "processor.line_crossing_object" in OBSERVATION_TYPES


def _now_iso(offset_seconds: float = 0.0) -> str:
    return (_BASE_TIME + timedelta(seconds=offset_seconds)).isoformat()


def _session(**ov: object) -> EligibleSession:
    return EligibleSession.model_validate(eligible_session_dict(**ov))


def _obj(
    *,
    tracker_id: str | None,
    label: str | None = "person",
    class_id: int = 2,
    conf: float | None = 0.51,
    line_labels: list[str] | None = None,
    direction: str | None = None,
) -> dict[str, Any]:
    obj: dict[str, Any] = {
        "class_id": class_id,
        "label": label,
        "confidence": conf,
        "bbox_pixels": {"left": 100.0, "top": 100.0, "width": 200.0, "height": 400.0},
        "bbox_normalized": {"x": 0.078, "y": 0.139, "w": 0.156, "h": 0.556},
        "tracker_id": tracker_id,
    }
    analytics: dict[str, Any] = {}
    if line_labels is not None:
        analytics["line_labels"] = list(line_labels)
    if direction is not None:
        analytics["direction"] = direction
    if analytics:
        obj["debug"] = {"analytics": analytics}
    return obj


def _frame_line(
    *,
    frame_num: int,
    objects: list[dict[str, Any]] | None = None,
    line_crossing_cumulative: dict[str, int] | None = None,
    timestamp: str | None = None,
) -> str:
    payload: dict[str, Any] = {
        "runtime": "deepstream",
        "metadata_source": "pyds_pad_probe",
        "stream_session_id": None,
        "frame": {"frame_num": frame_num, "width": 1280, "height": 720},
        "objects": objects or [],
    }
    if timestamp is not None:
        payload["frame"]["timestamp"] = timestamp
    if line_crossing_cumulative is not None:
        payload["source"] = {"analytics": {"line_crossing_cumulative": line_crossing_cumulative}}
    return "UPSTREAM_METADATA=" + json.dumps(payload)


def _analytics_worker(api: FakeApi, settings, pm: FakeMetadataPM) -> DeepStreamWorker:  # type: ignore[no-untyped-def]
    writer = ObservationWriter(api, _PROCESSOR_ID)
    return DeepStreamWorker(
        session=_session(internal_rtsp_url=_URL),
        settings=settings,
        api=api,
        package=resolve("deepstream.analytics_probe.v0"),
        writer=writer,
        pm_factory=lambda: pm,
    )


def _obs_of(api: FakeApi, type_: str) -> list[ObservationPayload]:
    return [o for o in api.observations if o.type == type_]


# --- emission ------------------------------------------------------------


async def test_crossing_emits_per_object_event_alongside_aggregate(fast_settings) -> None:  # type: ignore[no-untyped-def]
    """One physical crossing → one aggregate line_crossing (unchanged
    payload shape) AND one per-object line_crossing_object with full
    attribution. Coexistence, not replacement."""
    api = FakeApi()
    pm = FakeMetadataPM(
        [
            _frame_line(
                frame_num=0,
                objects=[
                    _obj(tracker_id="7", conf=0.84, line_labels=["line-1"]),
                ],
                line_crossing_cumulative={"line-1": 1},
                timestamp=_now_iso(0),
            ),
        ],
        exit_code=124,
    )
    await asyncio.wait_for(_analytics_worker(api, fast_settings, pm).run(), timeout=3)

    aggregate = _obs_of(api, "processor.line_crossing")
    assert len(aggregate) == 1
    # Aggregate payload keeps its exact pre-existing shape.
    assert aggregate[0].payload["count_delta"] == 1
    assert aggregate[0].payload["cumulative_count"] == 1
    assert "event_mode" not in aggregate[0].payload

    per_object = _obs_of(api, "processor.line_crossing_object")
    assert len(per_object) == 1
    body = per_object[0].payload
    assert body["line_id"] == "line-1"
    assert body["line_label"] == "line-1"
    assert body["track_id"] == "7"
    assert body["class_label"] == "person"
    assert body["confidence"] == 0.84
    assert body["event_mode"] == "object_crossing"
    assert body["sequence"] == 1
    assert body["idempotency_key"].endswith(":processor.line_crossing_object:line-1:7:1")
    assert "direction" not in body
    assert "analytics_direction_label" not in body


async def test_persistent_label_does_not_respam(fast_settings) -> None:  # type: ignore[no-untyped-def]
    """lcStatus held across consecutive sightings emits exactly once."""
    api = FakeApi()
    frames = [
        _frame_line(
            frame_num=i,
            objects=[_obj(tracker_id="7", line_labels=["line-1"])],
            line_crossing_cumulative={"line-1": 1},
            timestamp=_now_iso(i),
        )
        for i in range(4)
    ]
    pm = FakeMetadataPM(frames, exit_code=124)
    await asyncio.wait_for(_analytics_worker(api, fast_settings, pm).run(), timeout=3)
    assert len(_obs_of(api, "processor.line_crossing_object")) == 1


async def test_label_clear_then_recross_emits_with_bumped_sequence(fast_settings) -> None:  # type: ignore[no-untyped-def]
    api = FakeApi()
    pm = FakeMetadataPM(
        [
            _frame_line(
                frame_num=0,
                objects=[_obj(tracker_id="7", line_labels=["line-1"])],
                line_crossing_cumulative={"line-1": 1},
                timestamp=_now_iso(0),
            ),
            # Label clears (object sighted without lcStatus) → re-arms.
            _frame_line(
                frame_num=1,
                objects=[_obj(tracker_id="7", line_labels=[])],
                line_crossing_cumulative={"line-1": 1},
                timestamp=_now_iso(1),
            ),
            # Re-crossing.
            _frame_line(
                frame_num=2,
                objects=[_obj(tracker_id="7", line_labels=["line-1"])],
                line_crossing_cumulative={"line-1": 2},
                timestamp=_now_iso(2),
            ),
        ],
        exit_code=124,
    )
    await asyncio.wait_for(_analytics_worker(api, fast_settings, pm).run(), timeout=3)
    events = _obs_of(api, "processor.line_crossing_object")
    assert len(events) == 2
    assert [e.payload["sequence"] for e in events] == [1, 2]
    assert events[0].payload["idempotency_key"] != events[1].payload["idempotency_key"]


async def test_untracked_object_emits_aggregate_only(fast_settings) -> None:  # type: ignore[no-untyped-def]
    """No tracker_id → the crossing is still COUNTED (aggregate) but never
    attributed (no per-object event)."""
    api = FakeApi()
    pm = FakeMetadataPM(
        [
            _frame_line(
                frame_num=0,
                objects=[_obj(tracker_id=None, line_labels=["line-1"])],
                line_crossing_cumulative={"line-1": 1},
                timestamp=_now_iso(0),
            ),
        ],
        exit_code=124,
    )
    await asyncio.wait_for(_analytics_worker(api, fast_settings, pm).run(), timeout=3)
    assert len(_obs_of(api, "processor.line_crossing")) == 1
    assert _obs_of(api, "processor.line_crossing_object") == []


async def test_missing_class_and_confidence_are_omitted(fast_settings) -> None:  # type: ignore[no-untyped-def]
    api = FakeApi()
    pm = FakeMetadataPM(
        [
            _frame_line(
                frame_num=0,
                objects=[
                    _obj(tracker_id="7", label=None, conf=None, line_labels=["line-1"]),
                ],
                line_crossing_cumulative={"line-1": 1},
                timestamp=_now_iso(0),
            ),
        ],
        exit_code=124,
    )
    await asyncio.wait_for(_analytics_worker(api, fast_settings, pm).run(), timeout=3)
    events = _obs_of(api, "processor.line_crossing_object")
    assert len(events) == 1
    assert "class_label" not in events[0].payload
    assert "confidence" not in events[0].payload
    assert events[0].payload["track_id"] == "7"


async def test_analytics_direction_label_passthrough(fast_settings) -> None:  # type: ignore[no-untyped-def]
    """dirStatus, when nvdsanalytics provides one, rides as the clearly
    named analytics_direction_label — NEVER as a geometry-relative
    ``direction`` field."""
    api = FakeApi()
    pm = FakeMetadataPM(
        [
            _frame_line(
                frame_num=0,
                objects=[
                    _obj(tracker_id="7", line_labels=["line-1"], direction="DIR:North"),
                ],
                line_crossing_cumulative={"line-1": 1},
                timestamp=_now_iso(0),
            ),
        ],
        exit_code=124,
    )
    await asyncio.wait_for(_analytics_worker(api, fast_settings, pm).run(), timeout=3)
    events = _obs_of(api, "processor.line_crossing_object")
    assert len(events) == 1
    assert events[0].payload["analytics_direction_label"] == "DIR:North"
    assert "direction" not in events[0].payload


async def test_no_line_metadata_emits_nothing(fast_settings) -> None:  # type: ignore[no-untyped-def]
    api = FakeApi()
    pm = FakeMetadataPM(
        [
            _frame_line(
                frame_num=0,
                objects=[_obj(tracker_id="7")],  # no analytics at all
                timestamp=_now_iso(0),
            ),
        ],
        exit_code=124,
    )
    await asyncio.wait_for(_analytics_worker(api, fast_settings, pm).run(), timeout=3)
    assert _obs_of(api, "processor.line_crossing_object") == []
    assert _obs_of(api, "processor.line_crossing") == []


async def test_two_tracks_same_frame_attribute_separately(fast_settings) -> None:  # type: ignore[no-untyped-def]
    api = FakeApi()
    pm = FakeMetadataPM(
        [
            _frame_line(
                frame_num=0,
                objects=[
                    _obj(tracker_id="7", label="person", line_labels=["line-1"]),
                    _obj(tracker_id="8", label="car", class_id=0, line_labels=["line-1"]),
                ],
                line_crossing_cumulative={"line-1": 2},
                timestamp=_now_iso(0),
            ),
        ],
        exit_code=124,
    )
    await asyncio.wait_for(_analytics_worker(api, fast_settings, pm).run(), timeout=3)
    events = _obs_of(api, "processor.line_crossing_object")
    assert len(events) == 2
    by_track = {e.payload["track_id"]: e.payload for e in events}
    assert by_track["7"]["class_label"] == "person"
    assert by_track["8"]["class_label"] == "car"
    # The aggregate path still counts both (delta 2 → two count events).
    assert len(_obs_of(api, "processor.line_crossing")) == 2


# --- pure state: memory bound ---------------------------------------------


def test_state_prunes_stale_entries_beyond_max() -> None:
    state = LineCrossingObjectState(
        stream_session_id=uuid4(), max_entries=10, stale_after_seconds=60.0
    )
    t0 = _BASE_TIME
    # 12 distinct stale tracks crossing at t0.
    for i in range(12):
        state.absorb_object(
            track_id=f"t-{i}",
            line_labels={"line-1"},
            known_lines={"line-1"},
            occurred_at=t0,
        )
    assert len(state.active_by_track_line) == 12
    # A new sighting 10 minutes later prunes everything stale.
    state.absorb_object(
        track_id="fresh",
        line_labels={"line-1"},
        known_lines={"line-1"},
        occurred_at=t0 + timedelta(seconds=600),
    )
    assert len(state.active_by_track_line) == 1
    assert ("fresh", "line-1") in state.active_by_track_line


# --- P5-Gap: short-lived / fragmented tracks crossing a line ----------


async def test_short_lived_track_crossing_still_emits(fast_settings) -> None:  # type: ignore[no-untyped-def]
    """A track that crosses the line in ONE frame and then disappears
    still emits the per-object line_crossing_object (rising edge fires on
    the first sighting with the line label) AND the aggregate counter.
    Confirms object line crossing is not lost for short-lived tracks; the
    aggregate counter is the track-independent backstop either way."""
    api = FakeApi()
    pm = FakeMetadataPM(
        [
            # Frame 0: track 9 crosses (seen once with the line label).
            _frame_line(
                frame_num=0,
                objects=[_obj(tracker_id="9", line_labels=["line-1"])],
                line_crossing_cumulative={"line-1": 1},
                timestamp=_now_iso(0),
            ),
            # Frame 1: track 9 is GONE; the crossing was already emitted.
            _frame_line(
                frame_num=1,
                objects=[],
                line_crossing_cumulative={"line-1": 1},
                timestamp=_now_iso(1),
            ),
        ],
        exit_code=124,
    )
    await asyncio.wait_for(_analytics_worker(api, fast_settings, pm).run(), timeout=3)
    per_object = _obs_of(api, "processor.line_crossing_object")
    assert len(per_object) == 1
    assert per_object[0].payload["track_id"] == "9"
    assert len(_obs_of(api, "processor.line_crossing")) == 1  # aggregate backstop
