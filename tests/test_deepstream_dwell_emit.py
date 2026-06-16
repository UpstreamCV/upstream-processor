"""P4-E — DeepStream dwell observation emit path.

End-to-end (in-process) tests of the supervisor-managed analytics
worker actually writing ``processor.dwell_started``,
``processor.dwell_updated``, and ``processor.dwell_exceeded``
observations via the existing ``ObservationWriter``. No Docker /
GPU / pyds — ``FakeMetadataPM`` feeds scripted analytics frames
with explicit per-frame timestamps so the dwell timer can be
driven deterministically.

What this file is the contract for:

- Worker accepts dwell emission only in analytics mode with a
  writer wired.
- Dwell observations require a tracked object with stable
  ``roi_labels`` (P4-D + P4-D follow-up signal).
- Stable inside → ``dwell_started`` immediately after the matching
  ``zone_entered`` lands.
- ``dwell_updated`` emits at the configured interval.
- ``dwell_exceeded`` fires once when the threshold is crossed.
- ``zone_exited`` closes the session; no more updates / exceeded.
- ``processor.line_crossing`` / ``roi_occupancy`` / ``zone_*``
  paths still work in the same cycle.
"""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime, timedelta
from typing import Any

from app.errors import ApiServerError
from app.observations.schemas import OBSERVATION_TYPES, EligibleSession, ObservationPayload
from app.observations.writer import ObservationWriter
from app.packages.resolver import resolve
from app.runtimes.base import WorkerState
from app.runtimes.deepstream_runtime.worker import DeepStreamWorker

from tests._fakes import FakeApi
from tests.conftest import eligible_session_dict
from tests.test_deepstream_metadata_probe import FakeMetadataPM

_URL = "rtsp://127.0.0.1:18554/sess_abcdef0123456789"
_KEY = "abcdef0123456789"
_PROCESSOR_ID = "proc_test_001"
_BASE_TIME = datetime(2026, 5, 21, 16, 0, 0, tzinfo=UTC)


def _now_iso(offset_seconds: float = 0.0) -> str:
    return (_BASE_TIME + timedelta(seconds=offset_seconds)).isoformat()


# --- schema lock-step ---------------------------------------------------


def test_dwell_types_in_observation_types_tuple() -> None:
    assert "processor.dwell_started" in OBSERVATION_TYPES
    assert "processor.dwell_updated" in OBSERVATION_TYPES
    assert "processor.dwell_exceeded" in OBSERVATION_TYPES


def test_direction_track_telemetry_types_still_unsupported() -> None:
    """P4-E widens exactly three types. Track, direction, telemetry,
    debug stay out."""
    for t in (
        "processor.direction_detected",
        "processor.track_state",
        "processor.track_update",
        "processor.detection_seen",
        "processor.frame_summary",
        "processor.scene_quality_warning",
    ):
        assert t not in OBSERVATION_TYPES, t


# --- shared fixtures ---------------------------------------------------


def _session(**ov: object) -> EligibleSession:
    return EligibleSession.model_validate(eligible_session_dict(**ov))


def _person(
    *,
    tracker_id: str | None,
    roi_labels: list[str] | None = None,
    conf: float = 0.51,
) -> dict[str, Any]:
    """One object dict in the shape that lands in
    ``frame.objects[i]`` after pyds emit + NeutralFrameMetadata
    validation."""
    obj: dict[str, Any] = {
        "class_id": 2,
        "label": "person",
        "confidence": conf,
        "bbox_pixels": {"left": 100.0, "top": 100.0, "width": 200.0, "height": 400.0},
        "bbox_normalized": {"x": 0.078, "y": 0.139, "w": 0.156, "h": 0.556},
        "tracker_id": tracker_id,
    }
    if roi_labels is not None:
        obj["debug"] = {"analytics": {"roi_labels": list(roi_labels)}}
    return obj


def _frame_line(
    *,
    frame_num: int,
    objects: list[dict[str, Any]] | None = None,
    roi_counts: dict[str, int] | None = None,
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
    if roi_counts is not None:
        payload["source"] = {"analytics": {"roi_counts": roi_counts}}
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


# --- emit semantics ----------------------------------------------------


async def test_zone_entered_emits_dwell_started(fast_settings) -> None:  # type: ignore[no-untyped-def]
    """fast_settings sets stability_frames=1 so the first inside frame
    commits zone_entered → DwellState produces dwell_started."""
    api = FakeApi()
    pm = FakeMetadataPM(
        [
            _frame_line(
                frame_num=0,
                objects=[_person(tracker_id="0", roi_labels=["entry-roi"])],
                roi_counts={"entry-roi": 1},
                timestamp=_now_iso(0),
            ),
        ],
        exit_code=124,
    )
    w = _analytics_worker(api, fast_settings, pm)
    await asyncio.wait_for(w.run(), timeout=3)
    started = _obs_of(api, "processor.dwell_started")
    assert len(started) == 1
    body = started[0].payload
    assert body["zone_id"] == "entry-roi"
    assert body["track_id"] == "0"
    assert body["dwell_session_id"] == "entry-roi:0:1"
    assert body["duration_seconds"] == 0.0
    assert body["class_id"] == 2
    assert body["class_label"] == "person"
    assert body["debug"]["trigger"] == "zone_entered"


async def test_dwell_updated_emits_at_interval(fast_settings) -> None:  # type: ignore[no-untyped-def]
    settings = fast_settings.model_copy(
        update={
            "deepstream_dwell_update_interval_seconds": 5.0,
            "deepstream_dwell_exceeded_threshold_seconds": 600.0,
        }
    )
    api = FakeApi()
    pm = FakeMetadataPM(
        [
            _frame_line(
                frame_num=0,
                objects=[_person(tracker_id="0", roi_labels=["entry-roi"])],
                roi_counts={"entry-roi": 1},
                timestamp=_now_iso(0),
            ),
            _frame_line(
                frame_num=1,
                objects=[_person(tracker_id="0", roi_labels=["entry-roi"])],
                roi_counts={"entry-roi": 1},
                timestamp=_now_iso(5),
            ),
        ],
        exit_code=124,
    )
    w = _analytics_worker(api, settings, pm)
    await asyncio.wait_for(w.run(), timeout=3)
    updated = _obs_of(api, "processor.dwell_updated")
    assert len(updated) == 1
    body = updated[0].payload
    assert body["update_sequence"] == 1
    assert body["duration_seconds"] == 5.0


async def test_dwell_exceeded_emits_once_at_threshold(fast_settings) -> None:  # type: ignore[no-untyped-def]
    settings = fast_settings.model_copy(
        update={
            "deepstream_dwell_update_interval_seconds": 600.0,
            "deepstream_dwell_exceeded_threshold_seconds": 10.0,
        }
    )
    api = FakeApi()
    pm = FakeMetadataPM(
        [
            _frame_line(
                frame_num=0,
                objects=[_person(tracker_id="0", roi_labels=["entry-roi"])],
                roi_counts={"entry-roi": 1},
                timestamp=_now_iso(0),
            ),
            _frame_line(
                frame_num=1,
                objects=[_person(tracker_id="0", roi_labels=["entry-roi"])],
                roi_counts={"entry-roi": 1},
                timestamp=_now_iso(10),
            ),
            _frame_line(
                frame_num=2,
                objects=[_person(tracker_id="0", roi_labels=["entry-roi"])],
                roi_counts={"entry-roi": 1},
                timestamp=_now_iso(20),
            ),
        ],
        exit_code=124,
    )
    w = _analytics_worker(api, settings, pm)
    await asyncio.wait_for(w.run(), timeout=3)
    exceeded = _obs_of(api, "processor.dwell_exceeded")
    # Exactly one despite continuing to dwell at +20s.
    assert len(exceeded) == 1
    body = exceeded[0].payload
    assert body["threshold_seconds"] == 10.0
    assert body["duration_seconds"] >= 10.0


async def test_no_dwell_without_zone_membership(fast_settings) -> None:  # type: ignore[no-untyped-def]
    """No roi_labels on the object → no zone_entered → no dwell. The
    upstream metadata path didn't tag the object as inside the ROI,
    so DwellState never sees a zone_entered event."""
    api = FakeApi()
    pm = FakeMetadataPM(
        [
            _frame_line(
                frame_num=0,
                objects=[_person(tracker_id="0", roi_labels=[])],
                roi_counts={"entry-roi": 0},
                timestamp=_now_iso(0),
            ),
        ],
        exit_code=124,
    )
    w = _analytics_worker(api, fast_settings, pm)
    await asyncio.wait_for(w.run(), timeout=3)
    assert _obs_of(api, "processor.dwell_started") == []
    assert _obs_of(api, "processor.dwell_updated") == []
    assert _obs_of(api, "processor.dwell_exceeded") == []


async def test_no_dwell_without_tracker_id(fast_settings) -> None:  # type: ignore[no-untyped-def]
    api = FakeApi()
    pm = FakeMetadataPM(
        [
            _frame_line(
                frame_num=0,
                objects=[_person(tracker_id=None, roi_labels=["entry-roi"])],
                roi_counts={"entry-roi": 1},
                timestamp=_now_iso(0),
            ),
        ],
        exit_code=124,
    )
    w = _analytics_worker(api, fast_settings, pm)
    await asyncio.wait_for(w.run(), timeout=3)
    assert _obs_of(api, "processor.dwell_started") == []


async def test_zone_exited_closes_dwell_session(fast_settings) -> None:  # type: ignore[no-untyped-def]
    """After zone_exited, no more dwell_updated even if the operator
    re-appears stably outside."""
    settings = fast_settings.model_copy(
        update={
            "deepstream_dwell_update_interval_seconds": 1.0,
            "deepstream_dwell_exceeded_threshold_seconds": 600.0,
        }
    )
    api = FakeApi()
    pm = FakeMetadataPM(
        [
            _frame_line(
                frame_num=0,
                objects=[_person(tracker_id="0", roi_labels=["entry-roi"])],
                roi_counts={"entry-roi": 1},
                timestamp=_now_iso(0),
            ),
            # Step out (zone_exited fires with stability=1).
            _frame_line(
                frame_num=1,
                objects=[_person(tracker_id="0", roi_labels=[])],
                roi_counts={"entry-roi": 0},
                timestamp=_now_iso(1),
            ),
            # Frame 5s later — would trigger dwell_updated if session
            # was still open; should be silent.
            _frame_line(
                frame_num=2,
                objects=[_person(tracker_id="0", roi_labels=[])],
                roi_counts={"entry-roi": 0},
                timestamp=_now_iso(6),
            ),
        ],
        exit_code=124,
    )
    w = _analytics_worker(api, settings, pm)
    await asyncio.wait_for(w.run(), timeout=3)
    assert len(_obs_of(api, "processor.dwell_started")) == 1
    # zone_exited closed the session before the +6s tick could fire.
    assert _obs_of(api, "processor.dwell_updated") == []


async def test_re_enter_after_exit_starts_new_dwell_session(fast_settings) -> None:  # type: ignore[no-untyped-def]
    """Operator walks in, walks out, walks in again — two
    dwell_started events with distinct dwell_session_ids
    (sequence 1 then 2 in the suffix)."""
    api = FakeApi()
    pm = FakeMetadataPM(
        [
            _frame_line(
                frame_num=0,
                objects=[_person(tracker_id="0", roi_labels=["entry-roi"])],
                roi_counts={"entry-roi": 1},
                timestamp=_now_iso(0),
            ),
            _frame_line(
                frame_num=1,
                objects=[_person(tracker_id="0", roi_labels=[])],
                roi_counts={"entry-roi": 0},
                timestamp=_now_iso(1),
            ),
            _frame_line(
                frame_num=2,
                objects=[_person(tracker_id="0", roi_labels=["entry-roi"])],
                roi_counts={"entry-roi": 1},
                timestamp=_now_iso(2),
            ),
        ],
        exit_code=124,
    )
    w = _analytics_worker(api, fast_settings, pm)
    await asyncio.wait_for(w.run(), timeout=3)
    started = _obs_of(api, "processor.dwell_started")
    assert len(started) == 2
    sessions = [s.payload["dwell_session_id"] for s in started]
    assert sessions == ["entry-roi:0:1", "entry-roi:0:2"]


# --- mode gating + writer=None ----------------------------------------


async def test_writer_none_disables_dwell(fast_settings) -> None:  # type: ignore[no-untyped-def]
    api = FakeApi()
    pm = FakeMetadataPM(
        [
            _frame_line(
                frame_num=0,
                objects=[_person(tracker_id="0", roi_labels=["entry-roi"])],
                roi_counts={"entry-roi": 1},
                timestamp=_now_iso(0),
            ),
        ],
        exit_code=124,
    )
    w = DeepStreamWorker(
        session=_session(internal_rtsp_url=_URL),
        settings=fast_settings,
        api=api,
        package=resolve("deepstream.analytics_probe.v0"),
        writer=None,
        pm_factory=lambda: pm,
    )
    await asyncio.wait_for(w.run(), timeout=3)
    assert api.observations == []


async def test_metadata_probe_mode_does_not_emit_dwell(fast_settings) -> None:  # type: ignore[no-untyped-def]
    api = FakeApi()
    pm = FakeMetadataPM(
        [
            _frame_line(
                frame_num=0,
                objects=[_person(tracker_id="0", roi_labels=["entry-roi"])],
                roi_counts={"entry-roi": 1},
                timestamp=_now_iso(0),
            ),
        ],
        exit_code=124,
    )
    writer = ObservationWriter(api, _PROCESSOR_ID)
    w = DeepStreamWorker(
        session=_session(internal_rtsp_url=_URL),
        settings=fast_settings,
        api=api,
        package=resolve("deepstream.metadata_probe.v0"),
        writer=writer,
        pm_factory=lambda: pm,
    )
    await asyncio.wait_for(w.run(), timeout=3)
    assert _obs_of(api, "processor.dwell_started") == []


async def test_tracker_probe_mode_does_not_emit_dwell(fast_settings) -> None:  # type: ignore[no-untyped-def]
    api = FakeApi()
    pm = FakeMetadataPM(
        [
            _frame_line(
                frame_num=0,
                objects=[_person(tracker_id="0", roi_labels=["entry-roi"])],
                roi_counts={"entry-roi": 1},
                timestamp=_now_iso(0),
            ),
        ],
        exit_code=124,
    )
    writer = ObservationWriter(api, _PROCESSOR_ID)
    w = DeepStreamWorker(
        session=_session(internal_rtsp_url=_URL),
        settings=fast_settings,
        api=api,
        package=resolve("deepstream.tracker_probe.v0"),
        writer=writer,
        pm_factory=lambda: pm,
    )
    await asyncio.wait_for(w.run(), timeout=3)
    assert _obs_of(api, "processor.dwell_started") == []


# --- error handling ----------------------------------------------------


async def test_transient_dwell_write_failure_does_not_crash(fast_settings) -> None:  # type: ignore[no-untyped-def]
    api = FakeApi()
    api.observation_error = ApiServerError("simulated upstream 500", status=500)
    pm = FakeMetadataPM(
        [
            _frame_line(
                frame_num=0,
                objects=[_person(tracker_id="0", roi_labels=["entry-roi"])],
                roi_counts={"entry-roi": 1},
                timestamp=_now_iso(0),
            ),
        ],
        exit_code=124,
    )
    w = _analytics_worker(api, fast_settings, pm)
    await asyncio.wait_for(w.run(), timeout=3)
    assert w.state is WorkerState.STOPPED
    stopped = next(s for s in api.statuses if s.status == "stopped")
    p = stopped.payload
    assert p["dwell_observations_enabled"] is True
    # No raw stream key in any status payload.
    for s in api.statuses:
        assert _KEY not in json.dumps(s.payload)


# --- status payload + counters -----------------------------------------


async def test_status_payload_exposes_dwell_counters_and_config(fast_settings) -> None:  # type: ignore[no-untyped-def]
    settings = fast_settings.model_copy(
        update={
            "deepstream_dwell_update_interval_seconds": 7.5,
            "deepstream_dwell_exceeded_threshold_seconds": 15.0,
        }
    )
    api = FakeApi()
    pm = FakeMetadataPM(
        [
            _frame_line(
                frame_num=0,
                objects=[_person(tracker_id="0", roi_labels=["entry-roi"])],
                roi_counts={"entry-roi": 1},
                timestamp=_now_iso(0),
            ),
        ],
        exit_code=124,
    )
    w = _analytics_worker(api, settings, pm)
    await asyncio.wait_for(w.run(), timeout=3)
    stopped = next(s for s in api.statuses if s.status == "stopped")
    p = stopped.payload
    assert p["dwell_observations_enabled"] is True
    assert p["dwell_update_interval_seconds"] == 7.5
    assert p["dwell_exceeded_threshold_seconds"] == 15.0
    # One dwell_started fired; no updated / exceeded yet.
    assert p["dwell_started_observations_emitted"] == 1
    assert p["dwell_updated_observations_emitted"] == 0
    assert p["dwell_exceeded_observations_emitted"] == 0


async def test_status_payload_distinguishes_dwell_types_after_full_cycle(  # type: ignore[no-untyped-def]
    fast_settings,
) -> None:
    """Three frames covering started + updated + exceeded all in one
    cycle. The per-type counters must each go to >= 1."""
    settings = fast_settings.model_copy(
        update={
            "deepstream_dwell_update_interval_seconds": 5.0,
            "deepstream_dwell_exceeded_threshold_seconds": 10.0,
        }
    )
    api = FakeApi()
    pm = FakeMetadataPM(
        [
            _frame_line(
                frame_num=0,
                objects=[_person(tracker_id="0", roi_labels=["entry-roi"])],
                roi_counts={"entry-roi": 1},
                timestamp=_now_iso(0),
            ),
            _frame_line(
                frame_num=1,
                objects=[_person(tracker_id="0", roi_labels=["entry-roi"])],
                roi_counts={"entry-roi": 1},
                timestamp=_now_iso(5),
            ),
            _frame_line(
                frame_num=2,
                objects=[_person(tracker_id="0", roi_labels=["entry-roi"])],
                roi_counts={"entry-roi": 1},
                timestamp=_now_iso(10),
            ),
        ],
        exit_code=124,
    )
    w = _analytics_worker(api, settings, pm)
    await asyncio.wait_for(w.run(), timeout=3)
    stopped = next(s for s in api.statuses if s.status == "stopped")
    p = stopped.payload
    assert p["dwell_started_observations_emitted"] == 1
    assert p["dwell_updated_observations_emitted"] >= 1
    assert p["dwell_exceeded_observations_emitted"] == 1


# --- P5-Gap: disappearance closes the dwell (no leak / no false exceeded) ----


async def test_disappearance_closes_dwell_no_false_exceeded(fast_settings) -> None:  # type: ignore[no-untyped-def]
    """A person commits inside (dwell_started) then VANISHES. The inferred
    zone_exited must CLOSE the dwell so no dwell_exceeded / dwell_updated fires
    for the gone track — the leak the P5-Gap audit caught. grace=0 so the
    inferred exit fires on the first absent frame."""
    settings = fast_settings.model_copy(
        update={
            "deepstream_lifecycle_track_missing_grace_seconds": 0.0,
            "deepstream_dwell_update_interval_seconds": 2.0,
            "deepstream_dwell_exceeded_threshold_seconds": 5.0,
        }
    )
    api = FakeApi()
    pm = FakeMetadataPM(
        [
            # t=0: person inside -> zone_entered + dwell_started.
            _frame_line(
                frame_num=0,
                objects=[_person(tracker_id="0", roi_labels=["entry-roi"])],
                roi_counts={"entry-roi": 1},
                timestamp=_now_iso(0),
            ),
            # t=1: person GONE -> inferred zone_exited closes the dwell.
            _frame_line(
                frame_num=1, objects=[], roi_counts={"entry-roi": 0}, timestamp=_now_iso(1)
            ),
            # t=10: well past the 5s exceeded threshold; dwell is closed.
            _frame_line(
                frame_num=2, objects=[], roi_counts={"entry-roi": 0}, timestamp=_now_iso(10)
            ),
        ],
        exit_code=124,
    )
    await asyncio.wait_for(_analytics_worker(api, settings, pm).run(), timeout=5)
    assert len(_obs_of(api, "processor.dwell_started")) == 1
    assert _obs_of(api, "processor.dwell_exceeded") == []  # gone track must not exceed
    assert _obs_of(api, "processor.dwell_updated") == []
    exits = _obs_of(api, "processor.zone_exited")
    assert len(exits) == 1 and exits[0].payload["inferred"] is True
