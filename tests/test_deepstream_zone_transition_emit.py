"""P4-D — DeepStream zone transition observation emit path.

End-to-end (in-process) tests of the supervisor-managed analytics
worker actually writing ``processor.zone_entered`` and
``processor.zone_exited`` observations via the existing
``ObservationWriter``. No Docker / GPU / pyds — ``FakeMetadataPM``
feeds scripted analytics frames that include per-object ``debug.analytics.roi_labels``
(the P3.5 shape), the worker runs end-to-end through
``_drive_metadata_probe``, and ``FakeApi`` records ``post_observation``
calls.

What this file is the contract for:

- Worker accepts zone transition emission only in analytics mode and
  only when a writer is wired.
- Stable inside → ``processor.zone_entered``; stable outside (after a
  prior inside) → ``processor.zone_exited``. Disappearance does NOT
  emit.
- ``tracker_id`` is required.
- The line-crossing and ROI emit paths still work in the same cycle.
- Status payload exposes the configured threshold + both per-type
  counters + a shared error counter.
"""

from __future__ import annotations

import asyncio
import json
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


# --- schema lock-step regression ---------------------------------------


def test_zone_types_now_in_observation_types_tuple() -> None:
    assert "processor.zone_entered" in OBSERVATION_TYPES
    assert "processor.zone_exited" in OBSERVATION_TYPES


def test_direction_track_telemetry_types_still_unsupported() -> None:
    """P4-E widens the three dwell types. Direction, track-state,
    track-update, and telemetry types stay deferred."""
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
    validation. ``roi_labels`` (when supplied) goes under
    ``debug.analytics.roi_labels`` — that's the P3.5 contract."""
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
    line_crossing_cumulative: dict[str, int] | None = None,
    roi_counts: dict[str, int] | None = None,
) -> str:
    payload: dict[str, Any] = {
        "runtime": "deepstream",
        "metadata_source": "pyds_pad_probe",
        "stream_session_id": None,
        "frame": {"frame_num": frame_num, "width": 1280, "height": 720},
        "objects": objects or [],
    }
    analytics: dict[str, Any] = {}
    if line_crossing_cumulative is not None:
        analytics["line_crossing_cumulative"] = line_crossing_cumulative
    if roi_counts is not None:
        analytics["roi_counts"] = roi_counts
    if analytics:
        payload["source"] = {"analytics": analytics}
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


def _zone_obs(api: FakeApi, *, of_type: str | None = None) -> list[ObservationPayload]:
    if of_type is None:
        return [
            o
            for o in api.observations
            if o.type in ("processor.zone_entered", "processor.zone_exited")
        ]
    return [o for o in api.observations if o.type == of_type]


# --- emit semantics ----------------------------------------------------


async def test_stable_inside_emits_zone_entered(fast_settings) -> None:  # type: ignore[no-untyped-def]
    """fast_settings sets stability_frames=1 so the first inside frame
    commits + emits."""
    api = FakeApi()
    pm = FakeMetadataPM(
        [
            _frame_line(
                frame_num=0,
                objects=[_person(tracker_id="0", roi_labels=["entry-roi"])],
                roi_counts={"entry-roi": 1},
            ),
        ],
        exit_code=124,
    )
    w = _analytics_worker(api, fast_settings, pm)
    await asyncio.wait_for(w.run(), timeout=3)
    enters = _zone_obs(api, of_type="processor.zone_entered")
    assert len(enters) == 1
    body = enters[0].payload
    assert body["zone_id"] == "entry-roi"
    assert body["track_id"] == "0"
    assert body["sequence"] == 1
    assert body["class_id"] == 2
    assert body["class_label"] == "person"
    assert body["idempotency_key"].endswith(":entry-roi:0:1")
    assert body["debug"]["transition"] == "outside_to_inside"


async def test_stable_outside_after_inside_emits_zone_exited(fast_settings) -> None:  # type: ignore[no-untyped-def]
    """Track committed inside, then observed without the zone label →
    zone_exited fires."""
    api = FakeApi()
    pm = FakeMetadataPM(
        [
            _frame_line(
                frame_num=0,
                objects=[_person(tracker_id="0", roi_labels=["entry-roi"])],
                roi_counts={"entry-roi": 1},
            ),
            _frame_line(
                frame_num=1,
                objects=[_person(tracker_id="0", roi_labels=[])],
                roi_counts={"entry-roi": 0},
            ),
        ],
        exit_code=124,
    )
    w = _analytics_worker(api, fast_settings, pm)
    await asyncio.wait_for(w.run(), timeout=3)
    enters = _zone_obs(api, of_type="processor.zone_entered")
    exits = _zone_obs(api, of_type="processor.zone_exited")
    assert len(enters) == 1
    assert len(exits) == 1
    exit_body = exits[0].payload
    assert exit_body["zone_id"] == "entry-roi"
    assert exit_body["track_id"] == "0"
    assert exit_body["sequence"] == 1
    assert exit_body["debug"]["transition"] == "inside_to_outside"


async def test_short_roi_label_flap_does_not_emit_false_exit(fast_settings) -> None:  # type: ignore[no-untyped-def]
    """With stability=3, a single-frame outside flap doesn't trigger
    zone_exited."""
    settings = fast_settings.model_copy(update={"deepstream_zone_membership_stability_frames": 3})
    api = FakeApi()
    pm = FakeMetadataPM(
        [
            # Three frames inside to commit zone_entered.
            _frame_line(
                frame_num=0,
                objects=[_person(tracker_id="0", roi_labels=["entry-roi"])],
                roi_counts={"entry-roi": 1},
            ),
            _frame_line(
                frame_num=1,
                objects=[_person(tracker_id="0", roi_labels=["entry-roi"])],
                roi_counts={"entry-roi": 1},
            ),
            _frame_line(
                frame_num=2,
                objects=[_person(tracker_id="0", roi_labels=["entry-roi"])],
                roi_counts={"entry-roi": 1},
            ),
            # One flap outside.
            _frame_line(
                frame_num=3,
                objects=[_person(tracker_id="0", roi_labels=[])],
                roi_counts={"entry-roi": 0},
            ),
            # Back inside immediately.
            _frame_line(
                frame_num=4,
                objects=[_person(tracker_id="0", roi_labels=["entry-roi"])],
                roi_counts={"entry-roi": 1},
            ),
        ],
        exit_code=124,
    )
    w = _analytics_worker(api, settings, pm)
    await asyncio.wait_for(w.run(), timeout=3)
    enters = _zone_obs(api, of_type="processor.zone_entered")
    exits = _zone_obs(api, of_type="processor.zone_exited")
    assert len(enters) == 1
    assert exits == []  # no false zone_exited


async def test_no_emit_when_tracker_id_missing(fast_settings) -> None:  # type: ignore[no-untyped-def]
    """Tracker missing → no zone transition (we can't identify the
    subject of the event)."""
    api = FakeApi()
    pm = FakeMetadataPM(
        [
            _frame_line(
                frame_num=0,
                objects=[_person(tracker_id=None, roi_labels=["entry-roi"])],
                roi_counts={"entry-roi": 1},
            ),
        ],
        exit_code=124,
    )
    w = _analytics_worker(api, fast_settings, pm)
    await asyncio.wait_for(w.run(), timeout=3)
    assert _zone_obs(api) == []


async def test_no_emit_when_roi_labels_missing(fast_settings) -> None:  # type: ignore[no-untyped-def]
    """Object without ``debug.analytics.roi_labels`` (e.g. metadata
    arrived from a non-analytics path or the SDK didn't emit) → no
    zone transition."""
    api = FakeApi()
    pm = FakeMetadataPM(
        [
            _frame_line(
                frame_num=0,
                objects=[_person(tracker_id="0", roi_labels=None)],
                roi_counts={"entry-roi": 1},
            ),
        ],
        exit_code=124,
    )
    w = _analytics_worker(api, fast_settings, pm)
    await asyncio.wait_for(w.run(), timeout=3)
    assert _zone_obs(api) == []


async def test_no_emit_when_roi_labels_not_a_list(fast_settings) -> None:  # type: ignore[no-untyped-def]
    """Malformed ``roi_labels`` (string, dict, etc.) is silently
    ignored — no crash, no emit."""
    api = FakeApi()
    # Craft a frame where roi_labels is a string (wrong type).
    bad = "UPSTREAM_METADATA=" + json.dumps(
        {
            "runtime": "deepstream",
            "metadata_source": "pyds_pad_probe",
            "stream_session_id": None,
            "frame": {"frame_num": 0, "width": 1280, "height": 720},
            "objects": [
                {
                    "class_id": 2,
                    "label": "person",
                    "confidence": 0.5,
                    "bbox_pixels": {"left": 0, "top": 0, "width": 100, "height": 200},
                    "bbox_normalized": {"x": 0, "y": 0, "w": 0.1, "h": 0.2},
                    "tracker_id": "0",
                    "debug": {"analytics": {"roi_labels": "not-a-list"}},
                }
            ],
            "source": {"analytics": {"roi_counts": {"entry-roi": 1}}},
        }
    )
    pm = FakeMetadataPM([bad], exit_code=124)
    w = _analytics_worker(api, fast_settings, pm)
    await asyncio.wait_for(w.run(), timeout=3)
    assert _zone_obs(api) == []


async def test_no_emit_when_roi_counts_empty(fast_settings) -> None:  # type: ignore[no-untyped-def]
    """No known zones for this frame (nothing in ``roi_counts``) → no
    transition tracking. We need to know what zones exist before we
    can decide membership."""
    api = FakeApi()
    pm = FakeMetadataPM(
        [
            _frame_line(
                frame_num=0,
                objects=[_person(tracker_id="0", roi_labels=["entry-roi"])],
                # No roi_counts in source.analytics.
            ),
        ],
        exit_code=124,
    )
    w = _analytics_worker(api, fast_settings, pm)
    await asyncio.wait_for(w.run(), timeout=3)
    assert _zone_obs(api) == []


# --- mode gating + writer=None ----------------------------------------


async def test_writer_none_disables_zone_emission(fast_settings) -> None:  # type: ignore[no-untyped-def]
    api = FakeApi()
    pm = FakeMetadataPM(
        [
            _frame_line(
                frame_num=0,
                objects=[_person(tracker_id="0", roi_labels=["entry-roi"])],
                roi_counts={"entry-roi": 1},
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
    stopped = next(s for s in api.statuses if s.status == "stopped")
    assert stopped.payload.get("observations_enabled") is False


async def test_tracker_mode_does_not_emit_zone(fast_settings) -> None:  # type: ignore[no-untyped-def]
    api = FakeApi()
    pm = FakeMetadataPM(
        [
            _frame_line(
                frame_num=0,
                objects=[_person(tracker_id="0", roi_labels=["entry-roi"])],
                roi_counts={"entry-roi": 1},
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
    assert _zone_obs(api) == []


async def test_metadata_mode_does_not_emit_zone(fast_settings) -> None:  # type: ignore[no-untyped-def]
    api = FakeApi()
    pm = FakeMetadataPM(
        [
            _frame_line(
                frame_num=0,
                objects=[_person(tracker_id="0", roi_labels=["entry-roi"])],
                roi_counts={"entry-roi": 1},
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
    assert _zone_obs(api) == []


# --- error handling ----------------------------------------------------


async def test_transient_zone_write_failure_does_not_crash(fast_settings) -> None:  # type: ignore[no-untyped-def]
    api = FakeApi()
    api.observation_error = ApiServerError("simulated upstream 500", status=500)
    pm = FakeMetadataPM(
        [
            _frame_line(
                frame_num=0,
                objects=[_person(tracker_id="0", roi_labels=["entry-roi"])],
                roi_counts={"entry-roi": 1},
            ),
        ],
        exit_code=124,
    )
    w = _analytics_worker(api, fast_settings, pm)
    await asyncio.wait_for(w.run(), timeout=3)
    assert w.state is WorkerState.STOPPED
    stopped = next(s for s in api.statuses if s.status == "stopped")
    p = stopped.payload
    assert p["zone_transition_observations_enabled"] is True
    # All emit attempts failed → zone counters stay 0; shared error
    # counter advances (also covers any ROI/line emits that hit the
    # same failure on the same frame).
    assert p["zone_entered_observations_emitted"] == 0
    assert p["zone_exited_observations_emitted"] == 0
    assert p["observation_write_errors_count"] >= 1
    # No raw stream key in any status payload.
    for s in api.statuses:
        assert _KEY not in json.dumps(s.payload)


# --- status payload + counters -----------------------------------------


async def test_status_payload_exposes_zone_threshold_and_counters(fast_settings) -> None:  # type: ignore[no-untyped-def]
    settings = fast_settings.model_copy(update={"deepstream_zone_membership_stability_frames": 7})
    api = FakeApi()
    pm = FakeMetadataPM(
        [
            _frame_line(
                frame_num=0,
                objects=[_person(tracker_id="0", roi_labels=["entry-roi"])],
                roi_counts={"entry-roi": 1},
            ),
        ],
        exit_code=124,
    )
    w = _analytics_worker(api, settings, pm)
    await asyncio.wait_for(w.run(), timeout=3)
    stopped = next(s for s in api.statuses if s.status == "stopped")
    p = stopped.payload
    assert p["zone_transition_observations_enabled"] is True
    assert p["zone_membership_stability_frames"] == 7
    # One frame at stability=7 → no commit yet; both counters stay 0.
    assert p["zone_entered_observations_emitted"] == 0
    assert p["zone_exited_observations_emitted"] == 0


async def test_counters_distinguish_zone_entered_and_exited(fast_settings) -> None:  # type: ignore[no-untyped-def]
    """The two per-type counters must increment independently. Status
    payload's last ``processing`` post reflects the running totals."""
    api = FakeApi()
    pm = FakeMetadataPM(
        [
            _frame_line(
                frame_num=0,
                objects=[_person(tracker_id="0", roi_labels=["entry-roi"])],
                roi_counts={"entry-roi": 1},
            ),
            _frame_line(
                frame_num=1,
                objects=[_person(tracker_id="0", roi_labels=[])],
                roi_counts={"entry-roi": 0},
            ),
            _frame_line(
                frame_num=2,
                objects=[_person(tracker_id="0", roi_labels=["entry-roi"])],
                roi_counts={"entry-roi": 1},
            ),
        ],
        exit_code=124,
    )
    w = _analytics_worker(api, fast_settings, pm)
    await asyncio.wait_for(w.run(), timeout=3)
    stopped = next(s for s in api.statuses if s.status == "stopped")
    p = stopped.payload
    # Two enters (initial + re-enter) and one exit.
    assert p["zone_entered_observations_emitted"] == 2
    assert p["zone_exited_observations_emitted"] == 1


# --- multi-mode coexistence -------------------------------------------


async def test_combined_frame_emits_line_roi_and_zone_independently(fast_settings) -> None:  # type: ignore[no-untyped-def]
    """One frame can produce a line-crossing observation, an ROI
    occupancy observation, AND a zone_entered observation
    simultaneously. The three emit paths are independent."""
    api = FakeApi()
    pm = FakeMetadataPM(
        [
            # First frame: line cum=0, roi=0 (first ROI obs), no tracker
            # inside zone yet — but we set tracker present + outside so
            # zone state is None+False (no-op).
            _frame_line(
                frame_num=0,
                objects=[_person(tracker_id="0", roi_labels=[])],
                line_crossing_cumulative={"entry-line": 0},
                roi_counts={"entry-roi": 0},
            ),
            # Second frame: line cum bumps to 1 (one line crossing
            # observation), ROI count goes to 1 (one ROI observation
            # for the count change), and the tracked object enters the
            # ROI (one zone_entered observation).
            _frame_line(
                frame_num=1,
                objects=[_person(tracker_id="0", roi_labels=["entry-roi"])],
                line_crossing_cumulative={"entry-line": 1},
                roi_counts={"entry-roi": 1},
            ),
        ],
        exit_code=124,
    )
    w = _analytics_worker(api, fast_settings, pm)
    await asyncio.wait_for(w.run(), timeout=3)
    types = sorted({o.type for o in api.observations})
    assert "processor.line_crossing" in types
    assert "processor.roi_occupancy" in types
    assert "processor.zone_entered" in types


async def test_re_enter_after_exit_sequence_increments(fast_settings) -> None:  # type: ignore[no-untyped-def]
    """The worker-level analog of the adapter test for re-enter
    sequencing — confirms the per-(track, zone, type) counter rides
    through the worker without being overwritten."""
    api = FakeApi()
    pm = FakeMetadataPM(
        [
            _frame_line(
                frame_num=0,
                objects=[_person(tracker_id="0", roi_labels=["entry-roi"])],
                roi_counts={"entry-roi": 1},
            ),
            _frame_line(
                frame_num=1,
                objects=[_person(tracker_id="0", roi_labels=[])],
                roi_counts={"entry-roi": 0},
            ),
            _frame_line(
                frame_num=2,
                objects=[_person(tracker_id="0", roi_labels=["entry-roi"])],
                roi_counts={"entry-roi": 1},
            ),
            _frame_line(
                frame_num=3,
                objects=[_person(tracker_id="0", roi_labels=[])],
                roi_counts={"entry-roi": 0},
            ),
        ],
        exit_code=124,
    )
    w = _analytics_worker(api, fast_settings, pm)
    await asyncio.wait_for(w.run(), timeout=3)
    enters = _zone_obs(api, of_type="processor.zone_entered")
    exits = _zone_obs(api, of_type="processor.zone_exited")
    assert [o.payload["sequence"] for o in enters] == [1, 2]
    assert [o.payload["sequence"] for o in exits] == [1, 2]
    # Keys all distinct.
    all_keys = {o.payload["idempotency_key"] for o in enters + exits}
    assert len(all_keys) == 4


# --- P4-D follow-up: live-debug diagnostics ----------------------------


async def test_status_payload_exposes_analytics_class_ids_and_margin(fast_settings) -> None:  # type: ignore[no-untyped-def]
    """P4-D live didn't emit zone events because the per-object
    ``roi_labels`` was never attached (operator's foot point was below
    the ROI's y-max). The diagnostic fields make this triage trivial:
    the operator can see the configured class filter + margin and the
    per-class object counts in the status payload."""
    settings = fast_settings.model_copy(
        update={
            "deepstream_analytics_class_ids": "all",
            "deepstream_analytics_roi_margin_pct": 0.05,
        }
    )
    api = FakeApi()
    pm = FakeMetadataPM(
        [
            _frame_line(
                frame_num=0,
                objects=[
                    _person(tracker_id="0", roi_labels=["entry-roi"]),
                    _person(tracker_id="1", roi_labels=[]),
                ],
                roi_counts={"entry-roi": 1},
            ),
        ],
        exit_code=124,
    )
    w = _analytics_worker(api, settings, pm)
    await asyncio.wait_for(w.run(), timeout=3)
    stopped = next(s for s in api.statuses if s.status == "stopped")
    p = stopped.payload
    assert p["analytics_class_ids"] == "all"
    assert p["analytics_roi_margin_pct"] == 0.05
    # Two objects of class_id=2 (person, per the _person helper).
    assert p["analytics_object_class_id_counts"] == {"2": 2}
    # One object had non-empty roi_labels.
    assert p["objects_with_roi_labels_count"] == 1
    assert "entry-roi" in p["roi_labels_sample"]


async def test_status_payload_exposes_zero_roi_labels_when_geometry_misses(  # type: ignore[no-untyped-def]
    fast_settings,
) -> None:
    """Mirror of the P4-D live failure mode: objects detected but
    nvdsanalytics never tags them with a ROI label. Diagnostics show
    detection class counts > 0 BUT
    ``objects_with_roi_labels_count == 0`` — the triage signal that
    classifies the issue as 'analytics metadata' (not adapter)."""
    api = FakeApi()
    pm = FakeMetadataPM(
        [
            _frame_line(
                frame_num=0,
                objects=[_person(tracker_id="0", roi_labels=[])],
                roi_counts={"entry-roi": 0},
            ),
        ],
        exit_code=124,
    )
    w = _analytics_worker(api, fast_settings, pm)
    await asyncio.wait_for(w.run(), timeout=3)
    stopped = next(s for s in api.statuses if s.status == "stopped")
    p = stopped.payload
    assert p["analytics_object_class_id_counts"] == {"2": 1}
    assert p["objects_with_roi_labels_count"] == 0
    # No zone events fired — correct adapter behavior when roi_labels
    # is absent.
    assert _zone_obs(api) == []


async def test_overridden_class_ids_flows_to_status_payload(fast_settings) -> None:  # type: ignore[no-untyped-def]
    """Env override (e.g. ``UPSTREAM_DEEPSTREAM_ANALYTICS_CLASS_IDS=2``)
    propagates into the status payload."""
    settings = fast_settings.model_copy(update={"deepstream_analytics_class_ids": "2"})
    api = FakeApi()
    pm = FakeMetadataPM(
        [
            _frame_line(
                frame_num=0,
                objects=[_person(tracker_id="0", roi_labels=["entry-roi"])],
                roi_counts={"entry-roi": 1},
            ),
        ],
        exit_code=124,
    )
    w = _analytics_worker(api, settings, pm)
    await asyncio.wait_for(w.run(), timeout=3)
    stopped = next(s for s in api.statuses if s.status == "stopped")
    assert stopped.payload["analytics_class_ids"] == "2"
