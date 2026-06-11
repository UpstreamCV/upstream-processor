"""Side task — Workflow Condition Payload Enrichment.

Worker-level (in-process, no Docker/GPU/pyds) tests that the analytics
worker stamps ONLY provable fields onto observation payloads:

- zone_entered / zone_exited gain ``confidence`` from the object in the
  frame that committed the transition; omitted when the producer had none.
- dwell_started carries the zone-entry frame's confidence; dwell_updated /
  dwell_exceeded carry the CURRENT tick frame's confidence and omit it when
  the track isn't detected in that frame (a stale entry-time value is never
  re-stamped).
- roi_occupancy ``class_counts`` is populated from per-object
  ``analytics.roi_labels`` x class labels; unlabeled / metadata-less
  objects are excluded (the breakdown may sum below the nvdsanalytics
  aggregate ``total_count``) and the breakdown is empty — never guessed —
  when no object carries usable metadata.

Backward compatibility is covered by the pre-existing emit suites (which
build payloads without the new fields) continuing to pass unchanged.
"""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime, timedelta
from typing import Any

from app.observations.schemas import EligibleSession, ObservationPayload
from app.observations.writer import ObservationWriter
from app.packages.resolver import resolve
from app.runtimes.deepstream_runtime.worker import DeepStreamWorker

from tests._fakes import FakeApi
from tests.conftest import eligible_session_dict
from tests.test_deepstream_metadata_probe import FakeMetadataPM

_URL = "rtsp://127.0.0.1:18554/sess_abcdef0123456789"
_PROCESSOR_ID = "proc_test_001"
_BASE_TIME = datetime(2026, 6, 10, 16, 0, 0, tzinfo=UTC)


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
    roi_labels: list[str] | None = None,
) -> dict[str, Any]:
    """One object dict in the shape that lands in ``frame.objects[i]``
    after pyds emit + NeutralFrameMetadata validation. Unlike the other
    suites' ``_person`` helper this one lets ``label`` / ``conf`` be None
    so the omission paths can be exercised."""
    obj: dict[str, Any] = {
        "class_id": class_id,
        "label": label,
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


# --- zone transitions ---------------------------------------------------


async def test_zone_entered_includes_confidence(fast_settings) -> None:  # type: ignore[no-untyped-def]
    api = FakeApi()
    pm = FakeMetadataPM(
        [
            _frame_line(
                frame_num=0,
                objects=[_obj(tracker_id="7", conf=0.8612, roi_labels=["dock"])],
                roi_counts={"dock": 1},
                timestamp=_now_iso(0),
            ),
        ],
        exit_code=124,
    )
    await asyncio.wait_for(_analytics_worker(api, fast_settings, pm).run(), timeout=3)
    enters = _obs_of(api, "processor.zone_entered")
    assert len(enters) == 1
    assert enters[0].payload["confidence"] == 0.8612
    assert enters[0].payload["class_label"] == "person"
    assert enters[0].payload["track_id"] == "7"


async def test_zone_entered_omits_confidence_when_unavailable(fast_settings) -> None:  # type: ignore[no-untyped-def]
    """No detector confidence on the object → field omitted, not guessed."""
    api = FakeApi()
    pm = FakeMetadataPM(
        [
            _frame_line(
                frame_num=0,
                objects=[_obj(tracker_id="7", conf=None, roi_labels=["dock"])],
                roi_counts={"dock": 1},
                timestamp=_now_iso(0),
            ),
        ],
        exit_code=124,
    )
    await asyncio.wait_for(_analytics_worker(api, fast_settings, pm).run(), timeout=3)
    enters = _obs_of(api, "processor.zone_entered")
    assert len(enters) == 1
    assert "confidence" not in enters[0].payload


async def test_zone_exited_carries_exit_frame_confidence(fast_settings) -> None:  # type: ignore[no-untyped-def]
    """The exit commits on the first stable OUTSIDE frame — confidence
    comes from that frame's detection, not the entry frame's."""
    api = FakeApi()
    pm = FakeMetadataPM(
        [
            _frame_line(
                frame_num=0,
                objects=[_obj(tracker_id="7", conf=0.9, roi_labels=["dock"])],
                roi_counts={"dock": 1},
                timestamp=_now_iso(0),
            ),
            _frame_line(
                frame_num=1,
                objects=[_obj(tracker_id="7", conf=0.62, roi_labels=[])],
                roi_counts={"dock": 0},
                timestamp=_now_iso(1),
            ),
        ],
        exit_code=124,
    )
    await asyncio.wait_for(_analytics_worker(api, fast_settings, pm).run(), timeout=3)
    exits = _obs_of(api, "processor.zone_exited")
    assert len(exits) == 1
    assert exits[0].payload["confidence"] == 0.62


# --- dwell --------------------------------------------------------------


async def test_dwell_started_carries_entry_confidence(fast_settings) -> None:  # type: ignore[no-untyped-def]
    api = FakeApi()
    pm = FakeMetadataPM(
        [
            _frame_line(
                frame_num=0,
                objects=[_obj(tracker_id="7", conf=0.77, roi_labels=["dock"])],
                roi_counts={"dock": 1},
                timestamp=_now_iso(0),
            ),
        ],
        exit_code=124,
    )
    await asyncio.wait_for(_analytics_worker(api, fast_settings, pm).run(), timeout=3)
    started = _obs_of(api, "processor.dwell_started")
    assert len(started) == 1
    assert started[0].payload["confidence"] == 0.77


async def test_dwell_exceeded_carries_current_frame_confidence(fast_settings) -> None:  # type: ignore[no-untyped-def]
    """The exceeded event stamps the confidence from the frame whose tick
    crossed the threshold — proving freshness, not the entry value."""
    settings = fast_settings.model_copy(
        update={
            "deepstream_dwell_update_interval_seconds": 600.0,
            "deepstream_dwell_exceeded_threshold_seconds": 30.0,
        }
    )
    api = FakeApi()
    pm = FakeMetadataPM(
        [
            _frame_line(
                frame_num=0,
                objects=[_obj(tracker_id="7", conf=0.9, roi_labels=["dock"])],
                roi_counts={"dock": 1},
                timestamp=_now_iso(0),
            ),
            _frame_line(
                frame_num=1,
                objects=[_obj(tracker_id="7", conf=0.55, roi_labels=["dock"])],
                roi_counts={"dock": 1},
                timestamp=_now_iso(31),
            ),
        ],
        exit_code=124,
    )
    await asyncio.wait_for(_analytics_worker(api, settings, pm).run(), timeout=3)
    exceeded = _obs_of(api, "processor.dwell_exceeded")
    assert len(exceeded) == 1
    assert exceeded[0].payload["confidence"] == 0.55  # tick frame, not 0.9


async def test_dwell_exceeded_omits_confidence_when_track_undetected(fast_settings) -> None:  # type: ignore[no-untyped-def]
    """Disappearance doesn't close the session (P4-E invariant), but an
    undetected track has no current confidence — omit, never re-stamp the
    stale entry value."""
    settings = fast_settings.model_copy(
        update={
            "deepstream_dwell_update_interval_seconds": 600.0,
            "deepstream_dwell_exceeded_threshold_seconds": 30.0,
        }
    )
    api = FakeApi()
    pm = FakeMetadataPM(
        [
            _frame_line(
                frame_num=0,
                objects=[_obj(tracker_id="7", conf=0.9, roi_labels=["dock"])],
                roi_counts={"dock": 1},
                timestamp=_now_iso(0),
            ),
            # Track 7 vanishes; the dwell session stays open and the tick
            # still fires the exceeded event.
            _frame_line(
                frame_num=1,
                objects=[],
                roi_counts={"dock": 0},
                timestamp=_now_iso(31),
            ),
        ],
        exit_code=124,
    )
    await asyncio.wait_for(_analytics_worker(api, settings, pm).run(), timeout=3)
    exceeded = _obs_of(api, "processor.dwell_exceeded")
    assert len(exceeded) == 1
    assert "confidence" not in exceeded[0].payload


# --- roi occupancy class_counts ------------------------------------------


async def test_roi_class_counts_from_object_metadata(fast_settings) -> None:  # type: ignore[no-untyped-def]
    api = FakeApi()
    pm = FakeMetadataPM(
        [
            _frame_line(
                frame_num=0,
                objects=[
                    _obj(tracker_id="1", label="person", roi_labels=["dock"]),
                    _obj(tracker_id="2", label="person", roi_labels=["dock"]),
                    _obj(tracker_id="3", label="car", class_id=0, roi_labels=["dock"]),
                    _obj(tracker_id="4", label="person", roi_labels=["yard"]),
                ],
                roi_counts={"dock": 3, "yard": 1},
                timestamp=_now_iso(0),
            ),
        ],
        exit_code=124,
    )
    await asyncio.wait_for(_analytics_worker(api, fast_settings, pm).run(), timeout=3)
    rois = {o.payload["roi_id"]: o.payload for o in _obs_of(api, "processor.roi_occupancy")}
    assert rois["dock"]["total_count"] == 3
    assert rois["dock"]["class_counts"] == {"person": 2, "car": 1}
    assert rois["yard"]["class_counts"] == {"person": 1}


async def test_roi_class_counts_exclude_unlabeled_objects(fast_settings) -> None:  # type: ignore[no-untyped-def]
    """Unlabeled objects stay in the aggregate total but are excluded from
    the breakdown — the breakdown may sum BELOW total_count, never padded."""
    api = FakeApi()
    pm = FakeMetadataPM(
        [
            _frame_line(
                frame_num=0,
                objects=[
                    _obj(tracker_id="1", label="person", roi_labels=["dock"]),
                    _obj(tracker_id="2", label=None, roi_labels=["dock"]),
                ],
                roi_counts={"dock": 2},
                timestamp=_now_iso(0),
            ),
        ],
        exit_code=124,
    )
    await asyncio.wait_for(_analytics_worker(api, fast_settings, pm).run(), timeout=3)
    rois = _obs_of(api, "processor.roi_occupancy")
    assert len(rois) == 1
    assert rois[0].payload["total_count"] == 2
    assert rois[0].payload["class_counts"] == {"person": 1}


async def test_roi_class_counts_empty_when_no_object_metadata(fast_settings) -> None:  # type: ignore[no-untyped-def]
    """Aggregate counters with no per-object analytics metadata → empty
    breakdown (the pre-enrichment wire shape), never guessed."""
    api = FakeApi()
    pm = FakeMetadataPM(
        [
            _frame_line(
                frame_num=0,
                objects=[_obj(tracker_id="1", label="person")],  # no roi_labels
                roi_counts={"dock": 1},
                timestamp=_now_iso(0),
            ),
        ],
        exit_code=124,
    )
    await asyncio.wait_for(_analytics_worker(api, fast_settings, pm).run(), timeout=3)
    rois = _obs_of(api, "processor.roi_occupancy")
    assert len(rois) == 1
    assert rois[0].payload["class_counts"] == {}


async def test_roi_class_mix_change_at_constant_total_emits(fast_settings) -> None:  # type: ignore[no-untyped-def]
    """Pin the cadence consequence of populating class_counts: the change
    detector compares the WHOLE snapshot (total + breakdown), so a class
    mix change at constant total is a real, emitted change."""
    api = FakeApi()
    pm = FakeMetadataPM(
        [
            _frame_line(
                frame_num=0,
                objects=[
                    _obj(tracker_id="1", label="person", roi_labels=["dock"]),
                    _obj(tracker_id="2", label="person", roi_labels=["dock"]),
                ],
                roi_counts={"dock": 2},
                timestamp=_now_iso(0),
            ),
            _frame_line(
                frame_num=1,
                objects=[
                    _obj(tracker_id="1", label="person", roi_labels=["dock"]),
                    _obj(tracker_id="3", label="car", class_id=0, roi_labels=["dock"]),
                ],
                roi_counts={"dock": 2},
                timestamp=_now_iso(1),
            ),
        ],
        exit_code=124,
    )
    await asyncio.wait_for(_analytics_worker(api, fast_settings, pm).run(), timeout=3)
    rois = _obs_of(api, "processor.roi_occupancy")
    assert len(rois) == 2
    assert rois[1].payload["class_counts"] == {"person": 1, "car": 1}
    assert rois[1].payload["debug"]["emit_reason"] == "change"
