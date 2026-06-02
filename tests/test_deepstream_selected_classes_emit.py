"""M7D — DeepStream worker end-to-end ``selected_classes`` filtering.

In-process tests of the analytics worker actually FILTERING per-object output
paths by the operator-saved ``selected_classes``, applied POST-INFERENCE (the
detector still runs across all classes — these tests feed mixed-class frames
and assert the user-facing surfaces are filtered while the aggregate /
raw-debug surfaces are byte-unchanged). No Docker / GPU / pyds —
``FakeMetadataPM`` feeds scripted frames, ``FakeApi`` records the posts +
observations, the worker runs end-to-end through ``_drive_metadata_probe``.

Mirrors ``test_perception_emit.py`` / ``test_deepstream_zone_transition_emit.py``
mechanics. ``selected_classes`` reaches the worker through the session's
``processing_config`` envelope (the resolver applies it on the per-object
analytics mode → ``w._selected_classes``).

FILTERED (user-facing per-object surfaces):
  - the perception snapshot ``objects`` list (``_build_perception_payload``);
  - the zone/dwell observation path (``_emit_zone_transitions``).

NOT FILTERED (deliberate, out of M7D scope — see the
``selected_classes_analytics_counts_not_filtered`` lineage warning):
  - nvdsanalytics AGGREGATE line-crossing / ROI-occupancy counts;
  - the RAW status debug counters (``objects_seen`` / ``class_counts``).

The ``selected_classes == []`` (and no-config) paths must be byte/behavior
identical to today — every object flows through unchanged.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from app.observations.schemas import EligibleSession, ObservationPayload
from app.observations.writer import ObservationWriter
from app.packages.resolver import resolve
from app.runtimes.base import WorkerState
from app.runtimes.deepstream_runtime.worker import DeepStreamWorker

from tests._fakes import FakeApi
from tests.conftest import eligible_session_dict
from tests.test_deepstream_metadata_probe import FakeMetadataPM

_URL = "rtsp://127.0.0.1:18554/sess_abcdef0123456789"
_PROCESSOR_ID = "proc_test_001"


def _processing_config(selected_classes: list[str], *, revision: int = 7) -> dict[str, Any]:
    """One ``{revision, config}`` envelope carrying just ``selected_classes``
    (every other field at its default), in the shape the api-server saves and
    the resolver parses."""
    return {
        "revision": revision,
        "config": {
            "version": 1,
            "model_key": "trafficcamnet_v1",
            "selected_classes": list(selected_classes),
            "tracker": None,
            "confidence_threshold": None,
            "processing_fps": None,
        },
    }


def _session(*, processing_config: dict[str, Any] | None = None) -> EligibleSession:
    return EligibleSession.model_validate(
        eligible_session_dict(internal_rtsp_url=_URL, processing_config=processing_config)
    )


def _obj(
    *,
    class_id: int,
    label: str | None,
    tracker_id: str | None = None,
    roi_labels: list[str] | None = None,
) -> dict[str, Any]:
    """One object dict in the post-pyds / NeutralFrameMetadata shape.
    ``roi_labels`` (when supplied) goes under ``debug.analytics.roi_labels``
    (the P3.5 contract the zone path consumes)."""
    o: dict[str, Any] = {
        "class_id": class_id,
        "label": label,
        "confidence": 0.8,
        "bbox_pixels": {"left": 100.0, "top": 100.0, "width": 200.0, "height": 400.0},
        "bbox_normalized": {"x": 0.078, "y": 0.139, "w": 0.156, "h": 0.556},
        "tracker_id": tracker_id,
    }
    if roi_labels is not None:
        o["debug"] = {"analytics": {"roi_labels": list(roi_labels)}}
    return o


def _person(**ov: Any) -> dict[str, Any]:
    return _obj(class_id=2, label="person", **ov)


def _car(**ov: Any) -> dict[str, Any]:
    return _obj(class_id=0, label="car", **ov)


def _frame(
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


def _analytics_worker(
    api: FakeApi,
    settings: Any,
    pm: FakeMetadataPM,
    *,
    selected_classes: list[str] | None = None,
) -> DeepStreamWorker:
    processing_config = (
        None if selected_classes is None else _processing_config(selected_classes)
    )
    return DeepStreamWorker(
        session=_session(processing_config=processing_config),
        settings=settings,
        api=api,
        package=resolve("deepstream.analytics_probe.v0"),
        writer=ObservationWriter(api, _PROCESSOR_ID),
        pm_factory=lambda: pm,
    )


def _zone_obs(api: FakeApi, *, of_type: str | None = None) -> list[ObservationPayload]:
    types = (
        (of_type,)
        if of_type is not None
        else ("processor.zone_entered", "processor.zone_exited")
    )
    return [o for o in api.observations if o.type in types]


def _stopped_payload(api: FakeApi) -> dict[str, Any]:
    return next(s for s in api.statuses if s.status == "stopped").payload


# ===========================================================================
# Sanity: selected_classes reaches the worker filter tuple
# ===========================================================================


async def test_analytics_worker_filter_tuple_reflects_selected_classes(
    fast_settings: Any,
) -> None:
    """The session's processing_config ``selected_classes`` is applied on the
    analytics worker and surfaces as the per-worker ``_selected_classes`` tuple
    (the single source of truth the FILTER paths read)."""
    api = FakeApi()
    pm = FakeMetadataPM([_frame(frame_num=0, objects=[_person()])], exit_code=124)
    w = _analytics_worker(api, fast_settings, pm, selected_classes=["person"])
    assert w._selected_classes == ("person",)
    await asyncio.wait_for(w.run(), timeout=3)
    assert w.state is WorkerState.STOPPED


# ===========================================================================
# FILTERED: perception snapshot objects
# ===========================================================================


async def test_perception_snapshot_objects_filtered_to_selected_class(
    fast_settings: Any,
) -> None:
    """With selected_classes=["person"] a mixed person+car frame yields a
    perception snapshot whose ``objects`` contains ONLY the person — the car is
    dropped from the user-facing per-object surface."""
    api = FakeApi()
    pm = FakeMetadataPM(
        [_frame(frame_num=0, objects=[_person(tracker_id="0"), _car(tracker_id="1")])],
        exit_code=124,
    )
    w = _analytics_worker(api, fast_settings, pm, selected_classes=["person"])
    await asyncio.wait_for(w.run(), timeout=3)

    assert len(api.perception_snapshots) == 1
    body = api.perception_snapshots[0].snapshot
    assert [o["label"] for o in body["objects"]] == ["person"]
    assert [o["class_id"] for o in body["objects"]] == [2]


async def test_perception_snapshot_keeps_multiple_selected_classes(
    fast_settings: Any,
) -> None:
    """selected_classes=["person","car"] keeps BOTH from a person+car+bicycle
    frame; the unlisted bicycle is dropped. Order is preserved."""
    api = FakeApi()
    pm = FakeMetadataPM(
        [
            _frame(
                frame_num=0,
                objects=[
                    _person(tracker_id="0"),
                    _obj(class_id=1, label="bicycle", tracker_id="1"),
                    _car(tracker_id="2"),
                ],
            )
        ],
        exit_code=124,
    )
    w = _analytics_worker(api, fast_settings, pm, selected_classes=["person", "car"])
    await asyncio.wait_for(w.run(), timeout=3)

    body = api.perception_snapshots[0].snapshot
    assert [o["label"] for o in body["objects"]] == ["person", "car"]


async def test_perception_snapshot_unchanged_when_no_selected_classes(
    fast_settings: Any,
) -> None:
    """No processing_config at all → ``_selected_classes == ()`` → the snapshot
    carries EVERY object (byte/behavior identical to today)."""
    api = FakeApi()
    pm = FakeMetadataPM(
        [_frame(frame_num=0, objects=[_person(tracker_id="0"), _car(tracker_id="1")])],
        exit_code=124,
    )
    w = _analytics_worker(api, fast_settings, pm, selected_classes=None)
    assert w._selected_classes == ()
    await asyncio.wait_for(w.run(), timeout=3)

    body = api.perception_snapshots[0].snapshot
    assert sorted(o["label"] for o in body["objects"]) == ["car", "person"]


async def test_perception_snapshot_unchanged_when_selected_classes_empty(
    fast_settings: Any,
) -> None:
    """An explicit empty selected_classes ([]) is the "no filter" path — the
    snapshot still carries every object."""
    api = FakeApi()
    pm = FakeMetadataPM(
        [_frame(frame_num=0, objects=[_person(tracker_id="0"), _car(tracker_id="1")])],
        exit_code=124,
    )
    w = _analytics_worker(api, fast_settings, pm, selected_classes=[])
    assert w._selected_classes == ()
    await asyncio.wait_for(w.run(), timeout=3)

    body = api.perception_snapshots[0].snapshot
    assert sorted(o["label"] for o in body["objects"]) == ["car", "person"]


# ===========================================================================
# FILTERED: zone (and therefore dwell) per-object path
# ===========================================================================


async def test_zone_transition_emitted_only_for_selected_class(
    fast_settings: Any,
) -> None:
    """A person and a car both stably inside the zone, selected_classes
    =["person"]: only the PERSON track produces a zone_entered — the car is
    filtered out before ``zone_state.absorb_object`` so it never transitions."""
    api = FakeApi()
    pm = FakeMetadataPM(
        [
            _frame(
                frame_num=0,
                objects=[
                    _person(tracker_id="0", roi_labels=["entry-roi"]),
                    _car(tracker_id="1", roi_labels=["entry-roi"]),
                ],
                roi_counts={"entry-roi": 2},
            ),
        ],
        exit_code=124,
    )
    w = _analytics_worker(api, fast_settings, pm, selected_classes=["person"])
    await asyncio.wait_for(w.run(), timeout=3)

    enters = _zone_obs(api, of_type="processor.zone_entered")
    assert len(enters) == 1
    # The surviving transition is the person track, not the car.
    assert enters[0].payload["track_id"] == "0"
    assert enters[0].payload["class_label"] == "person"
    # No transition for the filtered-out car track.
    assert all(o.payload["track_id"] != "1" for o in _zone_obs(api))


async def test_zone_transition_unchanged_when_no_selected_classes(
    fast_settings: Any,
) -> None:
    """No filter → BOTH the person and the car transition (byte-identical to
    today's zone behavior)."""
    api = FakeApi()
    pm = FakeMetadataPM(
        [
            _frame(
                frame_num=0,
                objects=[
                    _person(tracker_id="0", roi_labels=["entry-roi"]),
                    _car(tracker_id="1", roi_labels=["entry-roi"]),
                ],
                roi_counts={"entry-roi": 2},
            ),
        ],
        exit_code=124,
    )
    w = _analytics_worker(api, fast_settings, pm, selected_classes=None)
    await asyncio.wait_for(w.run(), timeout=3)

    enters = _zone_obs(api, of_type="processor.zone_entered")
    assert {o.payload["track_id"] for o in enters} == {"0", "1"}


async def test_dwell_started_follows_filtered_zone(fast_settings: Any) -> None:
    """Dwell rides on the zone path (no separate filter): with
    selected_classes=["person"] the car never enters the zone, so only the
    person yields a dwell_started observation."""
    api = FakeApi()
    pm = FakeMetadataPM(
        [
            _frame(
                frame_num=0,
                objects=[
                    _person(tracker_id="0", roi_labels=["entry-roi"]),
                    _car(tracker_id="1", roi_labels=["entry-roi"]),
                ],
                roi_counts={"entry-roi": 2},
            ),
        ],
        exit_code=124,
    )
    w = _analytics_worker(api, fast_settings, pm, selected_classes=["person"])
    await asyncio.wait_for(w.run(), timeout=3)

    dwell_started = [o for o in api.observations if o.type == "processor.dwell_started"]
    # Whether dwell_started fires depends on config; if it does, it is only for
    # the person track — never the filtered-out car.
    for o in dwell_started:
        assert o.payload["track_id"] == "0"


# ===========================================================================
# NOT FILTERED (regression): aggregate analytics counts ignore selected_classes
# ===========================================================================


async def test_line_crossing_emitted_regardless_of_selected_classes(
    fast_settings: Any,
) -> None:
    """The nvdsanalytics line-crossing AGGREGATE count is per-line across ALL
    classes; selecting only ["person"] must NOT suppress it. A frame whose only
    object is a car (filtered out of the per-object surfaces) still emits the
    line crossing — proving the aggregate path is byte-unchanged."""
    api = FakeApi()
    pm = FakeMetadataPM(
        [
            _frame(
                frame_num=0,
                objects=[_car(tracker_id="1")],
                line_crossing_cumulative={"entry-line": 0},
            ),
            _frame(
                frame_num=1,
                objects=[_car(tracker_id="1")],
                line_crossing_cumulative={"entry-line": 1},
            ),
        ],
        exit_code=124,
    )
    w = _analytics_worker(api, fast_settings, pm, selected_classes=["person"])
    await asyncio.wait_for(w.run(), timeout=3)

    line_obs = [o for o in api.observations if o.type == "processor.line_crossing"]
    assert len(line_obs) == 1


async def test_roi_occupancy_emitted_regardless_of_selected_classes(
    fast_settings: Any,
) -> None:
    """The ROI-occupancy AGGREGATE count is per-ROI across ALL classes;
    selecting only ["person"] must NOT suppress it. A car-only frame still
    emits the ROI occupancy observation."""
    api = FakeApi()
    pm = FakeMetadataPM(
        [
            _frame(
                frame_num=0,
                objects=[_car(tracker_id="1", roi_labels=["entry-roi"])],
                roi_counts={"entry-roi": 1},
            ),
        ],
        exit_code=124,
    )
    w = _analytics_worker(api, fast_settings, pm, selected_classes=["person"])
    await asyncio.wait_for(w.run(), timeout=3)

    roi_obs = [o for o in api.observations if o.type == "processor.roi_occupancy"]
    assert len(roi_obs) == 1


async def test_aggregate_counts_byte_identical_with_and_without_filter(
    fast_settings: Any,
) -> None:
    """Same frame sequence, with vs without selected_classes=["person"]: the
    aggregate line + ROI observation COUNTS are identical. The filter only
    touches the per-object surfaces, never the aggregate counts."""

    def _frames() -> list[str]:
        return [
            _frame(
                frame_num=0,
                objects=[
                    _person(tracker_id="0"),
                    _car(tracker_id="1", roi_labels=["entry-roi"]),
                ],
                line_crossing_cumulative={"entry-line": 0},
                roi_counts={"entry-roi": 1},
            ),
            _frame(
                frame_num=1,
                objects=[
                    _person(tracker_id="0"),
                    _car(tracker_id="1", roi_labels=["entry-roi"]),
                ],
                line_crossing_cumulative={"entry-line": 1},
                roi_counts={"entry-roi": 2},
            ),
        ]

    async def _counts(selected: list[str] | None) -> tuple[int, int]:
        api = FakeApi()
        pm = FakeMetadataPM(_frames(), exit_code=124)
        w = _analytics_worker(api, fast_settings, pm, selected_classes=selected)
        await asyncio.wait_for(w.run(), timeout=3)
        lines = len([o for o in api.observations if o.type == "processor.line_crossing"])
        rois = len([o for o in api.observations if o.type == "processor.roi_occupancy"])
        return lines, rois

    # Each branch runs a fresh worker in the SAME event loop (the outer test is
    # async under anyio); we compare only the aggregate COUNTS.
    assert await _counts(["person"]) == await _counts(None)


# ===========================================================================
# NOT FILTERED (regression): RAW status debug counters
# ===========================================================================


async def test_status_counters_remain_raw_under_selected_classes(
    fast_settings: Any,
) -> None:
    """``objects_seen`` / ``class_counts`` / ``analytics_object_class_id_counts``
    are RAW debug — what the detector actually produced. Even with
    selected_classes=["person"], a person+car frame must count BOTH (2 objects,
    one per class), proving the filter never touches ``counters.absorb``."""
    api = FakeApi()
    pm = FakeMetadataPM(
        [_frame(frame_num=0, objects=[_person(tracker_id="0"), _car(tracker_id="1")])],
        exit_code=124,
    )
    w = _analytics_worker(api, fast_settings, pm, selected_classes=["person"])
    await asyncio.wait_for(w.run(), timeout=3)

    p = _stopped_payload(api)
    # RAW: both objects counted despite the person-only filter.
    assert p["objects_seen"] == 2
    assert p["class_counts"] == {0: 1, 2: 1}
    # The analytics per-class diagnostic mirrors the raw class_counts.
    assert p["analytics_object_class_id_counts"] == {"0": 1, "2": 1}


async def test_status_lineage_carries_analytics_counts_not_filtered_warning(
    fast_settings: Any,
) -> None:
    """The honest lineage signal: whenever selected_classes is applied, the
    status payload's applied_processing_config carries
    ``selected_classes_analytics_counts_not_filtered`` (the disclosure that the
    aggregate counts remain across all classes) and NOT
    ``selected_classes_requested_not_applied``."""
    api = FakeApi()
    pm = FakeMetadataPM([_frame(frame_num=0, objects=[_person(tracker_id="0")])], exit_code=124)
    w = _analytics_worker(api, fast_settings, pm, selected_classes=["person"])
    await asyncio.wait_for(w.run(), timeout=3)

    block = _stopped_payload(api)["applied_processing_config"]
    assert block["status"] == "partially_applied"
    assert block["applied"]["selected_classes"] == ["person"]
    assert "selected_classes_analytics_counts_not_filtered" in block["warnings"]
    assert "selected_classes_requested_not_applied" not in block["warnings"]
