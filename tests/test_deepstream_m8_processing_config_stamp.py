"""M8 — DeepStream worker end-to-end ``applied_processing_config`` stamping.

ADDITIVE METADATA ONLY: M8 stamps the resolver's ``status_payload()`` (the SAME
requested-vs-applied lineage block M7B/C/D already stamp on processor STATUS)
onto two more output surfaces — the perception snapshot's ``source`` block and
the line/ROI/zone/dwell observation ``source`` blocks. NO runtime behavior
changes: object counts, M7D selected_classes filtering, M7C confidence apply,
geometry lineage, aggregate line/ROI counts, and the tracker field are all
untouched. The single source of truth is reused (no per-path recompute / no
drift), so status, snapshot, and every observation carry the identical block.

In-process only (no Docker / GPU / pyds): ``FakeMetadataPM`` feeds scripted
frames, ``FakeApi`` records the posts + observations, the worker runs end-to-end
through ``_drive_metadata_probe``. Mirrors ``test_perception_emit.py`` /
``test_deepstream_selected_classes_emit.py`` / the geometry/zone/dwell/line/roi
emit tests.

The honest ``selected_classes_analytics_counts_not_filtered`` caveat warning is
already inside ``status_payload().warnings`` when selected_classes is applied; it
rides along automatically into the stamped block. These tests assert it is
PRESENT while the aggregate count value is UNAFFECTED.
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


def _processing_config(
    *,
    selected_classes: list[str] | None = None,
    confidence_threshold: float | None = None,
    revision: int = 7,
) -> dict[str, Any]:
    """One ``{revision, config}`` envelope (every other field at its default),
    in the shape the api-server saves and the resolver parses."""
    return {
        "revision": revision,
        "config": {
            "version": 1,
            "model_key": "trafficcamnet_v1",
            "selected_classes": list(selected_classes or []),
            "tracker": None,
            "confidence_threshold": confidence_threshold,
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
    processing_config: dict[str, Any] | None = None,
) -> DeepStreamWorker:
    return DeepStreamWorker(
        session=_session(processing_config=processing_config),
        settings=settings,
        api=api,
        package=resolve("deepstream.analytics_probe.v0"),
        writer=ObservationWriter(api, _PROCESSOR_ID),
        pm_factory=lambda: pm,
    )


def _obs(api: FakeApi, of_type: str) -> list[ObservationPayload]:
    return [o for o in api.observations if o.type == of_type]


def _stopped_payload(api: FakeApi) -> dict[str, Any]:
    return next(s for s in api.statuses if s.status == "stopped").payload


# ===========================================================================
# Perception snapshot carries the lineage (and it matches the STATUS stamp)
# ===========================================================================


async def test_snapshot_source_carries_applied_processing_config_with_selected_and_confidence(
    fast_settings: Any,
) -> None:
    """selected_classes=["person"] + confidence=0.55 on an analytics worker:
    the perception snapshot's ``source.applied_processing_config`` echoes the
    applied selected_classes / confidence and carries the honest
    analytics-counts caveat warning."""
    api = FakeApi()
    pm = FakeMetadataPM([_frame(frame_num=0, objects=[_person(tracker_id="0")])], exit_code=124)
    w = _analytics_worker(
        api,
        fast_settings,
        pm,
        processing_config=_processing_config(
            selected_classes=["person"], confidence_threshold=0.55
        ),
    )
    await asyncio.wait_for(w.run(), timeout=3)

    assert len(api.perception_snapshots) >= 1
    block = api.perception_snapshots[0].snapshot["source"]["applied_processing_config"]
    assert block["status"] == "partially_applied"
    assert block["applied"]["selected_classes"] == ["person"]
    assert block["applied"]["confidence_threshold"] == 0.55
    assert "selected_classes_analytics_counts_not_filtered" in block["warnings"]
    assert "selected_classes_requested_not_applied" not in block["warnings"]


async def test_snapshot_lineage_is_identical_to_status_stamp(fast_settings: Any) -> None:
    """Single source of truth / no drift: the block stamped on the perception
    snapshot is the IDENTICAL ``status_payload()`` dict M7B/C/D stamp on the
    processor status payload."""
    api = FakeApi()
    pm = FakeMetadataPM([_frame(frame_num=0, objects=[_person(tracker_id="0")])], exit_code=124)
    w = _analytics_worker(
        api,
        fast_settings,
        pm,
        processing_config=_processing_config(
            selected_classes=["person"], confidence_threshold=0.55, revision=12
        ),
    )
    await asyncio.wait_for(w.run(), timeout=3)

    snapshot_block = api.perception_snapshots[0].snapshot["source"]["applied_processing_config"]
    status_block = _stopped_payload(api)["applied_processing_config"]
    assert snapshot_block == status_block


async def test_snapshot_default_config_carries_default_lineage(fast_settings: Any) -> None:
    """A no-config / default worker still carries ``applied_processing_config``
    in the snapshot source — with source=default / status=applied_defaults and
    NO selected_classes warning (the block is now always present)."""
    api = FakeApi()
    pm = FakeMetadataPM([_frame(frame_num=0, objects=[_person(tracker_id="0")])], exit_code=124)
    w = _analytics_worker(api, fast_settings, pm, processing_config=None)
    await asyncio.wait_for(w.run(), timeout=3)

    block = api.perception_snapshots[0].snapshot["source"]["applied_processing_config"]
    assert block["source"] == "default"
    assert block["status"] == "applied_defaults"
    assert block["revision"] is None
    assert block["applied"]["selected_classes"] == []
    assert "selected_classes_analytics_counts_not_filtered" not in block["warnings"]


# ===========================================================================
# Observations: line / ROI / zone / dwell all carry the lineage (same block)
# ===========================================================================


async def test_zone_and_dwell_observation_source_carries_applied_processing_config(
    fast_settings: Any,
) -> None:
    """The per-object zone (and therefore dwell) observation source block
    carries ``applied_processing_config`` — the same single source_block is
    threaded into every emit."""
    api = FakeApi()
    pm = FakeMetadataPM(
        [
            _frame(
                frame_num=0,
                objects=[_person(tracker_id="0", roi_labels=["entry-roi"])],
                roi_counts={"entry-roi": 1},
            ),
        ],
        exit_code=124,
    )
    w = _analytics_worker(
        api,
        fast_settings,
        pm,
        processing_config=_processing_config(selected_classes=["person"]),
    )
    await asyncio.wait_for(w.run(), timeout=3)

    enters = _obs(api, "processor.zone_entered")
    assert len(enters) >= 1
    block = enters[0].payload["source"]["applied_processing_config"]
    assert block["applied"]["selected_classes"] == ["person"]
    assert "selected_classes_analytics_counts_not_filtered" in block["warnings"]

    # Dwell rides on the same source block (if it fired).
    for d in _obs(api, "processor.dwell_started"):
        d_block = d.payload["source"]["applied_processing_config"]
        assert d_block["applied"]["selected_classes"] == ["person"]


async def test_line_and_roi_observations_carry_lineage_while_counts_unaffected(
    fast_settings: Any,
) -> None:
    """The line-crossing and ROI-occupancy observation source blocks ALSO carry
    ``applied_processing_config`` (same single source_block) — AND the aggregate
    counts themselves are unchanged: a car-only frame (the only class is the
    UNSELECTED car) still emits both the line crossing and the ROI occupancy,
    proving the aggregate path is all-class while the stamped block honestly
    discloses that via the caveat warning."""
    api = FakeApi()
    pm = FakeMetadataPM(
        [
            _frame(
                frame_num=0,
                objects=[_car(tracker_id="1", roi_labels=["entry-roi"])],
                line_crossing_cumulative={"entry-line": 0},
                roi_counts={"entry-roi": 1},
            ),
            _frame(
                frame_num=1,
                objects=[_car(tracker_id="1", roi_labels=["entry-roi"])],
                line_crossing_cumulative={"entry-line": 1},
                roi_counts={"entry-roi": 1},
            ),
        ],
        exit_code=124,
    )
    w = _analytics_worker(
        api,
        fast_settings,
        pm,
        processing_config=_processing_config(selected_classes=["person"]),
    )
    await asyncio.wait_for(w.run(), timeout=3)

    line_obs = _obs(api, "processor.line_crossing")
    roi_obs = _obs(api, "processor.roi_occupancy")
    # Aggregate counts UNAFFECTED by the person-only filter: the car-only
    # frames still produced the crossing + the occupancy.
    assert len(line_obs) == 1
    assert len(roi_obs) >= 1

    line_block = line_obs[0].payload["source"]["applied_processing_config"]
    roi_block = roi_obs[0].payload["source"]["applied_processing_config"]
    for block in (line_block, roi_block):
        assert block["applied"]["selected_classes"] == ["person"]
        # The honest caveat is present in the stamped block...
        assert "selected_classes_analytics_counts_not_filtered" in block["warnings"]
    # ...while the COUNT value itself is the all-class aggregate (entry-line
    # incremented to 1 by the car, entry-roi total of 1 car).
    assert line_obs[0].payload["cumulative_count"] == 1
    assert roi_obs[0].payload["total_count"] == 1


async def test_all_observation_source_blocks_share_identical_lineage(fast_settings: Any) -> None:
    """No drift across output paths: the line / ROI / zone observations all
    carry the SAME ``applied_processing_config`` block (built once and threaded
    into every emit), and it equals the status stamp."""
    api = FakeApi()
    pm = FakeMetadataPM(
        [
            _frame(
                frame_num=0,
                objects=[_person(tracker_id="0", roi_labels=["entry-roi"])],
                line_crossing_cumulative={"entry-line": 0},
                roi_counts={"entry-roi": 1},
            ),
            _frame(
                frame_num=1,
                objects=[_person(tracker_id="0", roi_labels=["entry-roi"])],
                line_crossing_cumulative={"entry-line": 1},
                roi_counts={"entry-roi": 1},
            ),
        ],
        exit_code=124,
    )
    w = _analytics_worker(
        api,
        fast_settings,
        pm,
        processing_config=_processing_config(selected_classes=["person"], revision=8),
    )
    await asyncio.wait_for(w.run(), timeout=3)

    status_block = _stopped_payload(api)["applied_processing_config"]
    blocks = [
        o.payload["source"]["applied_processing_config"]
        for o in api.observations
        if o.type
        in (
            "processor.line_crossing",
            "processor.roi_occupancy",
            "processor.zone_entered",
        )
    ]
    assert blocks, "expected line/ROI/zone observations to be emitted"
    for block in blocks:
        assert block == status_block


# ===========================================================================
# Regression: M7C / M7D / geometry / tracker behavior is UNCHANGED by M8
# ===========================================================================


async def test_snapshot_objects_still_filtered_to_selected_class(fast_settings: Any) -> None:
    """M7D object filtering on the snapshot still works under M8: a mixed
    person+car frame yields a snapshot whose ``objects`` contains ONLY the
    person — the additive lineage stamp did not touch the filter."""
    api = FakeApi()
    pm = FakeMetadataPM(
        [_frame(frame_num=0, objects=[_person(tracker_id="0"), _car(tracker_id="1")])],
        exit_code=124,
    )
    w = _analytics_worker(
        api,
        fast_settings,
        pm,
        processing_config=_processing_config(selected_classes=["person"]),
    )
    await asyncio.wait_for(w.run(), timeout=3)

    body = api.perception_snapshots[0].snapshot
    assert [o["label"] for o in body["objects"]] == ["person"]
    assert [o["class_id"] for o in body["objects"]] == [2]


async def test_snapshot_geometry_lineage_and_tracker_unchanged(fast_settings: Any) -> None:
    """Geometry lineage (the ``geometry`` block + the source
    geometry_source/geometry_revision) and the tracker source field are
    unchanged by the additive processing-config stamp."""
    api = FakeApi()
    pm = FakeMetadataPM([_frame(frame_num=0, objects=[_person(tracker_id="0")])], exit_code=124)
    w = _analytics_worker(
        api,
        fast_settings,
        pm,
        processing_config=_processing_config(selected_classes=["person"]),
    )
    await asyncio.wait_for(w.run(), timeout=3)

    snap = api.perception_snapshots[0].snapshot
    # Geometry block is the probe-default analytics geometry (unchanged).
    assert snap["geometry"]["rois"][0]["id"] == "entry-roi"
    assert snap["geometry"]["lines"][0]["id"] == "entry-line"
    # Tracker source field unchanged.
    assert snap["source"]["tracker_algorithm"] == "iou"

    # Observation source blocks: geometry lineage (no operator geometry, so
    # source="default" + omitted revision) unchanged, tracker unchanged.
    line_obs = _obs(api, "processor.line_crossing")
    if line_obs:
        src = line_obs[0].payload["source"]
        assert src["tracker_algorithm"] == "iou"
        assert src["geometry_source"] == "default"
        assert "geometry_revision" not in src


async def test_status_stamp_unchanged_block_still_present(fast_settings: Any) -> None:
    """M7B/C/D status stamping is unregressed: the status payload still carries
    the SAME ``applied_processing_config`` block (M8 only ADDS it to other
    surfaces, it does not move or alter the status stamp)."""
    api = FakeApi()
    pm = FakeMetadataPM([_frame(frame_num=0, objects=[_person(tracker_id="0")])], exit_code=124)
    w = _analytics_worker(
        api,
        fast_settings,
        pm,
        processing_config=_processing_config(selected_classes=["person"]),
    )
    await asyncio.wait_for(w.run(), timeout=3)

    block = _stopped_payload(api)["applied_processing_config"]
    assert block["status"] == "partially_applied"
    assert block["applied"]["selected_classes"] == ["person"]
    assert "selected_classes_analytics_counts_not_filtered" in block["warnings"]


async def test_empty_selected_classes_keeps_default_lineage_and_all_objects(
    fast_settings: Any,
) -> None:
    """Regression: an explicit empty selected_classes ([]) is the no-filter
    path. The snapshot still carries EVERY object AND the now-always-present
    applied_processing_config block (status applied_defaults, no caveat
    warning, applied.selected_classes == [])."""
    api = FakeApi()
    pm = FakeMetadataPM(
        [_frame(frame_num=0, objects=[_person(tracker_id="0"), _car(tracker_id="1")])],
        exit_code=124,
    )
    w = _analytics_worker(
        api,
        fast_settings,
        pm,
        processing_config=_processing_config(selected_classes=[]),
    )
    await asyncio.wait_for(w.run(), timeout=3)

    body = api.perception_snapshots[0].snapshot
    # No filter → both objects present (byte/behavior identical to today).
    assert sorted(o["label"] for o in body["objects"]) == ["car", "person"]
    block = body["source"]["applied_processing_config"]
    assert block["status"] == "applied_defaults"
    assert block["applied"]["selected_classes"] == []
    assert "selected_classes_analytics_counts_not_filtered" not in block["warnings"]


async def test_worker_runs_to_completion_with_stamp(fast_settings: Any) -> None:
    """Smoke: the additive stamp never disturbs the worker lifecycle — it ends
    STOPPED exactly as before."""
    api = FakeApi()
    pm = FakeMetadataPM([_frame(frame_num=0, objects=[_person(tracker_id="0")])], exit_code=124)
    w = _analytics_worker(
        api,
        fast_settings,
        pm,
        processing_config=_processing_config(selected_classes=["person"]),
    )
    await asyncio.wait_for(w.run(), timeout=3)
    assert w.state is WorkerState.STOPPED
