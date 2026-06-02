"""P5-C — pure perception-snapshot builder.

``build_perception_snapshot`` converts one ``NeutralFrameMetadata`` frame into
the runtime-neutral snapshot dict. Fully defensive about missing per-object
fields; output is JSON-serializable.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from app.observations.metadata_models import (
    BoundingBoxNormalized,
    BoundingBoxPixels,
    NeutralFrame,
    NeutralFrameMetadata,
    NeutralObjectMetadata,
)
from app.perception.builder import build_perception_snapshot

_GEOM: dict[str, Any] = {
    "rois": [{"id": "entry-roi", "label": "entry-roi", "points": [{"x": 64, "y": 36}]}],
    "lines": [{"id": "entry-line", "label": "entry-line", "points": [{"x": 0, "y": 360}]}],
}
_OBSERVED = datetime(2026, 5, 21, 12, 0, 0, tzinfo=UTC)


def _frame(
    *,
    objects: list[NeutralObjectMetadata] | None = None,
    frame_num: int = 7,
    width: int = 1280,
    height: int = 720,
    source: dict[str, Any] | None = None,
) -> NeutralFrameMetadata:
    return NeutralFrameMetadata(
        runtime="deepstream",
        metadata_source="pyds_pad_probe",
        frame=NeutralFrame(frame_num=frame_num, width=width, height=height),
        objects=objects or [],
        source=source or {},
    )


def _build(metadata: NeutralFrameMetadata, *, max_objects: int = 50) -> dict[str, Any]:
    return build_perception_snapshot(
        metadata=metadata,
        geometry=_GEOM,
        camera_id=uuid4(),
        stream_session_id=uuid4(),
        processor_id="proc-1",
        package_id="deepstream.analytics_probe.v0",
        runtime="deepstream",
        metadata_source="pyds_pad_probe",
        analytics_plugin="nvdsanalytics",
        tracker_algorithm="iou",
        observed_at=_OBSERVED,
        max_objects=max_objects,
    )


def test_object_bbox_pixel_and_normalized_mapping() -> None:
    obj = NeutralObjectMetadata(
        class_id=2,
        label="person",
        confidence=0.82,
        tracker_id="0",
        bbox_pixels=BoundingBoxPixels(left=410, top=120, width=180, height=520),
        bbox_normalized=BoundingBoxNormalized(x=0.32, y=0.16, w=0.14, h=0.72),
        debug={"analytics": {"roi_labels": ["entry-roi"]}},
    )
    o = _build(_frame(objects=[obj]))["objects"][0]
    assert o["bbox"] == {"x": 410, "y": 120, "w": 180, "h": 520}
    assert o["bbox_normalized"] == {"x": 0.32, "y": 0.16, "w": 0.14, "h": 0.72}
    assert o["track_id"] == "0"
    assert o["class_id"] == 2
    assert o["label"] == "person"
    assert o["confidence"] == 0.82
    assert o["roi_labels"] == ["entry-roi"]


def test_missing_bbox_yields_none() -> None:
    o = _build(_frame(objects=[NeutralObjectMetadata(class_id=1)]))["objects"][0]
    assert o["bbox"] is None
    assert o["bbox_normalized"] is None


def test_missing_confidence_and_tracker_id_pass_through_as_none() -> None:
    o = _build(_frame(objects=[NeutralObjectMetadata(class_id=1)]))["objects"][0]
    assert o["confidence"] is None
    assert o["track_id"] is None


def test_roi_labels_default_empty_when_missing() -> None:
    o = _build(_frame(objects=[NeutralObjectMetadata(class_id=1)]))["objects"][0]
    assert o["roi_labels"] == []


def test_roi_labels_malformed_debug_handled() -> None:
    # debug.analytics is not a dict; debug.analytics.roi_labels is not a list.
    bad1 = NeutralObjectMetadata(class_id=1, debug={"analytics": "not-a-dict"})
    bad2 = NeutralObjectMetadata(class_id=1, debug={"analytics": {"roi_labels": "x"}})
    bad3 = NeutralObjectMetadata(class_id=1, debug={"analytics": {"roi_labels": [1, "ok"]}})
    snap = _build(_frame(objects=[bad1, bad2, bad3]))
    assert snap["objects"][0]["roi_labels"] == []
    assert snap["objects"][1]["roi_labels"] == []
    assert snap["objects"][2]["roi_labels"] == ["ok"]  # non-str dropped


def test_objects_capped_at_max() -> None:
    objs = [NeutralObjectMetadata(class_id=i) for i in range(5)]
    snap = _build(_frame(objects=objs), max_objects=2)
    assert len(snap["objects"]) == 2
    assert snap["debug"]["objects_seen"] == 5
    assert snap["debug"]["objects_truncated"] is True


def test_no_truncation_under_cap() -> None:
    objs = [NeutralObjectMetadata(class_id=i) for i in range(3)]
    snap = _build(_frame(objects=objs), max_objects=50)
    assert len(snap["objects"]) == 3
    assert snap["debug"]["objects_truncated"] is False


def test_geometry_embedded() -> None:
    assert _build(_frame())["geometry"] == _GEOM


def test_source_block_fields() -> None:
    src = _build(_frame())["source"]
    assert src == {
        "runtime": "deepstream",
        "metadata_source": "pyds_pad_probe",
        "analytics_plugin": "nvdsanalytics",
        "tracker_algorithm": "iou",
    }


def test_frame_block() -> None:
    snap = _build(_frame(frame_num=42, width=640, height=480))
    assert snap["frame"] == {
        "width": 640,
        "height": 480,
        "frame_num": 42,
        "fps": None,
    }


def test_frame_fps_from_source_when_present() -> None:
    snap = _build(_frame(source={"fps": 29.97}))
    assert snap["frame"]["fps"] == 29.97


def test_observed_at_is_iso_string() -> None:
    snap = _build(_frame())
    assert isinstance(snap["observed_at"], str)
    assert snap["observed_at"] == _OBSERVED.isoformat()


def test_output_is_json_serializable() -> None:
    obj = NeutralObjectMetadata(
        class_id=2,
        label="person",
        bbox_pixels=BoundingBoxPixels(left=1, top=2, width=3, height=4),
        debug={"analytics": {"roi_labels": ["entry-roi"]}},
    )
    snap = _build(_frame(objects=[obj]))
    # Must not raise — no datetime / UUID / set left in the dict.
    json.dumps(snap)


# --- M8: additive applied_processing_config stamping -------------------

# A representative resolver ``status_payload()`` block (selected_classes +
# confidence applied → partially_applied, carrying the honest analytics-counts
# caveat warning). The builder is opaque to the contents — it must store the
# dict verbatim under ``source`` — so a realistic shape is used to prove that.
_APPLIED_PC: dict[str, Any] = {
    "source": "operator_saved",
    "revision": 7,
    "status": "partially_applied",
    "reason": None,
    "requested": {
        "model_key": "trafficcamnet_v1",
        "selected_classes": ["person"],
        "tracker": None,
        "confidence_threshold": 0.55,
        "processing_fps": None,
    },
    "applied": {
        "model_key": "trafficcamnet_v1",
        "selected_classes": ["person"],
        "tracker": "iou",
        "confidence_threshold": 0.55,
        "processing_fps": None,
    },
    "warnings": ["selected_classes_analytics_counts_not_filtered"],
}


def test_applied_processing_config_omitted_by_default() -> None:
    """Back-compat: with the param omitted (default None) the ``source`` block
    carries NO ``applied_processing_config`` key — pre-M8 builder output is
    byte-identical (this is the existing ``test_source_block_fields`` shape)."""
    src = _build(_frame())["source"]
    assert "applied_processing_config" not in src
    assert src == {
        "runtime": "deepstream",
        "metadata_source": "pyds_pad_probe",
        "analytics_plugin": "nvdsanalytics",
        "tracker_algorithm": "iou",
    }


def test_applied_processing_config_stamped_into_source_block_verbatim() -> None:
    """When supplied, the dict is placed verbatim under ``source`` as a sibling
    of the existing provenance keys — the builder does not reshape or recompute
    it (single source of truth = the resolver's status_payload())."""
    snap = build_perception_snapshot(
        metadata=_frame(),
        geometry=_GEOM,
        camera_id=uuid4(),
        stream_session_id=uuid4(),
        processor_id="proc-1",
        package_id="deepstream.analytics_probe.v0",
        runtime="deepstream",
        metadata_source="pyds_pad_probe",
        analytics_plugin="nvdsanalytics",
        tracker_algorithm="iou",
        observed_at=_OBSERVED,
        applied_processing_config=_APPLIED_PC,
    )
    src = snap["source"]
    assert src["applied_processing_config"] == _APPLIED_PC
    # Existing provenance keys are untouched alongside the additive one.
    assert src["runtime"] == "deepstream"
    assert src["metadata_source"] == "pyds_pad_probe"
    assert src["analytics_plugin"] == "nvdsanalytics"
    assert src["tracker_algorithm"] == "iou"
    # The caveat warning rides along inside the stamped block, not stripped.
    assert (
        "selected_classes_analytics_counts_not_filtered"
        in src["applied_processing_config"]["warnings"]
    )


def test_applied_processing_config_does_not_touch_objects_or_geometry() -> None:
    """The additive stamp is metadata only: objects / geometry / debug are
    identical whether or not the block is supplied."""
    obj = NeutralObjectMetadata(
        class_id=2,
        label="person",
        tracker_id="0",
        bbox_pixels=BoundingBoxPixels(left=1, top=2, width=3, height=4),
        debug={"analytics": {"roi_labels": ["entry-roi"]}},
    )
    without = _build(_frame(objects=[obj]))
    with_pc = build_perception_snapshot(
        metadata=_frame(objects=[obj]),
        geometry=_GEOM,
        camera_id=uuid4(),
        stream_session_id=uuid4(),
        processor_id="proc-1",
        package_id="deepstream.analytics_probe.v0",
        runtime="deepstream",
        metadata_source="pyds_pad_probe",
        analytics_plugin="nvdsanalytics",
        tracker_algorithm="iou",
        observed_at=_OBSERVED,
        applied_processing_config=_APPLIED_PC,
    )
    assert with_pc["objects"] == without["objects"]
    assert with_pc["geometry"] == without["geometry"]
    assert with_pc["debug"] == without["debug"]
    # The ONLY difference is the additive source key.
    assert set(with_pc["source"]) - set(without["source"]) == {"applied_processing_config"}


def test_applied_processing_config_output_is_json_serializable() -> None:
    """The stamped block must not break the snapshot's JSON-serializability
    contract."""
    snap = build_perception_snapshot(
        metadata=_frame(),
        geometry=_GEOM,
        camera_id=uuid4(),
        stream_session_id=uuid4(),
        processor_id="proc-1",
        package_id="deepstream.analytics_probe.v0",
        runtime="deepstream",
        metadata_source="pyds_pad_probe",
        analytics_plugin="nvdsanalytics",
        tracker_algorithm="iou",
        observed_at=_OBSERVED,
        applied_processing_config=_APPLIED_PC,
    )
    json.dumps(snap)
