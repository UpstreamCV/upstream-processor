"""P2 — tests for neutral intermediate metadata models.

GPU/Docker/pyds-free. The fixture sample is the EXACT shape the manual
P2 pyds probe emitted on the live ``person`` detection run."""

from __future__ import annotations

import json

import pytest
from app.observations.metadata_models import (
    UNTRACKED_OBJECT_ID,
    BoundingBoxNormalized,
    BoundingBoxPixels,
    NeutralFrame,
    NeutralFrameMetadata,
    NeutralObjectMetadata,
    bbox_normalized_from_pixels,
    tracker_id_from_pyds,
)
from pydantic import ValidationError

# The shape the manual P2 pyds probe printed live (one frame, one object).
PROBE_SAMPLE_DICT = {
    "runtime": "deepstream",
    "metadata_source": "pyds_pad_probe",
    "frame": {"frame_num": 0, "width": 1280, "height": 720},
    "objects": [
        {
            "class_id": 2,
            "label": "person",
            "confidence": 0.5121,
            "bbox_pixels": {
                "left": 420.87,
                "top": 91.65,
                "width": 495.94,
                "height": 627.02,
            },
            "bbox_normalized": {
                "x": 0.3288,
                "y": 0.1273,
                "w": 0.3875,
                "h": 0.8709,
            },
            "tracker_id": None,
        }
    ],
}


def test_parses_real_probe_sample() -> None:
    m = NeutralFrameMetadata.model_validate(PROBE_SAMPLE_DICT)
    assert m.runtime == "deepstream"
    assert m.metadata_source == "pyds_pad_probe"
    assert m.frame.frame_num == 0
    assert m.frame.width == 1280 and m.frame.height == 720
    assert len(m.objects) == 1
    obj = m.objects[0]
    assert obj.class_id == 2
    assert obj.label == "person"
    assert obj.confidence == pytest.approx(0.5121)
    assert obj.tracker_id is None
    assert obj.bbox_pixels and obj.bbox_pixels.width == pytest.approx(495.94)
    assert obj.bbox_normalized and obj.bbox_normalized.x == pytest.approx(0.3288)


def test_json_roundtrips() -> None:
    m = NeutralFrameMetadata.model_validate(PROBE_SAMPLE_DICT)
    same = NeutralFrameMetadata.model_validate(json.loads(m.model_dump_json()))
    assert same == m


def test_defaults_are_neutral() -> None:
    m = NeutralFrameMetadata(
        runtime="mock",
        metadata_source="fake",
        frame=NeutralFrame(frame_num=0, width=1, height=1),
    )
    assert m.objects == [] and m.source == {} and m.debug == {}
    assert m.stream_session_id is None


def test_extra_fields_rejected_at_every_level() -> None:
    # Top-level extra.
    bad_top = {**PROBE_SAMPLE_DICT, "rogue": 1}
    with pytest.raises(ValidationError):
        NeutralFrameMetadata.model_validate(bad_top)
    # Per-object extra.
    bad_obj = dict(PROBE_SAMPLE_DICT)
    bad_obj["objects"] = [{**PROBE_SAMPLE_DICT["objects"][0], "x_extra": 1}]
    with pytest.raises(ValidationError):
        NeutralFrameMetadata.model_validate(bad_obj)


def test_blank_runtime_or_source_rejected() -> None:
    base = dict(PROBE_SAMPLE_DICT)
    with pytest.raises(ValidationError):
        NeutralFrameMetadata.model_validate({**base, "runtime": ""})
    with pytest.raises(ValidationError):
        NeutralFrameMetadata.model_validate({**base, "metadata_source": ""})


def test_negative_frame_num_rejected() -> None:
    with pytest.raises(ValidationError):
        NeutralFrame(frame_num=-1, width=1, height=1)


def test_bbox_normalized_from_pixels_matches_probe_rounding() -> None:
    n = bbox_normalized_from_pixels(420.87, 91.65, 495.94, 627.02, 1280, 720)
    assert n == BoundingBoxNormalized(x=0.3288, y=0.1273, w=0.3875, h=0.8709)


def test_bbox_normalized_from_pixels_zero_dim_returns_none() -> None:
    assert bbox_normalized_from_pixels(1, 1, 1, 1, 0, 720) is None
    assert bbox_normalized_from_pixels(1, 1, 1, 1, 1280, 0) is None


def test_tracker_id_from_pyds_untracked_to_none() -> None:
    assert tracker_id_from_pyds(UNTRACKED_OBJECT_ID) is None
    assert tracker_id_from_pyds(0) == "0"
    assert tracker_id_from_pyds(2**63) == str(2**63)


def test_untracked_constant_value() -> None:
    assert UNTRACKED_OBJECT_ID == 0xFFFFFFFFFFFFFFFF


def test_pixel_bbox_is_a_pure_value_object() -> None:
    a = BoundingBoxPixels(left=1, top=2, width=3, height=4)
    b = BoundingBoxPixels(left=1, top=2, width=3, height=4)
    assert a == b


def test_neutral_object_metadata_minimum() -> None:
    """An object with only ``class_id`` must be valid (other producers may
    not have all of label/confidence/bbox/tracker_id)."""
    o = NeutralObjectMetadata(class_id=7)
    assert o.label is None and o.confidence is None
    assert o.bbox_pixels is None and o.bbox_normalized is None
    assert o.tracker_id is None
