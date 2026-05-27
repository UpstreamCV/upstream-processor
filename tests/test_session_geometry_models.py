"""P6-C — pydantic validation of the eligible-session geometry envelope.

The processor re-validates the wire envelope locally so a malformed config
from a future api-server drift surfaces as a clean ``ValidationError``
(which the resolver maps to ``source="fallback_default_on_error"``) instead
of corrupting downstream pixel-coord math.
"""

from __future__ import annotations

import pytest
from app.geometry.session_geometry import (
    SessionGeometryConfig,
    SessionGeometryEnvelope,
    SessionGeometryFrame,
    SessionGeometryLine,
    SessionGeometryPoint,
    SessionGeometryRoi,
)
from pydantic import ValidationError


def _valid_point() -> dict:
    return {"x": 0.1, "y": 0.2}


def _valid_roi() -> dict:
    return {
        "id": "front-door",
        "label": "Front Door",
        "type": "polygon",
        "points": [
            {"x": 0.0, "y": 0.0},
            {"x": 1.0, "y": 0.0},
            {"x": 0.5, "y": 1.0},
        ],
        "enabled": True,
    }


def _valid_line() -> dict:
    return {
        "id": "entry",
        "label": "Entry",
        "points": [{"x": 0.2, "y": 0.5}, {"x": 0.8, "y": 0.5}],
        "direction": "any",
        "enabled": True,
    }


def _valid_envelope() -> dict:
    return {
        "revision": 3,
        "config": {
            "version": 1,
            "frame": {"width": 1280, "height": 720, "coordinate_space": "normalized"},
            "rois": [_valid_roi()],
            "lines": [_valid_line()],
        },
    }


def test_envelope_accepts_canonical_shape() -> None:
    env = SessionGeometryEnvelope.model_validate(_valid_envelope())
    assert env.revision == 3
    assert env.config.version == 1
    assert env.config.rois[0].id == "front-door"
    assert env.config.lines[0].direction == "any"


def test_point_rejects_out_of_unit_interval() -> None:
    for bad in ({"x": -0.01, "y": 0.5}, {"x": 0.5, "y": 1.01}):
        with pytest.raises(ValidationError):
            SessionGeometryPoint.model_validate(bad)


def test_roi_requires_three_or_more_points() -> None:
    roi = _valid_roi()
    roi["points"] = roi["points"][:2]
    with pytest.raises(ValidationError):
        SessionGeometryRoi.model_validate(roi)


def test_roi_caps_at_64_points() -> None:
    roi = _valid_roi()
    roi["points"] = [_valid_point() for _ in range(65)]
    with pytest.raises(ValidationError):
        SessionGeometryRoi.model_validate(roi)


def test_line_requires_exactly_two_points() -> None:
    line = _valid_line()
    line["points"] = line["points"] + [{"x": 0.5, "y": 0.5}]
    with pytest.raises(ValidationError):
        SessionGeometryLine.model_validate(line)
    line["points"] = line["points"][:1]
    with pytest.raises(ValidationError):
        SessionGeometryLine.model_validate(line)


def test_id_must_be_slug_shape() -> None:
    """IDs flow through to observation rows — must match the api-server's
    slug pattern so they're URL/log-safe."""
    roi = _valid_roi()
    for bad_id in ("", "Front Door", "front_door", "-front", "FrontDoor", "front!"):
        roi["id"] = bad_id
        with pytest.raises(ValidationError):
            SessionGeometryRoi.model_validate(roi)


def test_version_must_be_one() -> None:
    cfg = _valid_envelope()["config"]
    cfg["version"] = 2
    with pytest.raises(ValidationError):
        SessionGeometryConfig.model_validate(cfg)


def test_coordinate_space_must_be_normalized() -> None:
    frame = {"width": 1280, "height": 720, "coordinate_space": "absolute"}
    with pytest.raises(ValidationError):
        SessionGeometryFrame.model_validate(frame)


def test_line_direction_enum() -> None:
    line = _valid_line()
    line["direction"] = "diagonal"
    with pytest.raises(ValidationError):
        SessionGeometryLine.model_validate(line)


def test_roi_type_must_be_polygon() -> None:
    roi = _valid_roi()
    roi["type"] = "rectangle"
    with pytest.raises(ValidationError):
        SessionGeometryRoi.model_validate(roi)


def test_unknown_fields_rejected() -> None:
    """``extra="forbid"`` so a forward-compatible api-server field we don't
    yet understand causes a clean degrade instead of a silent ignore."""
    env = _valid_envelope()
    env["unexpected"] = "drift"
    with pytest.raises(ValidationError):
        SessionGeometryEnvelope.model_validate(env)


def test_revision_must_be_non_negative() -> None:
    env = _valid_envelope()
    env["revision"] = -1
    with pytest.raises(ValidationError):
        SessionGeometryEnvelope.model_validate(env)


def test_label_is_optional() -> None:
    """The api-server allows ``label: null`` (operator left it blank).
    The resolver substitutes the slug ID as label downstream."""
    roi = _valid_roi()
    roi["label"] = None
    parsed = SessionGeometryRoi.model_validate(roi)
    assert parsed.label is None
