"""P6-C — resolver behavior: envelope → ``ResolvedGeometry``.

The resolver is the single seam where operator-saved geometry becomes the
runtime's pixel-coord ``AnalyticsGeometry``. These tests pin three buckets:

- ``source="default"``: null envelope → fallback is returned untouched.
- ``source="operator_saved"``: valid envelope → normalized→pixel conversion,
  ID/label preservation, ``enabled=False`` filtering.
- ``source="fallback_default_on_error"``: invalid envelope or post-filter-
  empty → fallback returned with ``error_reason`` set.
"""

from __future__ import annotations

import pytest
from app.geometry.resolver import resolve_session_geometry
from app.runtimes.deepstream_runtime.inference import (
    AnalyticsGeometry,
    compute_analytics_geometry,
)


def _fallback() -> AnalyticsGeometry:
    return compute_analytics_geometry(frame_width=1280, frame_height=720)


def _valid_envelope() -> dict:
    return {
        "revision": 7,
        "config": {
            "version": 1,
            "frame": {"width": 1280, "height": 720, "coordinate_space": "normalized"},
            "rois": [
                {
                    "id": "lobby",
                    "label": "Lobby",
                    "type": "polygon",
                    "points": [
                        {"x": 0.1, "y": 0.1},
                        {"x": 0.9, "y": 0.1},
                        {"x": 0.9, "y": 0.9},
                        {"x": 0.1, "y": 0.9},
                    ],
                    "enabled": True,
                }
            ],
            "lines": [
                {
                    "id": "front-door",
                    "label": "Front Door",
                    "points": [{"x": 0.0, "y": 0.5}, {"x": 1.0, "y": 0.5}],
                    "direction": "in",
                    "enabled": True,
                }
            ],
        },
    }


# --- source="default" --------------------------------------------------


def test_null_envelope_returns_default_with_fallback_geometry() -> None:
    fb = _fallback()
    r = resolve_session_geometry(None, frame_width=1280, frame_height=720, fallback=fb)
    assert r.source == "default"
    assert r.revision is None
    assert r.error_reason is None
    assert r.geometry is fb
    assert r.line_count == len(fb.lines)
    assert r.roi_count == len(fb.rois)


# --- source="operator_saved" -------------------------------------------


def test_valid_envelope_resolves_to_operator_saved() -> None:
    r = resolve_session_geometry(
        _valid_envelope(), frame_width=1280, frame_height=720, fallback=_fallback()
    )
    assert r.source == "operator_saved"
    assert r.revision == 7
    assert r.error_reason is None
    assert r.line_count == 1
    assert r.roi_count == 1


def test_pixel_conversion_truncates_and_clamps() -> None:
    """``int()`` truncation matches ``compute_analytics_geometry``'s math;
    upper bound clamps to ``dim - 1`` so operator ``1.0`` doesn't land at
    ``dim`` (which is one past nvdsanalytics' valid pixel range)."""
    env = _valid_envelope()
    env["config"]["rois"] = []
    env["config"]["lines"] = [
        {
            "id": "edge-line",
            "label": "Edge",
            "points": [{"x": 0.0, "y": 0.0}, {"x": 1.0, "y": 1.0}],
            "direction": "any",
            "enabled": True,
        }
    ]
    r = resolve_session_geometry(env, frame_width=1280, frame_height=720, fallback=_fallback())
    assert r.source == "operator_saved"
    line = r.geometry.lines[0]
    assert (line.points[0].x, line.points[0].y) == (0, 0)
    # 1.0 -> clamp to frame-1, not frame.
    assert (line.points[1].x, line.points[1].y) == (1279, 719)


def test_midpoint_pixel_conversion() -> None:
    env = _valid_envelope()
    env["config"]["rois"] = []
    env["config"]["lines"] = [
        {
            "id": "mid",
            "label": None,
            "points": [{"x": 0.5, "y": 0.5}, {"x": 0.75, "y": 0.25}],
            "direction": "any",
            "enabled": True,
        }
    ]
    r = resolve_session_geometry(env, frame_width=1280, frame_height=720, fallback=_fallback())
    line = r.geometry.lines[0]
    # 0.5 * 1280 = 640; 0.75 * 1280 = 960; 0.5 * 720 = 360; 0.25 * 720 = 180.
    assert (line.points[0].x, line.points[0].y) == (640, 360)
    assert (line.points[1].x, line.points[1].y) == (960, 180)


def test_ids_and_labels_preserved() -> None:
    r = resolve_session_geometry(
        _valid_envelope(), frame_width=1280, frame_height=720, fallback=_fallback()
    )
    assert r.geometry.rois[0].id == "lobby"
    assert r.geometry.rois[0].label == "Lobby"
    assert r.geometry.lines[0].id == "front-door"
    assert r.geometry.lines[0].label == "Front Door"


def test_label_defaults_to_id_when_blank() -> None:
    env = _valid_envelope()
    env["config"]["rois"][0]["label"] = None
    env["config"]["lines"][0]["label"] = None
    r = resolve_session_geometry(env, frame_width=1280, frame_height=720, fallback=_fallback())
    assert r.geometry.rois[0].label == "lobby"
    assert r.geometry.lines[0].label == "front-door"


def test_enabled_false_geometry_is_filtered() -> None:
    """A disabled line/ROI should not enter the runtime — the dashboard
    owns disabled state; the runtime sees only what it should act on."""
    env = _valid_envelope()
    env["config"]["rois"].append(
        {
            "id": "muted-roi",
            "label": None,
            "type": "polygon",
            "points": [
                {"x": 0.0, "y": 0.0},
                {"x": 0.1, "y": 0.0},
                {"x": 0.05, "y": 0.1},
            ],
            "enabled": False,
        }
    )
    env["config"]["lines"].append(
        {
            "id": "muted-line",
            "label": None,
            "points": [{"x": 0.0, "y": 0.0}, {"x": 0.1, "y": 0.1}],
            "direction": "any",
            "enabled": False,
        }
    )
    r = resolve_session_geometry(env, frame_width=1280, frame_height=720, fallback=_fallback())
    assert r.source == "operator_saved"
    assert {roi.id for roi in r.geometry.rois} == {"lobby"}
    assert {line.id for line in r.geometry.lines} == {"front-door"}


# --- source="fallback_default_on_error" --------------------------------


def test_all_disabled_envelope_falls_back() -> None:
    env = _valid_envelope()
    env["config"]["rois"][0]["enabled"] = False
    env["config"]["lines"][0]["enabled"] = False
    r = resolve_session_geometry(env, frame_width=1280, frame_height=720, fallback=_fallback())
    assert r.source == "fallback_default_on_error"
    assert r.revision == 7  # surfaced so the operator can correlate the bad save
    assert r.error_reason and "no enabled" in r.error_reason


def test_malformed_envelope_falls_back() -> None:
    bad = {"revision": 1, "config": {"version": 1}}  # missing frame/rois/lines
    r = resolve_session_geometry(bad, frame_width=1280, frame_height=720, fallback=_fallback())
    assert r.source == "fallback_default_on_error"
    assert r.error_reason is not None
    # No raw envelope contents in the reason — only the pydantic loc/msg.
    assert "revision" not in r.error_reason.lower() or "field" in r.error_reason.lower()


def test_wrong_coordinate_space_falls_back() -> None:
    env = _valid_envelope()
    env["config"]["frame"]["coordinate_space"] = "absolute"
    r = resolve_session_geometry(env, frame_width=1280, frame_height=720, fallback=_fallback())
    assert r.source == "fallback_default_on_error"
    assert r.error_reason is not None


def test_bad_envelope_revision_falls_back() -> None:
    env = _valid_envelope()
    env["revision"] = "not-an-int"
    r = resolve_session_geometry(env, frame_width=1280, frame_height=720, fallback=_fallback())
    assert r.source == "fallback_default_on_error"


def test_extra_top_level_field_falls_back() -> None:
    """``extra="forbid"`` on the envelope — a foreign field is suspicious
    enough to degrade instead of silently accept (api-server drift signal)."""
    env = _valid_envelope()
    env["unknown_field"] = "drift"
    r = resolve_session_geometry(env, frame_width=1280, frame_height=720, fallback=_fallback())
    assert r.source == "fallback_default_on_error"


# --- status_payload() lineage shape -----------------------------------


def test_status_payload_shape() -> None:
    r = resolve_session_geometry(
        _valid_envelope(), frame_width=1280, frame_height=720, fallback=_fallback()
    )
    payload = r.status_payload()
    assert payload == {
        "source": "operator_saved",
        "revision": 7,
        "line_count": 1,
        "roi_count": 1,
    }


def test_status_payload_includes_error_reason_when_degraded() -> None:
    env = _valid_envelope()
    env["config"]["frame"]["coordinate_space"] = "absolute"
    r = resolve_session_geometry(env, frame_width=1280, frame_height=720, fallback=_fallback())
    payload = r.status_payload()
    assert payload["source"] == "fallback_default_on_error"
    assert "error_reason" in payload


# --- frame dimension validation ---------------------------------------


def test_rejects_non_positive_frame_dimensions() -> None:
    with pytest.raises(ValueError):
        resolve_session_geometry(None, frame_width=0, frame_height=720, fallback=_fallback())
    with pytest.raises(ValueError):
        resolve_session_geometry(None, frame_width=1280, frame_height=-1, fallback=_fallback())
