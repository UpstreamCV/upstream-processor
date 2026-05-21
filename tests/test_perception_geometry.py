"""P5-C — analytics geometry helper + INI-render regression.

``compute_analytics_geometry`` is the single source of the ROI/line numbers.
``build_analytics_config`` was refactored to consume it; the golden test below
pins that the rendered nvdsanalytics geometry lines stayed byte-identical.
"""

from __future__ import annotations

import pytest
from app.runtimes.deepstream_runtime.inference import (
    build_analytics_config,
    compute_analytics_geometry,
)


def test_default_roi_polygon_points() -> None:
    geom = compute_analytics_geometry(frame_width=1280, frame_height=720, roi_margin_pct=0.05)
    roi = geom.rois[0]
    assert roi.id == "entry-roi"
    assert roi.label == "entry-roi"
    assert [(p.x, p.y) for p in roi.points] == [
        (64, 36),
        (1216, 36),
        (1216, 684),
        (64, 684),
    ]


def test_zero_margin_roi_is_full_frame() -> None:
    geom = compute_analytics_geometry(frame_width=1280, frame_height=720, roi_margin_pct=0.0)
    assert [(p.x, p.y) for p in geom.rois[0].points] == [
        (0, 0),
        (1280, 0),
        (1280, 720),
        (0, 720),
    ]


def test_default_line_points() -> None:
    geom = compute_analytics_geometry(frame_width=1280, frame_height=720)
    line = geom.lines[0]
    assert line.id == "entry-line"
    assert [(p.x, p.y) for p in line.points] == [(320, 360), (960, 360)]


def test_to_json_shape() -> None:
    j = compute_analytics_geometry().to_json()
    assert set(j) == {"rois", "lines"}
    assert j["rois"][0]["points"][0] == {"x": 64, "y": 36}
    assert j["rois"][0]["id"] == "entry-roi"
    assert j["lines"][0]["id"] == "entry-line"
    assert j["lines"][0]["points"] == [{"x": 320, "y": 360}, {"x": 960, "y": 360}]


def test_rejects_bad_dimensions() -> None:
    with pytest.raises(ValueError, match="must be > 0"):
        compute_analytics_geometry(frame_width=0, frame_height=720)


def test_rejects_invalid_margin() -> None:
    with pytest.raises(ValueError, match="roi_margin_pct"):
        compute_analytics_geometry(roi_margin_pct=0.5)


def test_rejects_empty_labels() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        compute_analytics_geometry(roi_label="")


def test_build_analytics_config_geometry_render_unchanged() -> None:
    """Golden: the geometry lines the refactor must render byte-identically.

    ROI polygon and line-crossing coords (segment + direction-reference
    points) for the default 1280x720 / 5%-margin config.
    """
    ini = build_analytics_config(frame_width=1280, frame_height=720)
    assert "roi-entry-roi=64;36;1216;36;1216;684;64;684\n" in ini
    assert "line-crossing-entry-line=320;360;960;360;640;280;640;440\n" in ini
