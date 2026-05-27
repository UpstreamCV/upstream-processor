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

    P6-C: the LC direction-reference math switched from hardcoded vertical
    offsets to per-line perpendicular; for a horizontal line this reduces
    to the prior math, which this test pins to byte-for-byte.
    """
    ini = build_analytics_config(frame_width=1280, frame_height=720)
    assert "roi-entry-roi=64;36;1216;36;1216;684;64;684\n" in ini
    assert "line-crossing-entry-line=320;360;960;360;640;280;640;440\n" in ini


def test_build_analytics_config_renders_multi_roi_and_multi_line() -> None:
    """P6-C: operator-saved geometry can have multiple lines and ROIs. Each
    must produce its own ``roi-<id>=`` / ``line-crossing-<id>=`` key under
    the single ``[roi-filtering-stream-0]`` / ``[line-crossing-stream-0]``
    sections nvdsanalytics expects."""
    from app.runtimes.deepstream_runtime.inference import (
        AnalyticsGeometry,
        GeometryPoint,
        LineGeometry,
        RoiGeometry,
    )

    geometry = AnalyticsGeometry(
        rois=[
            RoiGeometry(
                id="lobby",
                label="Lobby",
                points=[
                    GeometryPoint(10, 10),
                    GeometryPoint(100, 10),
                    GeometryPoint(100, 100),
                    GeometryPoint(10, 100),
                ],
            ),
            RoiGeometry(
                id="back-office",
                label="Back Office",
                points=[
                    GeometryPoint(200, 200),
                    GeometryPoint(300, 200),
                    GeometryPoint(300, 300),
                ],
            ),
        ],
        lines=[
            LineGeometry(
                id="entry",
                label="Entry",
                points=[GeometryPoint(320, 360), GeometryPoint(960, 360)],
            ),
            LineGeometry(
                id="exit",
                label="Exit",
                points=[GeometryPoint(640, 100), GeometryPoint(640, 600)],
            ),
        ],
    )
    ini = build_analytics_config(frame_width=1280, frame_height=720, geometry=geometry)
    # Section headers present exactly once.
    assert ini.count("[roi-filtering-stream-0]") == 1
    assert ini.count("[line-crossing-stream-0]") == 1
    # Each ROI / line emits its own key under the shared section.
    assert "roi-lobby=10;10;100;10;100;100;10;100\n" in ini
    assert "roi-back-office=200;200;300;200;300;300\n" in ini
    assert "line-crossing-entry=320;360;960;360;640;280;640;440\n" in ini
    # Vertical line: midpoint (640, 350); perpendicular is horizontal,
    # offset 80 px → ref0=(720, 350), ref1=(560, 350). Pinned to confirm
    # the per-line perpendicular math works for non-horizontal lines.
    assert "line-crossing-exit=640;100;640;600;720;350;560;350\n" in ini


def test_build_analytics_config_diagonal_line_perpendicular_refs() -> None:
    """For a 45° diagonal line, reference points sit on the perpendicular
    bisector (no longer stuck on the frame's vertical axis)."""
    from app.runtimes.deepstream_runtime.inference import (
        AnalyticsGeometry,
        GeometryPoint,
        LineGeometry,
    )

    geometry = AnalyticsGeometry(
        rois=[],
        lines=[
            LineGeometry(
                id="diag",
                label="Diagonal",
                points=[GeometryPoint(100, 100), GeometryPoint(300, 300)],
            )
        ],
    )
    ini = build_analytics_config(frame_width=1280, frame_height=720, geometry=geometry)
    # mid = (200, 200); dx=dy=200, length=sqrt(80000)≈282.84; perpendicular
    # rotated CCW = (-dy, dx) = (-200, 200), scaled to 80 → (-56.5685, 56.5685).
    # ref0 = int(200 - -56.57), int(200 - 56.57) = (256, 143).
    # ref1 = int(200 + -56.57), int(200 + 56.57) = (143, 256).
    assert "line-crossing-diag=100;100;300;300;256;143;143;256\n" in ini


def test_build_analytics_config_empty_geometry_yields_no_keys() -> None:
    """An empty supplied geometry (no ROIs, no lines) should still produce
    a valid INI skeleton — useful as a defensive default for tests."""
    from app.runtimes.deepstream_runtime.inference import AnalyticsGeometry

    geometry = AnalyticsGeometry(rois=[], lines=[])
    ini = build_analytics_config(frame_width=1280, frame_height=720, geometry=geometry)
    assert "[roi-filtering-stream-0]" in ini
    assert "[line-crossing-stream-0]" in ini
    assert "roi-" not in ini.replace("roi-filtering", "")
    assert "line-crossing-" not in ini.replace("line-crossing-stream", "")
