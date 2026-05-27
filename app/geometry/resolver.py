"""Resolve the eligible-session ``geometry_config`` envelope into the
runtime's ``AnalyticsGeometry`` (pixel coords) plus a lineage wrapper.

Three resolution outcomes:

- ``operator_saved`` — envelope parsed, has at least one enabled ROI or
  line. Pixel coordinates derive from the session's ``approved`` frame
  dimensions (truncate + clamp; matches ``compute_analytics_geometry``).
- ``default`` — envelope is ``None`` (camera has no dashboard-authored
  geometry). The probe default is used.
- ``fallback_default_on_error`` — envelope is present but doesn't parse,
  has wrong coordinate space, or after filtering ``enabled=False`` has zero
  lines AND zero ROIs. The probe default is used, ``error_reason`` is set,
  and the worker propagates ``error_code="geometry_invalid"`` on status so
  operators can see something is wrong without the camera going dark.

Pure / side-effect free: no I/O, no logging. Logging is the caller's job.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from pydantic import ValidationError

from app.geometry.session_geometry import (
    SessionGeometryEnvelope,
    SessionGeometryLine,
    SessionGeometryRoi,
)
from app.runtimes.deepstream_runtime.inference import (
    AnalyticsGeometry,
    GeometryPoint,
    LineGeometry,
    RoiGeometry,
)

ResolvedGeometrySource = Literal["operator_saved", "default", "fallback_default_on_error"]


@dataclass(frozen=True)
class ResolvedGeometry:
    """The geometry the runtime will actually apply, with lineage."""

    geometry: AnalyticsGeometry
    source: ResolvedGeometrySource
    revision: int | None
    line_count: int
    roi_count: int
    # Populated only when ``source == "fallback_default_on_error"``. Plain
    # English, no PII / no raw envelope contents — safe to log and to surface
    # on processor status payloads.
    error_reason: str | None = None

    def status_payload(self) -> dict[str, Any]:
        """Shape injected into ``ProcessorStatusPayload.payload`` for lineage."""
        block: dict[str, Any] = {
            "source": self.source,
            "revision": self.revision,
            "line_count": self.line_count,
            "roi_count": self.roi_count,
        }
        if self.error_reason is not None:
            block["error_reason"] = self.error_reason
        return block


def _to_pixel(coord: float, dim: int) -> int:
    """Normalized [0,1] -> pixel [0, dim-1]. Truncates (matches the existing
    probe-default math in ``compute_analytics_geometry``) and clamps the
    upper bound — operator ``1.0`` would otherwise land at ``dim`` which is
    one past the valid pixel range for nvdsanalytics' ``config-width`` /
    ``config-height``.
    """
    px = int(coord * dim)
    if px < 0:
        return 0
    if px >= dim:
        return dim - 1
    return px


def _convert_roi(roi: SessionGeometryRoi, *, frame_width: int, frame_height: int) -> RoiGeometry:
    return RoiGeometry(
        id=roi.id,
        # Label defaults to the ID when the operator left it blank — gives
        # downstream observation rows a human-readable name without
        # diverging from the api-server's "label is optional" contract.
        label=roi.label or roi.id,
        points=[
            GeometryPoint(_to_pixel(p.x, frame_width), _to_pixel(p.y, frame_height))
            for p in roi.points
        ],
    )


def _convert_line(
    line: SessionGeometryLine, *, frame_width: int, frame_height: int
) -> LineGeometry:
    return LineGeometry(
        id=line.id,
        label=line.label or line.id,
        points=[
            GeometryPoint(_to_pixel(p.x, frame_width), _to_pixel(p.y, frame_height))
            for p in line.points
        ],
    )


def resolve_session_geometry(
    envelope: dict[str, Any] | None,
    *,
    frame_width: int,
    frame_height: int,
    fallback: AnalyticsGeometry,
) -> ResolvedGeometry:
    """Resolve the wire envelope into a ``ResolvedGeometry``. See module docstring."""
    if frame_width <= 0 or frame_height <= 0:
        raise ValueError(
            f"frame_width and frame_height must be > 0; got {frame_width}x{frame_height}"
        )
    if envelope is None:
        return ResolvedGeometry(
            geometry=fallback,
            source="default",
            revision=None,
            line_count=len(fallback.lines),
            roi_count=len(fallback.rois),
        )
    try:
        parsed = SessionGeometryEnvelope.model_validate(envelope)
    except ValidationError as exc:
        # Surface a short, secret-free hint without leaking the raw envelope.
        errors = exc.errors()
        if errors:
            first = errors[0]
            loc = ".".join(str(p) for p in first.get("loc", ()))
            msg = first.get("msg", "validation failed")
        else:
            loc = ""
            msg = "validation failed"
        reason = (
            f"invalid geometry envelope: {loc}: {msg}"
            if loc
            else f"invalid geometry envelope: {msg}"
        )
        return ResolvedGeometry(
            geometry=fallback,
            source="fallback_default_on_error",
            revision=None,
            line_count=len(fallback.lines),
            roi_count=len(fallback.rois),
            error_reason=reason[:200],
        )

    enabled_rois = [r for r in parsed.config.rois if r.enabled]
    enabled_lines = [line for line in parsed.config.lines if line.enabled]

    if not enabled_rois and not enabled_lines:
        return ResolvedGeometry(
            geometry=fallback,
            source="fallback_default_on_error",
            revision=parsed.revision,
            line_count=len(fallback.lines),
            roi_count=len(fallback.rois),
            error_reason="geometry envelope has no enabled rois or lines",
        )

    rois = [
        _convert_roi(r, frame_width=frame_width, frame_height=frame_height) for r in enabled_rois
    ]
    lines = [
        _convert_line(line, frame_width=frame_width, frame_height=frame_height)
        for line in enabled_lines
    ]
    return ResolvedGeometry(
        geometry=AnalyticsGeometry(rois=rois, lines=lines),
        source="operator_saved",
        revision=parsed.revision,
        line_count=len(lines),
        roi_count=len(rois),
    )
