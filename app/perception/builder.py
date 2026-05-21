"""P5-C — pure latest-perception-snapshot builder.

Converts one ``NeutralFrameMetadata`` frame into the runtime-neutral snapshot
``dict`` the processor POSTs to the API. Pure: no I/O, no clock, no settings —
callers pass everything in. Fully defensive about the per-object metadata
(missing bbox / confidence / tracker_id / roi_labels never raise). The output
is JSON-serializable (UUIDs stringified, ``observed_at`` an ISO string, no
``set``/``datetime`` left in the dict).
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from app.observations.metadata_models import (
    BoundingBoxNormalized,
    BoundingBoxPixels,
    NeutralFrameMetadata,
    NeutralObjectMetadata,
)


def _bbox_pixels_json(bbox: BoundingBoxPixels | None) -> dict[str, float] | None:
    """Pixel bbox ``left/top/width/height`` -> the snapshot's ``x/y/w/h``."""
    if bbox is None:
        return None
    return {"x": bbox.left, "y": bbox.top, "w": bbox.width, "h": bbox.height}


def _bbox_normalized_json(bbox: BoundingBoxNormalized | None) -> dict[str, float] | None:
    if bbox is None:
        return None
    return {"x": bbox.x, "y": bbox.y, "w": bbox.w, "h": bbox.h}


def _roi_labels(obj: NeutralObjectMetadata) -> list[str]:
    """Per-object ROI labels live under ``debug["analytics"]["roi_labels"]``
    (P3.5). Defensive: any missing / wrong-typed level yields ``[]``."""
    analytics = obj.debug.get("analytics")
    if not isinstance(analytics, dict):
        return []
    raw = analytics.get("roi_labels")
    if not isinstance(raw, list):
        return []
    return [label for label in raw if isinstance(label, str)]


def _object_to_json(obj: NeutralObjectMetadata) -> dict[str, Any]:
    return {
        "track_id": obj.tracker_id,
        "class_id": obj.class_id,
        "label": obj.label,
        "confidence": obj.confidence,
        "bbox": _bbox_pixels_json(obj.bbox_pixels),
        "bbox_normalized": _bbox_normalized_json(obj.bbox_normalized),
        "roi_labels": _roi_labels(obj),
    }


def build_perception_snapshot(
    *,
    metadata: NeutralFrameMetadata,
    geometry: dict[str, Any],
    camera_id: UUID,
    stream_session_id: UUID,
    processor_id: str,
    package_id: str,
    runtime: str,
    metadata_source: str,
    analytics_plugin: str,
    tracker_algorithm: str,
    observed_at: datetime,
    max_objects: int = 50,
) -> dict[str, Any]:
    """Build the latest-perception-snapshot payload dict.

    ``geometry`` is the already-serialized ``AnalyticsGeometry.to_json()``
    output (``{"rois": [...], "lines": [...]}``) — passed pre-serialized so
    this builder stays free of any runtime-specific import. ``max_objects``
    bounds the ``objects`` list; ``debug.objects_seen`` keeps the true count.
    """
    all_objects = metadata.objects
    capped = all_objects[: max(max_objects, 0)]
    fps = metadata.source.get("fps")
    return {
        "camera_id": str(camera_id),
        "stream_session_id": str(stream_session_id),
        "processor_id": processor_id,
        "package_id": package_id,
        "observed_at": observed_at.isoformat(),
        "frame": {
            "width": metadata.frame.width,
            "height": metadata.frame.height,
            "frame_num": metadata.frame.frame_num,
            "fps": fps if isinstance(fps, (int, float)) else None,
        },
        "objects": [_object_to_json(o) for o in capped],
        "geometry": geometry,
        "source": {
            "runtime": runtime,
            "metadata_source": metadata_source,
            "analytics_plugin": analytics_plugin,
            "tracker_algorithm": tracker_algorithm,
        },
        "debug": {
            "objects_seen": len(all_objects),
            "objects_truncated": len(all_objects) > len(capped),
            "objects_with_tracker_id": sum(1 for o in all_objects if o.tracker_id is not None),
            "objects_with_roi_labels": sum(1 for o in all_objects if _roi_labels(o)),
        },
    }
