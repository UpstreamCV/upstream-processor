"""P2 — neutral intermediate metadata models.

Runtime-neutral, pyds-free, observation-free. Mirrors the neutral shape the
manual P2 pyds probe emits (``scripts/deepstream_pyds_metadata_probe.py``)
as typed pydantic models so future pyds / mock / broker workers can produce
the **same shape** into the bounded bridge queue without leaking DeepStream
C structs or pyds objects into ``app/``.

These are **NOT** final UpStream observations — the
``processor.<type>``-typed ``ObservationPayload`` lives in
``app/observations/schemas.py`` and is what the eventual observation
adapter (P4) maps to. Keeping the two layers distinct is deliberate:
``MetadataSource`` -> ``BoundedBridgeQueue`` -> consumer reads
``NeutralFrameMetadata`` here, then a *later* adapter translates to one or
more ``ObservationPayload``s.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, NonNegativeInt

# pyds' UNTRACKED_OBJECT_ID (max u64). The producer (e.g. a pad probe) is
# responsible for mapping that value to ``tracker_id=None``; this constant
# is provided here so future pyds-in-``app/`` code doesn't redefine it.
# Duplicates ``scripts/_pyds_probe_helpers.py:UNTRACKED_OBJECT_ID`` to keep
# the runtime probe import-free of ``app/``.
UNTRACKED_OBJECT_ID = 0xFFFFFFFFFFFFFFFF


class BoundingBoxPixels(BaseModel):
    """Axis-aligned bbox in pixel coordinates (origin top-left)."""

    model_config = ConfigDict(extra="forbid")
    left: float
    top: float
    width: float
    height: float


class BoundingBoxNormalized(BaseModel):
    """Same bbox in 0..1 normalized coordinates. ``x``/``y`` are the
    top-left; ``w``/``h`` are the size — same convention as the manual P2
    probe output. Values are NOT clamped (a label that slightly overhangs
    the frame edge can legitimately produce values just outside 0..1)."""

    model_config = ConfigDict(extra="forbid")
    x: float
    y: float
    w: float
    h: float


class NeutralObjectMetadata(BaseModel):
    """One detected object's neutral metadata.

    ``tracker_id`` is ``str`` (not ``int``) because the upstream pyds value
    is uint64; some are larger than ``Number.MAX_SAFE_INTEGER`` for JS/JSON
    consumers, so the producer stringifies. ``UNTRACKED_OBJECT_ID`` MUST be
    mapped to ``None`` by the producer (no tracker attached)."""

    model_config = ConfigDict(extra="forbid")
    class_id: int
    label: str | None = None
    confidence: float | None = None
    bbox_pixels: BoundingBoxPixels | None = None
    bbox_normalized: BoundingBoxNormalized | None = None
    tracker_id: str | None = None


class NeutralFrame(BaseModel):
    model_config = ConfigDict(extra="forbid")
    frame_num: NonNegativeInt
    width: int
    height: int
    timestamp: datetime | None = None


class NeutralFrameMetadata(BaseModel):
    """One frame's worth of neutral, runtime-agnostic detection metadata.

    The producer fills ``runtime`` (e.g. ``"deepstream"``) and
    ``metadata_source`` (e.g. ``"pyds_pad_probe"``). DeepStream-specific
    extras MUST go under ``source``/``debug``, not as top-level fields,
    so the consumer side stays runtime-neutral."""

    model_config = ConfigDict(extra="forbid")
    runtime: str = Field(min_length=1)
    metadata_source: str = Field(min_length=1)
    stream_session_id: str | None = None
    frame: NeutralFrame
    objects: list[NeutralObjectMetadata] = Field(default_factory=list)
    source: dict[str, Any] = Field(default_factory=dict)
    debug: dict[str, Any] = Field(default_factory=dict)


def bbox_normalized_from_pixels(
    left: float,
    top: float,
    width: float,
    height: float,
    frame_w: int,
    frame_h: int,
) -> BoundingBoxNormalized | None:
    """Compute a 0..1 normalized bbox from pixel coords + frame dimensions.

    Returns ``None`` if either frame dimension is non-positive (frame unset).
    Rounds to 4 decimal places to keep JSON output compact and matches the
    rounding the manual P2 probe uses."""
    if frame_w <= 0 or frame_h <= 0:
        return None
    return BoundingBoxNormalized(
        x=round(left / frame_w, 4),
        y=round(top / frame_h, 4),
        w=round(width / frame_w, 4),
        h=round(height / frame_h, 4),
    )


def tracker_id_from_pyds(raw: int) -> str | None:
    """Map a raw pyds ``object_id`` (uint64) to the neutral
    ``tracker_id``: ``UNTRACKED_OBJECT_ID`` -> ``None``, else stringified.
    Pure helper; takes an ``int`` so it stays pyds-free."""
    if raw == UNTRACKED_OBJECT_ID:
        return None
    return str(raw)
