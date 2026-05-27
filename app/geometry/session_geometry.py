"""Pydantic models for the operator-saved geometry envelope.

These mirror ``upstream-api-server``'s ``schemas/geometry.py`` exactly — the
api-server is the source of truth for the wire contract, and the processor
re-validates here because the eligible-session payload carries
``geometry_config`` as an opaque dict that has only seen JSON-decode (no
schema validation) up to this point. Re-validating locally means a
malformed config from a future api drift surfaces as a clean fallback in
the processor instead of corrupting pixel-coord math downstream.

Drift policy: ``extra="forbid"`` here is deliberate. The api-server enforces
its schema on write, so an unexpected key in the wire envelope means
*either* an api-server bug *or* a malicious row that bypassed validation.
Either way, the resolver should degrade to probe defaults rather than
silently accept the foreign field.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

# Stable IDs are slug-shaped — matches the api-server's ``GeometrySlug``
# constraint. ROI/line IDs flow through to observation rows so operators can
# correlate dashboard geometry to emitted events; relaxing this pattern would
# permit IDs that downstream consumers (URLs, log keys) can't safely embed.
GeometrySlug = Annotated[
    str, StringConstraints(pattern=r"^[a-z0-9][a-z0-9-]{0,63}$", min_length=1, max_length=64)
]


class SessionGeometryPoint(BaseModel):
    """One normalized vertex. ``x``/``y`` ∈ [0.0, 1.0] (frame-relative)."""

    model_config = ConfigDict(extra="forbid")
    x: float = Field(ge=0.0, le=1.0)
    y: float = Field(ge=0.0, le=1.0)


class SessionGeometryFrame(BaseModel):
    """Reference frame metadata. ``coordinate_space`` must be ``"normalized"``;
    the dashboard authors only normalized geometry today, and the resolver
    rejects any other value rather than guessing a pixel convention."""

    model_config = ConfigDict(extra="forbid")
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    coordinate_space: Literal["normalized"]


class SessionGeometryRoi(BaseModel):
    """One polygon ROI authored in the dashboard."""

    model_config = ConfigDict(extra="forbid")
    id: GeometrySlug
    label: str | None = Field(default=None, max_length=80)
    type: Literal["polygon"]
    # Polygon needs at least 3 vertices; the dashboard caps authoring at 64.
    points: list[SessionGeometryPoint] = Field(min_length=3, max_length=64)
    enabled: bool


class SessionGeometryLine(BaseModel):
    """One counting line authored in the dashboard. Exactly two endpoints."""

    model_config = ConfigDict(extra="forbid")
    id: GeometrySlug
    label: str | None = Field(default=None, max_length=80)
    points: list[SessionGeometryPoint] = Field(min_length=2, max_length=2)
    # P6-C preserves ``direction`` in lineage only — nvdsanalytics is run
    # in ``mode=loose`` today, so direction is not enforced at the runtime
    # (follow-up). Validating the enum here still guards against drift.
    direction: Literal["any", "in", "out"]
    enabled: bool


class SessionGeometryConfig(BaseModel):
    """Top-level operator-saved geometry."""

    model_config = ConfigDict(extra="forbid")
    version: Literal[1]
    frame: SessionGeometryFrame
    rois: list[SessionGeometryRoi]
    lines: list[SessionGeometryLine]


class SessionGeometryEnvelope(BaseModel):
    """The eligible-session ``geometry_config`` field. ``revision`` is the
    api-server's monotonic save counter — surfaced on status and observation
    lineage so consumers can correlate emitted events to a specific save."""

    model_config = ConfigDict(extra="forbid")
    revision: int = Field(ge=0)
    config: SessionGeometryConfig
