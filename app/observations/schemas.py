"""Pydantic v2 models — the single point that pins the M8.0 wire contract.

Mirrors `upstream-api-server/src/upstream_api/schemas/processor.py` exactly
(verified at commit 7a6057a). ``extra="ignore"`` so a forward-compatible
server addition does not break this client; required field names/types are
intentionally strict.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

ProcessorStatusValue = Literal[
    "starting",
    "attached",
    "processing",
    "lagging",
    "failed",
    "stopped",
]

ObservationType = Literal[
    "processor.mock_frame_seen",
    "processor.mock_heartbeat",
    "processor.mock_detection",
    # P4-B: first real DeepStream-backed observation. Lock-step with
    # ``upstream-api-server``'s ``ProcessorObservationType`` (any drift
    # 422s the write at the server's pydantic layer).
    "processor.line_crossing",
    # P4-C: ROI occupancy snapshot (on-change OR interval, never per-frame).
    "processor.roi_occupancy",
    # P4-D: per-track zone transitions. Stable outside-to-inside emits
    # zone_entered; stable inside-to-outside emits zone_exited. Both
    # gated by ``UPSTREAM_DEEPSTREAM_ZONE_MEMBERSHIP_STABILITY_FRAMES``
    # (default 5) so jitter doesn't flood; object disappearance does
    # NOT emit zone_exited.
    "processor.zone_entered",
    "processor.zone_exited",
]

OBSERVATION_TYPES: tuple[ObservationType, ...] = (
    "processor.mock_frame_seen",
    "processor.mock_heartbeat",
    "processor.mock_detection",
    "processor.line_crossing",
    "processor.roi_occupancy",
    "processor.zone_entered",
    "processor.zone_exited",
)


class ApprovedConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")
    codec: str
    width: int
    height: int
    fps: int
    bitrate_kbps: int


class ValidationSummary(BaseModel):
    model_config = ConfigDict(extra="ignore")
    passed: bool
    observed: dict[str, Any]
    expected: dict[str, Any]
    mismatch_fields: list[str]
    created_at: datetime


class EligibleSession(BaseModel):
    """One item of ``GET /eligible-sessions`` → ``{"sessions":[...]}``."""

    model_config = ConfigDict(extra="ignore")
    session_id: UUID
    stream_key: str
    camera_id: UUID
    account_id: UUID
    internal_rtsp_url: str
    package_id: str
    runtime: str
    approved: ApprovedConfig
    validation: ValidationSummary
    zones_lines: Any | None = None
    started_at: datetime | None = None
    processor_eligible: bool = True


class EligibleSessionsResponse(BaseModel):
    """The object envelope — NOT a bare array."""

    model_config = ConfigDict(extra="ignore")
    sessions: list[EligibleSession]


class ProcessorStatusPayload(BaseModel):
    """``POST /status`` request body.

    Deliberately has no ``account_id``/``camera_id`` — the server derives
    tenant from the session and a client must not assert it.
    """

    model_config = ConfigDict(extra="forbid")
    processor_id: str = Field(min_length=1, max_length=128)
    stream_session_id: UUID
    runtime: str = Field(min_length=1, max_length=64)
    package_id: str = Field(min_length=1, max_length=128)
    status: ProcessorStatusValue
    fps_input: float | None = Field(default=None, ge=0, le=1000)
    fps_processed: float | None = Field(default=None, ge=0, le=1000)
    last_frame_at: datetime | None = None
    last_observation_at: datetime | None = None
    error_code: str | None = Field(default=None, max_length=64)
    error_message: str | None = Field(default=None, max_length=512)
    payload: dict[str, Any] = Field(default_factory=dict)


class ProcessorStatusResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: UUID
    account_id: UUID
    camera_id: UUID
    stream_session_id: UUID
    processor_id: str
    runtime: str
    package_id: str
    status: ProcessorStatusValue
    created_at: datetime
    updated_at: datetime


class ObservationPayload(BaseModel):
    """``POST /observations`` request body."""

    model_config = ConfigDict(extra="forbid")
    stream_session_id: UUID
    type: ObservationType
    payload: dict[str, Any] = Field(default_factory=dict)
    observed_at: datetime
    processor_id: str | None = Field(default=None, max_length=128)
    package_id: str | None = Field(default=None, max_length=128)


class ObservationResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: UUID
    count: int
