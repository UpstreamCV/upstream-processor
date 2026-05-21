"""P5-C perception-snapshot wire models.

Mirror ``upstream-api-server``'s perception schemas: the ``POST
/v1/internal/processor/perception-snapshots`` request body and the
``{"snapshot": {...}}`` response envelope. ``extra="ignore"`` on responses so
a forward-compatible server addition does not break this client.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class PerceptionSnapshotPayload(BaseModel):
    """``POST /v1/internal/processor/perception-snapshots`` request body.

    No ``account_id`` / ``camera_id`` — the server derives tenant from the
    session and a client must not assert it. ``snapshot`` is the
    runtime-neutral dict from ``app.perception.builder.build_perception_snapshot``.
    """

    model_config = ConfigDict(extra="forbid")
    stream_session_id: UUID
    processor_id: str | None = Field(default=None, max_length=128)
    package_id: str | None = Field(default=None, max_length=128)
    observed_at: datetime
    snapshot: dict[str, Any] = Field(default_factory=dict)


class PerceptionSnapshotRecord(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: UUID
    camera_id: UUID
    stream_session_id: UUID
    processor_id: str | None = None
    package_id: str | None = None
    snapshot: dict[str, Any]
    observed_at: datetime
    updated_at: datetime


class PerceptionSnapshotResponse(BaseModel):
    """The ``{"snapshot": {...}}`` envelope the server returns."""

    model_config = ConfigDict(extra="ignore")
    snapshot: PerceptionSnapshotRecord
