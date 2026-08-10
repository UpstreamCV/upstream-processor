"""Wire models for pull-based MediaMTX path provisioning.

Mirrors ``upstream_api.schemas.processor.PendingProvision`` /
``ProvisionAck*``. ``extra="ignore"`` throughout so the api-server can add
fields without breaking a box that has not been redeployed.
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, SecretStr


class PendingProvision(BaseModel):
    """One session whose local MediaMTX path this box must create.

    ``srt_publish_passphrase`` is a ``SecretStr`` so it cannot be leaked by an
    accidental ``repr()``/log of the model. It is the only secret carried on
    any ``/v1/internal/processor/*`` response.
    """

    model_config = ConfigDict(extra="ignore")

    session_id: UUID
    camera_id: UUID
    stream_key: str
    srt_publish_passphrase: SecretStr
    created_at: datetime


class PendingProvisionsResponse(BaseModel):
    """Object envelope — NOT a bare array (matches eligible-sessions)."""

    model_config = ConfigDict(extra="ignore")

    items: list[PendingProvision] = []


class ProvisionAckPayload(BaseModel):
    model_config = ConfigDict(extra="ignore")

    ok: bool
    error_code: str | None = None
    error_message: str | None = None


class ProvisionAckResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    applied: bool = False
