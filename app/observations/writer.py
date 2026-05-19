"""Single chokepoint for emitting observations.

Workers never call the API client for observations directly — they go through
``ObservationWriter.emit`` so the allowed-type guard and payload shape live in
one place. Error classification (404 quiet / 422 loud / transient retry) is
the worker's job; the writer just builds + posts and raises typed errors.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from app.api_client.base import ProcessorApi
from app.clock import ensure_aware, now_utc
from app.errors import ProcessorError
from app.observations.schemas import (
    OBSERVATION_TYPES,
    ObservationPayload,
    ObservationResponse,
    ObservationType,
)


class ObservationWriter:
    def __init__(self, api_client: ProcessorApi, processor_id: str) -> None:
        self._api = api_client
        self._processor_id = processor_id

    async def emit(
        self,
        *,
        stream_session_id: UUID,
        obs_type: ObservationType,
        payload: dict[str, Any],
        package_id: str | None = None,
        observed_at: datetime | None = None,
    ) -> ObservationResponse:
        # Defensive even though `obs_type` is a Literal: a dynamic caller must
        # not slip a non-mock type past M8.1.
        if obs_type not in OBSERVATION_TYPES:
            raise ProcessorError(
                f"observation type {obs_type!r} not allowed in M8.1; "
                f"allowed: {list(OBSERVATION_TYPES)}"
            )
        body = ObservationPayload(
            stream_session_id=stream_session_id,
            type=obs_type,
            payload=payload,
            observed_at=ensure_aware(observed_at) if observed_at else now_utc(),
            processor_id=self._processor_id,
            package_id=package_id,
        )
        return await self._api.post_observation(body)
