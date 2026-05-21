"""Structural interface the supervisor/runtime depend on.

Decouples them from the concrete httpx client and lets tests inject an
in-memory fake. ``UpstreamApiClient`` satisfies this structurally.
"""

from __future__ import annotations

from typing import Protocol

from app.observations.schemas import (
    EligibleSessionsResponse,
    ObservationPayload,
    ObservationResponse,
    ProcessorStatusPayload,
    ProcessorStatusResponse,
)
from app.perception.schemas import (
    PerceptionSnapshotPayload,
    PerceptionSnapshotResponse,
)


class ProcessorApi(Protocol):
    async def get_eligible_sessions(self, limit: int = 50) -> EligibleSessionsResponse: ...

    async def post_processor_status(
        self, payload: ProcessorStatusPayload
    ) -> ProcessorStatusResponse: ...

    async def post_observation(self, observation: ObservationPayload) -> ObservationResponse: ...

    async def post_perception_snapshot(
        self, payload: PerceptionSnapshotPayload
    ) -> PerceptionSnapshotResponse: ...

    async def aclose(self) -> None: ...
