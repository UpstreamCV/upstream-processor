"""In-memory fakes for supervisor/runtime tests (no network, no real timers)."""

from __future__ import annotations

from collections import deque
from uuid import uuid4

from app.observations.schemas import (
    EligibleSession,
    EligibleSessionsResponse,
    ObservationPayload,
    ObservationResponse,
    ProcessorStatusPayload,
    ProcessorStatusResponse,
)
from app.perception.schemas import (
    PerceptionSnapshotPayload,
    PerceptionSnapshotRecord,
    PerceptionSnapshotResponse,
)


class FakeApi:
    """Stand-in for ``UpstreamApiClient`` with recording + injectable faults."""

    def __init__(self) -> None:
        self.statuses: list[ProcessorStatusPayload] = []
        self.observations: list[ObservationPayload] = []
        self.perception_snapshots: list[PerceptionSnapshotPayload] = []
        # Queue of eligible-session batches; the last batch repeats forever.
        self._eligible: deque[list[EligibleSession]] = deque()
        self.eligible_error: Exception | None = None
        self.status_error: Exception | None = None
        self.observation_error: Exception | None = None
        self.perception_snapshot_error: Exception | None = None
        self.closed = False

    # configuration -------------------------------------------------------

    def queue_eligible(self, batch: list[EligibleSession]) -> None:
        self._eligible.append(batch)

    # client surface ------------------------------------------------------

    async def get_eligible_sessions(self, limit: int = 50) -> EligibleSessionsResponse:
        if self.eligible_error is not None:
            raise self.eligible_error
        if len(self._eligible) > 1:
            batch = self._eligible.popleft()
        elif self._eligible:
            batch = self._eligible[0]
        else:
            batch = []
        return EligibleSessionsResponse(sessions=batch)

    async def post_processor_status(
        self, payload: ProcessorStatusPayload
    ) -> ProcessorStatusResponse:
        self.statuses.append(payload)
        if self.status_error is not None:
            raise self.status_error
        from datetime import UTC, datetime

        now = datetime.now(tz=UTC)
        return ProcessorStatusResponse(
            id=uuid4(),
            account_id=uuid4(),
            camera_id=uuid4(),
            stream_session_id=payload.stream_session_id,
            processor_id=payload.processor_id,
            runtime=payload.runtime,
            package_id=payload.package_id,
            status=payload.status,
            created_at=now,
            updated_at=now,
        )

    async def post_observation(self, observation: ObservationPayload) -> ObservationResponse:
        self.observations.append(observation)
        if self.observation_error is not None:
            raise self.observation_error
        return ObservationResponse(id=uuid4(), count=1)

    async def post_perception_snapshot(
        self, payload: PerceptionSnapshotPayload
    ) -> PerceptionSnapshotResponse:
        self.perception_snapshots.append(payload)
        if self.perception_snapshot_error is not None:
            raise self.perception_snapshot_error
        from datetime import UTC, datetime

        now = datetime.now(tz=UTC)
        return PerceptionSnapshotResponse(
            snapshot=PerceptionSnapshotRecord(
                id=uuid4(),
                camera_id=uuid4(),
                stream_session_id=payload.stream_session_id,
                processor_id=payload.processor_id,
                package_id=payload.package_id,
                snapshot=payload.snapshot,
                observed_at=payload.observed_at,
                updated_at=now,
            )
        )

    async def aclose(self) -> None:
        self.closed = True

    # helpers -------------------------------------------------------------

    def status_sequence(self) -> list[str]:
        return [s.status for s in self.statuses]
