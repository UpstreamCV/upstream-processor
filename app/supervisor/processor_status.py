"""Build ``ProcessorStatusPayload`` requests.

One place that (a) sources tenant-free identity from the session, (b) never
sets ``account_id``/``camera_id`` (the server derives them), and (c) keeps
every datetime tz-aware.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from app.clock import ensure_aware
from app.observations.schemas import (
    EligibleSession,
    ProcessorStatusPayload,
    ProcessorStatusValue,
)


def build_status_request(
    *,
    session: EligibleSession,
    processor_id: str,
    status: ProcessorStatusValue,
    runtime: str | None = None,
    package_id: str | None = None,
    payload: dict[str, Any] | None = None,
    fps_input: float | None = None,
    fps_processed: float | None = None,
    last_frame_at: datetime | None = None,
    last_observation_at: datetime | None = None,
    error_code: str | None = None,
    error_message: str | None = None,
) -> ProcessorStatusPayload:
    # `runtime`/`package_id` default to the session's values (M8.0 returns
    # runtime="mock"); the DeepStream worker overrides them with the resolved
    # package's runtime/id so status rows reflect what actually ran.
    return ProcessorStatusPayload(
        processor_id=processor_id,
        stream_session_id=session.session_id,
        runtime=runtime if runtime is not None else session.runtime,
        package_id=package_id if package_id is not None else session.package_id,
        status=status,
        fps_input=fps_input,
        fps_processed=fps_processed,
        last_frame_at=ensure_aware(last_frame_at) if last_frame_at else None,
        last_observation_at=(ensure_aware(last_observation_at) if last_observation_at else None),
        error_code=error_code,
        error_message=error_message,
        payload=payload or {},
    )
