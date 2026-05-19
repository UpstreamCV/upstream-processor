"""Time source. One place that guarantees tz-aware UTC datetimes (the M8.0
contract uses ISO-8601 with timezone). Not in the brief's file list but kept
tiny and shared so ``observed_at``/``last_*_at`` are never naive."""

from __future__ import annotations

from datetime import UTC, datetime


def now_utc() -> datetime:
    return datetime.now(tz=UTC)


def ensure_aware(value: datetime) -> datetime:
    """Treat a naive datetime as UTC; pass an aware one through."""
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)
