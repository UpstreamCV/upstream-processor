"""Deterministic synthetic observation payloads.

Pure functions, no I/O, no clock — the worker stamps ``observed_at``. Kept
deterministic (sequence-driven) so tests can assert exact payloads.
"""

from __future__ import annotations

from typing import Any


def frame_seen_payload(processor_id: str, frame_index: int) -> dict[str, Any]:
    return {
        "frame_index": frame_index,
        "source": {"runtime": "mock", "processor_id": processor_id},
    }


def heartbeat_payload(processor_id: str, tick: int) -> dict[str, Any]:
    return {
        "tick": tick,
        "healthy": True,
        "source": {"runtime": "mock", "processor_id": processor_id},
    }


def detection_payload(processor_id: str, seq: int) -> dict[str, Any]:
    return {
        "class": "person",
        "confidence": 0.99,
        "track_id": f"mock_trk_{seq}",
        "source": {"runtime": "mock", "processor_id": processor_id},
    }
