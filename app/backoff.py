"""Bounded, jittered, shutdown-interruptible backoff.

Async port of ``upstream-agent/upstream_agent/runtime/backoff.py``. Not in the
brief's file list but small and shared by the poller and every worker; each
holds its own instance so one flapping session can't widen another's delay.
"""

from __future__ import annotations

import asyncio
import random


class Backoff:
    def __init__(
        self,
        *,
        initial: float,
        factor: float,
        maximum: float,
        jitter: float,
    ) -> None:
        self._initial = initial
        self._factor = factor
        self._maximum = maximum
        self._jitter = jitter
        self._attempt = 0

    def reset(self) -> None:
        self._attempt = 0

    def next_delay(self) -> float:
        raw = self._initial * (self._factor**self._attempt)
        self._attempt += 1
        capped = min(raw, self._maximum)
        spread = capped * self._jitter
        return max(0.0, capped + random.uniform(-spread, spread))


async def wait_or_event(event: asyncio.Event, seconds: float) -> bool:
    """Sleep up to ``seconds``, waking immediately if ``event`` is set.

    Returns True if the event fired (i.e. caller should stop), else False.
    """
    if event.is_set():
        return True
    try:
        await asyncio.wait_for(event.wait(), timeout=seconds)
        return True
    except TimeoutError:
        return False
