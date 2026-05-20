"""P2 — bounded thread→asyncio bridge queue for neutral metadata.

Ferries ``NeutralFrameMetadata`` from a **non-async producer** (e.g. a
future DeepStream pyds pad probe running in the GLib mainloop thread) to
an **asyncio consumer** (the supervisor-managed worker), without blocking
either side. The producer never waits indefinitely: when the queue is full
it either rejects the newest item or evicts the oldest, depending on the
configured drop policy. Counters are exposed via ``stats()`` for
observability.

Why not a plain ``asyncio.Queue``? It is not safe to call from a
non-asyncio thread without ``loop.call_soon_threadsafe`` plumbing, and the
producer here will live on a GStreamer thread. ``queue.Queue`` is
thread-safe; the async ``get`` is implemented via ``asyncio.to_thread`` so
the event loop stays responsive.
"""

from __future__ import annotations

import asyncio
import enum
import queue
import threading
from dataclasses import dataclass

from app.observations.metadata_models import NeutralFrameMetadata


class DropPolicy(enum.StrEnum):
    """How the queue handles a producer ``put`` when it is already full."""

    DROP_NEWEST = "drop_newest"
    DROP_OLDEST = "drop_oldest"


@dataclass(frozen=True)
class BridgeQueueStats:
    accepted: int
    dropped: int
    dropped_newest: int
    dropped_oldest: int
    consumed: int
    current_size: int
    closed: bool


# Sentinel pushed on close() so a consumer blocked in ``queue.Queue.get()``
# wakes up promptly. ``object()`` is hash-stable and is identity-compared.
_CLOSED = object()


class BoundedBridgeQueue:
    """Thread-safe bounded queue with explicit drop policy.

    Producer side (``put_from_thread``): non-blocking, callable from any
    thread, returns ``True`` if accepted or ``False`` if dropped/closed.

    Consumer side (``get``): awaitable; returns the next
    ``NeutralFrameMetadata`` or ``None`` when the queue is closed and
    drained. Designed for **a single async consumer** (the supervisor
    worker); multi-consumer fan-out is not in scope.

    ``close()`` is idempotent and wakes any waiting consumer immediately.
    """

    def __init__(
        self,
        maxsize: int,
        drop_policy: DropPolicy = DropPolicy.DROP_NEWEST,
    ) -> None:
        if maxsize <= 0:
            raise ValueError("maxsize must be > 0")
        self._maxsize = int(maxsize)
        self._drop_policy = drop_policy
        # NOTE: we leave the inner queue.Queue's own maxsize unbounded and
        # gate ``maxsize`` ourselves under ``_lock``. queue.Queue's maxsize
        # would *block* put() until space frees, which is exactly what we
        # need to avoid on the producer thread.
        self._q: queue.Queue[object] = queue.Queue()
        self._lock = threading.Lock()
        self._accepted = 0
        self._dropped_newest = 0
        self._dropped_oldest = 0
        self._consumed = 0
        self._closed = False

    @property
    def maxsize(self) -> int:
        return self._maxsize

    @property
    def drop_policy(self) -> DropPolicy:
        return self._drop_policy

    # --- producer side (thread-safe, non-blocking) ----------------------

    def put_from_thread(self, item: NeutralFrameMetadata) -> bool:
        """Best-effort enqueue. Returns ``True`` if the item entered the
        queue (possibly displacing an older item under ``DROP_OLDEST``),
        ``False`` if it was rejected (``DROP_NEWEST`` and full, or
        closed). Never blocks the caller indefinitely."""
        with self._lock:
            if self._closed:
                return False
            # qsize() can lag in general; safe here because we hold the
            # lock and we are the only mover.
            if self._q.qsize() >= self._maxsize:
                if self._drop_policy is DropPolicy.DROP_NEWEST:
                    self._dropped_newest += 1
                    return False
                # DROP_OLDEST: evict one item to make room.
                try:
                    self._q.get_nowait()
                    self._dropped_oldest += 1
                except queue.Empty:  # pragma: no cover  (defensive)
                    pass
            self._q.put(item)
            self._accepted += 1
            return True

    # --- consumer side (asyncio-friendly) -------------------------------

    async def get(self) -> NeutralFrameMetadata | None:
        """Async-await the next item, or ``None`` if the queue is closed
        and fully drained. Items queued **before** close are still
        delivered (FIFO) before the ``None`` terminator."""
        # Fast path: take any already-queued item without entering a worker
        # thread. This also lets the consumer drain post-close items.
        try:
            item = self._q.get_nowait()
        except queue.Empty:
            if self._closed:
                return None
            # Slow path: park in a worker thread so the event loop stays
            # responsive. ``close()`` pushes a sentinel to unblock us.
            item = await asyncio.to_thread(self._q.get)
        if item is _CLOSED:
            return None
        with self._lock:
            self._consumed += 1
        return item  # type: ignore[return-value]

    # --- shutdown -------------------------------------------------------

    def close(self) -> None:
        """Idempotently mark the queue closed and wake the consumer.
        Items already queued remain deliverable until drained."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
        # Push the close sentinel *outside* the lock to avoid holding it
        # across a (potentially blocking) put.
        self._q.put(_CLOSED)

    # --- introspection --------------------------------------------------

    def stats(self) -> BridgeQueueStats:
        with self._lock:
            dropped = self._dropped_newest + self._dropped_oldest
            # The close sentinel inflates qsize by 1 once close() ran;
            # subtract it so ``current_size`` reflects pending real items.
            raw = self._q.qsize()
            size = max(0, raw - 1 if self._closed else raw)
            return BridgeQueueStats(
                accepted=self._accepted,
                dropped=dropped,
                dropped_newest=self._dropped_newest,
                dropped_oldest=self._dropped_oldest,
                consumed=self._consumed,
                current_size=size,
                closed=self._closed,
            )
