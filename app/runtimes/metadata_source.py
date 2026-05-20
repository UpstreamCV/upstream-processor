"""P2 — ``MetadataSource`` Protocol + ``BridgeQueueMetadataSource``.

The metadata seam between a producer (e.g. a future DeepStream pyds pad
probe running in the GStreamer thread, or a mock / Triton / broker
producer) and an asyncio consumer. **Runtime-neutral**: nothing here
imports pyds, gi, or Gst — implementations must yield ``NeutralFrameMetadata``
only. The eventual observation adapter / ``ObservationWriter`` integration
is a later milestone (P4) and is deliberately *not* coupled to this seam.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol, runtime_checkable

from app.observations.bridge_queue import BoundedBridgeQueue
from app.observations.metadata_models import NeutralFrameMetadata


@runtime_checkable
class MetadataSource(Protocol):
    """Async-iterable source of neutral metadata frames.

    ``async for frame in source:`` yields one
    :class:`~app.observations.metadata_models.NeutralFrameMetadata` per
    frame the producer emitted, until the source is exhausted (the
    producer side has closed and the buffer is drained)."""

    def __aiter__(self) -> AsyncIterator[NeutralFrameMetadata]: ...


class BridgeQueueMetadataSource:
    """``MetadataSource`` over a :class:`BoundedBridgeQueue`.

    Producer side (thread): calls ``queue.put_from_thread(frame)`` on every
    metadata frame; that path is non-blocking with explicit drop policy.

    Consumer side (asyncio): ``async for frame in source:`` drains the
    queue in FIFO order and exits cleanly when ``queue.close()`` is called
    and the buffered frames have been delivered."""

    def __init__(self, queue: BoundedBridgeQueue) -> None:
        self._queue = queue

    @property
    def queue(self) -> BoundedBridgeQueue:
        return self._queue

    def __aiter__(self) -> BridgeQueueMetadataSource:
        return self

    async def __anext__(self) -> NeutralFrameMetadata:
        item = await self._queue.get()
        if item is None:
            raise StopAsyncIteration
        return item
