"""P2 — tests for ``MetadataSource`` Protocol + ``BridgeQueueMetadataSource``.

GPU/Docker/pyds-free."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import AsyncIterator

from app.observations.bridge_queue import BoundedBridgeQueue, DropPolicy
from app.observations.metadata_models import NeutralFrame, NeutralFrameMetadata
from app.runtimes.metadata_source import (
    BridgeQueueMetadataSource,
    MetadataSource,
)


def _frame(i: int) -> NeutralFrameMetadata:
    return NeutralFrameMetadata(
        runtime="mock",
        metadata_source="test",
        frame=NeutralFrame(frame_num=i, width=1, height=1),
    )


class _ListMetadataSource:
    """Trivial reference implementation for Protocol-conformance tests."""

    def __init__(self, items: list[NeutralFrameMetadata]) -> None:
        self._items = list(items)

    def __aiter__(self) -> AsyncIterator[NeutralFrameMetadata]:
        return self._iter()

    async def _iter(self) -> AsyncIterator[NeutralFrameMetadata]:
        for item in self._items:
            await asyncio.sleep(0)
            yield item


def test_protocol_runtime_checkable() -> None:
    assert isinstance(_ListMetadataSource([]), MetadataSource)
    assert isinstance(BridgeQueueMetadataSource(BoundedBridgeQueue(maxsize=1)), MetadataSource)
    # A non-implementer must NOT pass the check.
    assert not isinstance(object(), MetadataSource)


async def test_bridge_source_yields_then_stops_on_close() -> None:
    q = BoundedBridgeQueue(maxsize=4)
    src = BridgeQueueMetadataSource(q)
    for i in range(3):
        q.put_from_thread(_frame(i))
    q.close()

    seen: list[int] = []
    async for fm in src:
        seen.append(fm.frame.frame_num)
    assert seen == [0, 1, 2]


async def test_bridge_source_async_for_wakes_on_late_put() -> None:
    q = BoundedBridgeQueue(maxsize=4)
    src = BridgeQueueMetadataSource(q)

    async def drain() -> list[int]:
        out: list[int] = []
        async for fm in src:
            out.append(fm.frame.frame_num)
        return out

    task = asyncio.create_task(drain())
    await asyncio.sleep(0.02)
    q.put_from_thread(_frame(10))
    q.put_from_thread(_frame(20))
    await asyncio.sleep(0.02)
    q.close()
    out = await asyncio.wait_for(task, timeout=1.0)
    assert out == [10, 20]


async def test_bridge_source_drains_pre_close_then_exits() -> None:
    q = BoundedBridgeQueue(maxsize=4)
    src = BridgeQueueMetadataSource(q)

    seen: list[int] = []

    async def drain() -> None:
        async for fm in src:
            seen.append(fm.frame.frame_num)

    # Fill 2 items, then close; the source MUST still deliver them before
    # exiting the async for.
    q.put_from_thread(_frame(1))
    q.put_from_thread(_frame(2))
    q.close()
    await asyncio.wait_for(drain(), timeout=1.0)
    assert seen == [1, 2]


async def test_bridge_source_with_real_producer_thread() -> None:
    q = BoundedBridgeQueue(maxsize=8, drop_policy=DropPolicy.DROP_NEWEST)
    src = BridgeQueueMetadataSource(q)

    n_put = 60

    def producer() -> None:
        for i in range(n_put):
            q.put_from_thread(_frame(i))
        q.close()

    t = threading.Thread(target=producer)
    t.start()

    received: list[int] = []
    async for fm in src:
        received.append(fm.frame.frame_num)
    t.join(timeout=1.0)
    assert not t.is_alive()

    s = q.stats()
    # All received items come out in FIFO order…
    assert received == sorted(received)
    # …and the producer's accepted set equals what we received.
    assert s.consumed == len(received) == s.accepted
    # No accounting leak.
    assert s.accepted + s.dropped == n_put


def test_bridge_source_exposes_underlying_queue() -> None:
    q = BoundedBridgeQueue(maxsize=2)
    src = BridgeQueueMetadataSource(q)
    assert src.queue is q
