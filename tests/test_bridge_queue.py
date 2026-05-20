"""P2 — tests for the bounded thread→asyncio bridge queue.

GPU/Docker/pyds-free. Covers normal FIFO flow, both drop policies, close
ordering (pre-close items still deliverable, post-close ``put`` rejected,
async consumer waking on close), and a real ``threading.Thread`` producer
to confirm no deadlock and bounded memory."""

from __future__ import annotations

import asyncio
import threading
import time

import pytest
from app.observations.bridge_queue import (
    BoundedBridgeQueue,
    BridgeQueueStats,
    DropPolicy,
)
from app.observations.metadata_models import NeutralFrame, NeutralFrameMetadata


def _frame(i: int) -> NeutralFrameMetadata:
    return NeutralFrameMetadata(
        runtime="mock",
        metadata_source="test",
        frame=NeutralFrame(frame_num=i, width=10, height=10),
    )


# --- construction & guards ---------------------------------------------


def test_rejects_nonpositive_maxsize() -> None:
    with pytest.raises(ValueError):
        BoundedBridgeQueue(maxsize=0)
    with pytest.raises(ValueError):
        BoundedBridgeQueue(maxsize=-1)


def test_exposes_maxsize_and_policy() -> None:
    q = BoundedBridgeQueue(maxsize=3, drop_policy=DropPolicy.DROP_OLDEST)
    assert q.maxsize == 3
    assert q.drop_policy is DropPolicy.DROP_OLDEST


# --- normal flow --------------------------------------------------------


async def test_normal_flow_preserves_order_and_counters() -> None:
    q = BoundedBridgeQueue(maxsize=10)
    for i in range(5):
        assert q.put_from_thread(_frame(i)) is True

    received: list[int] = []
    for _ in range(5):
        item = await q.get()
        assert item is not None
        received.append(item.frame.frame_num)
    assert received == [0, 1, 2, 3, 4]

    s = q.stats()
    assert (s.accepted, s.consumed, s.current_size) == (5, 5, 0)
    assert s.dropped == 0 and s.dropped_newest == 0 and s.dropped_oldest == 0


# --- drop policies ------------------------------------------------------


async def test_drop_newest_rejects_overflow_and_keeps_oldest() -> None:
    q = BoundedBridgeQueue(maxsize=3, drop_policy=DropPolicy.DROP_NEWEST)
    accepted = [q.put_from_thread(_frame(i)) for i in range(5)]
    assert accepted == [True, True, True, False, False]

    seen = [(await q.get()).frame.frame_num for _ in range(3)]  # type: ignore[union-attr]
    assert seen == [0, 1, 2]  # the original three were preserved
    s = q.stats()
    assert s.accepted == 3 and s.dropped_newest == 2 and s.dropped_oldest == 0
    assert s.dropped == 2 and s.consumed == 3 and s.current_size == 0


async def test_drop_oldest_evicts_oldest_and_accepts_newest() -> None:
    q = BoundedBridgeQueue(maxsize=3, drop_policy=DropPolicy.DROP_OLDEST)
    accepted = [q.put_from_thread(_frame(i)) for i in range(5)]
    assert accepted == [True, True, True, True, True]  # all accepted

    seen = [(await q.get()).frame.frame_num for _ in range(3)]  # type: ignore[union-attr]
    assert seen == [2, 3, 4]  # newest three remained
    s = q.stats()
    assert s.accepted == 5 and s.dropped_oldest == 2 and s.dropped_newest == 0
    assert s.dropped == 2 and s.consumed == 3 and s.current_size == 0


# --- close behavior -----------------------------------------------------


async def test_pre_close_items_drain_then_none() -> None:
    q = BoundedBridgeQueue(maxsize=5)
    for i in range(3):
        q.put_from_thread(_frame(i))
    q.close()
    drained = []
    while True:
        item = await q.get()
        if item is None:
            break
        drained.append(item.frame.frame_num)
    assert drained == [0, 1, 2]
    assert q.stats().closed is True


async def test_put_after_close_returns_false() -> None:
    q = BoundedBridgeQueue(maxsize=2)
    q.close()
    assert q.put_from_thread(_frame(99)) is False
    # Closed and empty → consumer immediately sees None.
    assert (await q.get()) is None
    s = q.stats()
    assert s.closed and s.accepted == 0 and s.dropped == 0


def test_close_is_idempotent() -> None:
    q = BoundedBridgeQueue(maxsize=2)
    q.close()
    q.close()  # must not raise
    assert q.stats().closed is True


async def test_async_consumer_wakes_on_close() -> None:
    q = BoundedBridgeQueue(maxsize=2)

    async def consumer() -> NeutralFrameMetadata | None:
        return await q.get()

    task = asyncio.create_task(consumer())
    await asyncio.sleep(0.05)
    assert not task.done()  # blocked on empty queue
    q.close()
    result = await asyncio.wait_for(task, timeout=1.0)
    assert result is None


async def test_async_consumer_wakes_on_late_item() -> None:
    q = BoundedBridgeQueue(maxsize=2)

    async def consumer() -> NeutralFrameMetadata | None:
        return await q.get()

    task = asyncio.create_task(consumer())
    await asyncio.sleep(0.05)
    assert not task.done()
    q.put_from_thread(_frame(7))
    item = await asyncio.wait_for(task, timeout=1.0)
    assert item is not None and item.frame.frame_num == 7


# --- producer thread + slow consumer -----------------------------------


async def test_synthetic_producer_thread_no_deadlock_bounded_memory() -> None:
    q = BoundedBridgeQueue(maxsize=5, drop_policy=DropPolicy.DROP_NEWEST)
    n_put = 200

    def producer() -> None:
        for i in range(n_put):
            q.put_from_thread(_frame(i))

    t = threading.Thread(target=producer)
    t.start()

    # Drain "slowly" — yield often so producer races ahead and many puts
    # are dropped. After the producer finishes, drain the rest.
    received = 0
    deadline = time.monotonic() + 5.0
    while True:
        if not t.is_alive() and q.stats().current_size == 0:
            break
        item = await asyncio.wait_for(q.get(), timeout=1.0)
        if item is not None:
            received += 1
        await asyncio.sleep(0)
        assert q.stats().current_size <= q.maxsize
        if time.monotonic() > deadline:
            raise AssertionError("drain deadline exceeded (deadlock?)")

    t.join(timeout=1.0)
    assert not t.is_alive()

    s = q.stats()
    assert s.accepted + s.dropped == n_put
    assert s.dropped > 0  # producer outran the consumer
    assert s.consumed == received == s.accepted
    assert s.current_size == 0


async def test_consumer_wakes_on_close_with_buffered_items() -> None:
    q = BoundedBridgeQueue(maxsize=3)
    q.put_from_thread(_frame(1))
    q.put_from_thread(_frame(2))

    async def drain() -> list[int]:
        out: list[int] = []
        while True:
            it = await q.get()
            if it is None:
                break
            out.append(it.frame.frame_num)
        return out

    task = asyncio.create_task(drain())
    await asyncio.sleep(0.02)
    q.close()
    out = await asyncio.wait_for(task, timeout=1.0)
    assert out == [1, 2]


# --- stats shape --------------------------------------------------------


def test_stats_is_a_value_object() -> None:
    q = BoundedBridgeQueue(maxsize=2)
    s1 = q.stats()
    s2 = q.stats()
    assert s1 == s2
    assert isinstance(s1, BridgeQueueStats)


def test_dropped_total_equals_sum_of_per_policy_counts() -> None:
    q = BoundedBridgeQueue(maxsize=1, drop_policy=DropPolicy.DROP_NEWEST)
    q.put_from_thread(_frame(0))
    q.put_from_thread(_frame(1))  # dropped (newest)
    q.put_from_thread(_frame(2))  # dropped (newest)
    s = q.stats()
    assert s.dropped == s.dropped_newest + s.dropped_oldest == 2
