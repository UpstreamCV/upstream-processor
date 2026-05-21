"""P4-A — observation adapter contract probe tests.

Contract-level only. NO Docker / GPU / pyds / api-server. Asserts:

- ``LineCrossingState`` turns frame-level
  ``line_crossing_cumulative`` snapshots into discrete
  ``processor.line_crossing`` events with one event per increment.
- Idempotency keys are deterministic and unique per ``(session, line,
  cumulative_count)`` — both via the pure helper and via the state
  machine.
- A frame whose counter is unchanged from the prior frame emits
  nothing (dedupe correctness — the load-bearing assertion before any
  live POST happens).
- A frame whose counter rewinds (a sketchy edge case) does NOT emit
  spurious events and does NOT lower the watermark.
- An empty / missing analytics block on the frame emits nothing.
- DeepStream-specific vocabulary stays under ``source`` / ``debug``
  on the event (top-level fields are runtime-neutral).
- ``RoiOccupancyState`` emits on first observation of an ROI, on
  count change, and on interval elapse — but NOT every frame when
  unchanged.
- The ``CANDIDATE_*`` observation type strings are NOT yet in
  ``schemas.OBSERVATION_TYPES`` (the api-server widening hasn't
  happened; the adapter knows this; nothing has accidentally been
  added to the Literal whitelist).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from app.observations.observation_adapter import (
    CANDIDATE_LINE_CROSSING_TYPE,
    CANDIDATE_OBSERVATION_TYPES,
    CANDIDATE_ROI_OCCUPANCY_TYPE,
    LineCrossingState,
    RoiOccupancyState,
    build_line_crossing_idempotency_key,
    build_roi_occupancy_idempotency_key,
    deepstream_source_block,
)
from app.observations.schemas import OBSERVATION_TYPES

# --- type registry guards ---------------------------------------------


def test_line_crossing_now_in_server_whitelist() -> None:
    """P4-B widened the api-server ``ProcessorObservationType`` Literal
    and this repo's ``schemas.OBSERVATION_TYPES`` tuple in lock-step,
    so ``processor.line_crossing`` is now accepted by ``ObservationWriter``.
    Test pins that post-P4-B state — a future demotion fails here."""
    assert CANDIDATE_LINE_CROSSING_TYPE in OBSERVATION_TYPES


def test_roi_occupancy_now_in_server_whitelist() -> None:
    """P4-C widened the api-server + this repo's whitelist in
    lock-step, so ``processor.roi_occupancy`` is now accepted by
    ``ObservationWriter``. The adapter's ``RoiOccupancyState`` has
    been the production write source since this commit."""
    assert CANDIDATE_ROI_OCCUPANCY_TYPE in OBSERVATION_TYPES


def test_zone_entered_now_in_server_whitelist() -> None:
    """P4-D widened the api-server + this repo's whitelist in
    lock-step with ``processor.zone_entered``. Pinned post-P4-D
    state lock — a future demotion fails here."""
    assert "processor.zone_entered" in OBSERVATION_TYPES


def test_zone_exited_now_in_server_whitelist() -> None:
    """Same as zone_entered; the two transition types ship together."""
    assert "processor.zone_exited" in OBSERVATION_TYPES


def test_dwell_types_now_in_server_whitelist() -> None:
    """P4-E widened the api-server + this repo's whitelist with the
    three dwell types in lock-step."""
    assert "processor.dwell_started" in OBSERVATION_TYPES
    assert "processor.dwell_updated" in OBSERVATION_TYPES
    assert "processor.dwell_exceeded" in OBSERVATION_TYPES


def test_direction_track_telemetry_types_still_unsupported() -> None:
    """P4-E widens exactly three types (the dwell triplet). The next
    candidates (direction, track, telemetry, debug) MUST NOT have
    been accidentally widened — this is the spiritual successor to
    the P4-D zone guard, now pointed at the still-deferred types."""
    future_types = (
        "processor.direction_detected",
        "processor.track_state",
        "processor.track_update",
        "processor.detection_seen",
        "processor.frame_summary",
        "processor.scene_quality_warning",
    )
    for t in future_types:
        assert t not in OBSERVATION_TYPES, (
            f"{t!r} unexpectedly in OBSERVATION_TYPES — P4-E only widens "
            "the three dwell types; direction / track / telemetry / "
            "debug types stay deferred"
        )


def test_candidate_types_use_processor_dotted_namespace() -> None:
    """First-internal-observations naming convention: ``processor.<thing>``.
    Locks the namespace in this milestone so a future rename has to
    happen deliberately."""
    for t in CANDIDATE_OBSERVATION_TYPES:
        assert t.startswith("processor."), t


# --- idempotency helpers ------------------------------------------------


def test_line_crossing_idempotency_key_is_deterministic() -> None:
    sid = uuid4()
    a = build_line_crossing_idempotency_key(
        stream_session_id=sid, line_id="entry-line", cumulative_count=2
    )
    b = build_line_crossing_idempotency_key(
        stream_session_id=sid, line_id="entry-line", cumulative_count=2
    )
    assert a == b
    assert a == f"sess:{sid}:processor.line_crossing:entry-line:2"


def test_line_crossing_idempotency_key_changes_with_count() -> None:
    sid = uuid4()
    k1 = build_line_crossing_idempotency_key(
        stream_session_id=sid, line_id="entry-line", cumulative_count=1
    )
    k2 = build_line_crossing_idempotency_key(
        stream_session_id=sid, line_id="entry-line", cumulative_count=2
    )
    assert k1 != k2


def test_line_crossing_idempotency_key_distinct_per_session() -> None:
    a = build_line_crossing_idempotency_key(
        stream_session_id=uuid4(), line_id="entry-line", cumulative_count=1
    )
    b = build_line_crossing_idempotency_key(
        stream_session_id=uuid4(), line_id="entry-line", cumulative_count=1
    )
    assert a != b


def test_roi_idempotency_key_uses_sequence_not_frame_num() -> None:
    """ROI keys must NOT be keyed on a raw frame number — adapter only
    emits on change/interval, so the sequence is a separate counter."""
    sid = uuid4()
    a = build_roi_occupancy_idempotency_key(stream_session_id=sid, roi_id="entry-roi", sequence=1)
    b = build_roi_occupancy_idempotency_key(stream_session_id=sid, roi_id="entry-roi", sequence=2)
    assert a != b
    assert a == f"sess:{sid}:processor.roi_occupancy:entry-roi:1"


# --- LineCrossingState core ---------------------------------------------


def _now() -> datetime:
    return datetime(2026, 5, 20, 16, 30, 0, tzinfo=UTC)


def test_no_events_on_initial_zero() -> None:
    """First frame: cumulative=0 means no crossings yet. No event."""
    state = LineCrossingState(stream_session_id=uuid4())
    events = state.absorb_cumulative(
        cumulative_by_line={"entry-line": 0},
        occurred_at=_now(),
    )
    assert events == []


def test_one_event_on_zero_to_one_transition() -> None:
    """The simplest happy path: 0 → 1 emits exactly one event."""
    sid = uuid4()
    state = LineCrossingState(stream_session_id=sid)
    state.absorb_cumulative(cumulative_by_line={"entry-line": 0}, occurred_at=_now())
    events = state.absorb_cumulative(
        cumulative_by_line={"entry-line": 1},
        occurred_at=_now() + timedelta(seconds=1),
    )
    assert len(events) == 1
    e = events[0]
    assert e.type == CANDIDATE_LINE_CROSSING_TYPE
    assert e.stream_session_id == sid
    assert e.line_id == "entry-line"
    assert e.line_label == "entry-line"
    assert e.count_delta == 1
    assert e.cumulative_count == 1
    assert e.idempotency_key.endswith(":entry-line:1")
    # Top-level fields are runtime-neutral.
    assert "deepstream" not in e.line_id
    assert "deepstream" not in e.line_label


def test_repeat_cumulative_emits_nothing() -> None:
    """Same cumulative across two frames: dedupe holds."""
    state = LineCrossingState(stream_session_id=uuid4())
    state.absorb_cumulative(cumulative_by_line={"entry-line": 2}, occurred_at=_now())
    second = state.absorb_cumulative(
        cumulative_by_line={"entry-line": 2},
        occurred_at=_now() + timedelta(seconds=1),
    )
    assert second == []


def test_zero_to_one_to_two_emits_two_distinct_events() -> None:
    """The exact pattern from the P3.5 live run: cumulative ramped
    0 → 1 → 2 across two operator crossings. Adapter must emit
    one event per increment with distinct idempotency keys."""
    sid = uuid4()
    state = LineCrossingState(stream_session_id=sid)
    # First crossing (0 → 1).
    e1 = state.absorb_cumulative(
        cumulative_by_line={"entry-line": 1},
        occurred_at=_now(),
    )
    # Second crossing (1 → 2) at a later time.
    e2 = state.absorb_cumulative(
        cumulative_by_line={"entry-line": 2},
        occurred_at=_now() + timedelta(seconds=5),
    )
    assert len(e1) == 1
    assert len(e2) == 1
    assert e1[0].cumulative_count == 1
    assert e2[0].cumulative_count == 2
    assert e1[0].idempotency_key != e2[0].idempotency_key


def test_jump_of_three_emits_three_events_with_consecutive_counts() -> None:
    """If the producer skips frames so the cumulative jumps 0 → 3 in a
    single observation, the adapter still emits three events with
    cumulative_count = 1, 2, 3 (each its own idempotency key)."""
    state = LineCrossingState(stream_session_id=uuid4())
    events = state.absorb_cumulative(
        cumulative_by_line={"entry-line": 3},
        occurred_at=_now(),
    )
    assert [e.cumulative_count for e in events] == [1, 2, 3]
    assert [e.count_delta for e in events] == [1, 1, 1]
    assert len({e.idempotency_key for e in events}) == 3


def test_counter_rewind_does_not_emit_and_does_not_lower_watermark() -> None:
    """Defensive: producer cannot trick the adapter into re-emitting by
    rewinding. We saw N=5 once; if the next frame says N=2 we treat
    that as noise."""
    state = LineCrossingState(stream_session_id=uuid4())
    state.absorb_cumulative(cumulative_by_line={"entry-line": 5}, occurred_at=_now())
    rewound = state.absorb_cumulative(
        cumulative_by_line={"entry-line": 2},
        occurred_at=_now() + timedelta(seconds=1),
    )
    assert rewound == []
    # Watermark must still be 5; the next genuine new crossing emits at 6.
    later = state.absorb_cumulative(
        cumulative_by_line={"entry-line": 6},
        occurred_at=_now() + timedelta(seconds=2),
    )
    assert len(later) == 1
    assert later[0].cumulative_count == 6


def test_empty_cumulative_dict_emits_nothing() -> None:
    """Frame with no analytics block (or analytics block has no
    line_crossing_cumulative) → empty dict → no event."""
    state = LineCrossingState(stream_session_id=uuid4())
    assert state.absorb_cumulative(cumulative_by_line={}, occurred_at=_now()) == []


def test_multiple_lines_tracked_independently() -> None:
    """Cumulative counts are keyed per line label. Two lines crossing
    at different times produce events with different ``line_id``."""
    state = LineCrossingState(stream_session_id=uuid4())
    state.absorb_cumulative(
        cumulative_by_line={"entry-line": 0, "exit-line": 0}, occurred_at=_now()
    )
    events = state.absorb_cumulative(
        cumulative_by_line={"entry-line": 1, "exit-line": 0},
        occurred_at=_now() + timedelta(seconds=1),
    )
    assert len(events) == 1
    assert events[0].line_id == "entry-line"
    events2 = state.absorb_cumulative(
        cumulative_by_line={"entry-line": 1, "exit-line": 1},
        occurred_at=_now() + timedelta(seconds=2),
    )
    assert len(events2) == 1
    assert events2[0].line_id == "exit-line"


# --- payload shape -----------------------------------------------------


def test_event_payload_is_runtime_neutral_at_top_level() -> None:
    """DeepStream specifics must live in ``source`` / ``debug`` on the
    payload, not as first-class keys. Top-level shape is what a future
    Triton / OpenCV runtime would emit too."""
    sid = uuid4()
    state = LineCrossingState(stream_session_id=sid)
    state.absorb_cumulative(cumulative_by_line={"entry-line": 0}, occurred_at=_now())
    events = state.absorb_cumulative(
        cumulative_by_line={"entry-line": 1},
        occurred_at=_now() + timedelta(seconds=1),
        track_id="42",
        class_id=2,
        class_label="person",
        source=deepstream_source_block(
            package_id="deepstream.analytics_probe.v0",
            tracker_algorithm="iou",
        ),
        debug={"frame_num": 1234, "geometry_source": "probe_default"},
    )
    body = events[0].to_payload()
    # Top-level keys: runtime-neutral.
    assert "line_id" in body and "line_label" in body
    assert "cumulative_count" in body and "count_delta" in body
    assert "track_id" in body and "class_id" in body and "class_label" in body
    assert "idempotency_key" in body
    # No DeepStream vocabulary at the top level.
    for forbidden in (
        "runtime",
        "package_id",
        "metadata_source",
        "analytics_plugin",
        "tracker_algorithm",
        "frame_num",
        "raw_line_crossing_cumulative",
    ):
        assert forbidden not in body, f"top-level leak: {forbidden}"
    # Provenance is under source.
    assert body["source"]["runtime"] == "deepstream"
    assert body["source"]["package_id"] == "deepstream.analytics_probe.v0"
    assert body["source"]["metadata_source"] == "pyds_pad_probe"
    assert body["source"]["analytics_plugin"] == "nvdsanalytics"
    assert body["source"]["tracker_algorithm"] == "iou"
    # Raw counters are under debug.
    assert body["debug"]["frame_num"] == 1234
    assert body["debug"]["geometry_source"] == "probe_default"


def test_event_payload_omits_optional_fields_when_absent() -> None:
    """When the producer didn't have a track_id / class / direction, the
    payload omits those keys entirely (cleaner than ``"track_id": null``
    everywhere)."""
    state = LineCrossingState(stream_session_id=uuid4())
    state.absorb_cumulative(cumulative_by_line={"entry-line": 0}, occurred_at=_now())
    events = state.absorb_cumulative(
        cumulative_by_line={"entry-line": 1},
        occurred_at=_now() + timedelta(seconds=1),
    )
    body = events[0].to_payload()
    assert "track_id" not in body
    assert "class_id" not in body
    assert "class_label" not in body
    assert "direction" not in body


# --- ROI occupancy state -----------------------------------------------


def test_roi_first_observation_always_emits() -> None:
    """Even a zero-count ROI on first sight should emit so the consumer
    knows the ROI exists. P4-C labels this case as ``emit_reason="first"``
    in the event's debug dict."""
    state = RoiOccupancyState(
        stream_session_id=uuid4(), emit_on_change=True, min_interval_seconds=None
    )
    events = state.absorb_roi_counts(roi_counts={"entry-roi": 0}, occurred_at=_now())
    assert len(events) == 1
    assert events[0].roi_id == "entry-roi"
    assert events[0].total_count == 0
    assert events[0].sequence == 1
    assert events[0].debug["emit_reason"] == "first"


def test_roi_no_emit_when_unchanged_and_no_interval() -> None:
    """Steady state with neither change nor interval = no events."""
    state = RoiOccupancyState(
        stream_session_id=uuid4(), emit_on_change=True, min_interval_seconds=None
    )
    state.absorb_roi_counts(roi_counts={"entry-roi": 1}, occurred_at=_now())
    second = state.absorb_roi_counts(
        roi_counts={"entry-roi": 1}, occurred_at=_now() + timedelta(seconds=10)
    )
    assert second == []


def test_roi_emits_on_change() -> None:
    """Total count went 1 → 2 between frames → one event with
    ``emit_reason="change"``."""
    state = RoiOccupancyState(
        stream_session_id=uuid4(), emit_on_change=True, min_interval_seconds=None
    )
    state.absorb_roi_counts(roi_counts={"entry-roi": 1}, occurred_at=_now())
    second = state.absorb_roi_counts(
        roi_counts={"entry-roi": 2}, occurred_at=_now() + timedelta(seconds=1)
    )
    assert len(second) == 1
    assert second[0].total_count == 2
    assert second[0].sequence == 2  # second emit for this ROI
    assert second[0].debug["emit_reason"] == "change"


def test_roi_emits_on_interval_even_when_unchanged() -> None:
    """If the count hasn't moved but ``min_interval_seconds`` has
    elapsed, emit a heartbeat-style snapshot labelled
    ``emit_reason="interval"``."""
    state = RoiOccupancyState(
        stream_session_id=uuid4(), emit_on_change=False, min_interval_seconds=30.0
    )
    state.absorb_roi_counts(roi_counts={"entry-roi": 1}, occurred_at=_now())
    # 10 s later — too soon.
    soon = state.absorb_roi_counts(
        roi_counts={"entry-roi": 1}, occurred_at=_now() + timedelta(seconds=10)
    )
    assert soon == []
    # 31 s later — interval elapsed.
    later = state.absorb_roi_counts(
        roi_counts={"entry-roi": 1}, occurred_at=_now() + timedelta(seconds=31)
    )
    assert len(later) == 1
    assert later[0].sequence == 2
    assert later[0].debug["emit_reason"] == "interval"


def test_roi_class_counts_threaded_through() -> None:
    """When the adapter gets a per-class breakdown, it includes it in
    the event payload."""
    state = RoiOccupancyState(
        stream_session_id=uuid4(), emit_on_change=True, min_interval_seconds=None
    )
    events = state.absorb_roi_counts(
        roi_counts={"entry-roi": 3},
        class_counts_by_roi={"entry-roi": {"person": 2, "car": 1}},
        occurred_at=_now(),
    )
    body = events[0].to_payload()
    assert body["total_count"] == 3
    assert body["class_counts"] == {"person": 2, "car": 1}


def test_roi_emit_reason_lives_in_debug_dict() -> None:
    """The emit reason rides under ``debug.emit_reason``, NOT as a
    top-level payload field — keeps the runtime-neutral surface clean
    while still letting an operator inspect why a heartbeat fired."""
    state = RoiOccupancyState(
        stream_session_id=uuid4(), emit_on_change=True, min_interval_seconds=30.0
    )
    e1 = state.absorb_roi_counts(roi_counts={"entry-roi": 0}, occurred_at=_now())
    e2 = state.absorb_roi_counts(
        roi_counts={"entry-roi": 1}, occurred_at=_now() + timedelta(seconds=1)
    )
    e3 = state.absorb_roi_counts(
        roi_counts={"entry-roi": 1}, occurred_at=_now() + timedelta(seconds=35)
    )
    bodies = [e[0].to_payload() for e in (e1, e2, e3)]
    # Top-level body must NOT carry emit_reason.
    for body in bodies:
        assert "emit_reason" not in body
        assert body["debug"]["emit_reason"] in ("first", "change", "interval")
    # And: the three cases produced the three reasons in order.
    assert bodies[0]["debug"]["emit_reason"] == "first"
    assert bodies[1]["debug"]["emit_reason"] == "change"
    assert bodies[2]["debug"]["emit_reason"] == "interval"


def test_roi_emit_reason_does_not_overwrite_caller_debug() -> None:
    """If the caller passes extra debug fields (frame_num,
    raw_roi_counts, …), the adapter merges them with ``emit_reason``;
    neither side drops the other's keys."""
    state = RoiOccupancyState(
        stream_session_id=uuid4(), emit_on_change=True, min_interval_seconds=None
    )
    events = state.absorb_roi_counts(
        roi_counts={"entry-roi": 1},
        occurred_at=_now(),
        debug={"frame_num": 42, "geometry_source": "probe_default"},
    )
    body = events[0].to_payload()
    assert body["debug"]["emit_reason"] == "first"
    assert body["debug"]["frame_num"] == 42
    assert body["debug"]["geometry_source"] == "probe_default"


# --- P4-C cleanup: debounce / hysteresis -------------------------------


def test_roi_change_stability_frames_must_be_at_least_one() -> None:
    """Adapter rejects 0 / negative thresholds at construction so a
    typo in env config fails fast, not silently."""
    import pytest

    with pytest.raises(ValueError, match="change_stability_frames must be >= 1"):
        RoiOccupancyState(stream_session_id=uuid4(), change_stability_frames=0)
    with pytest.raises(ValueError, match="change_stability_frames must be >= 1"):
        RoiOccupancyState(stream_session_id=uuid4(), change_stability_frames=-3)


def test_roi_stability_1_matches_legacy_immediate_change() -> None:
    """stability=1 is the P4-C legacy path: any new value emits on
    the next frame. Critical for backward compat (the adapter dataclass
    default is 1; the production Settings default is 5)."""
    state = RoiOccupancyState(
        stream_session_id=uuid4(),
        emit_on_change=True,
        min_interval_seconds=None,
        change_stability_frames=1,
    )
    state.absorb_roi_counts(roi_counts={"entry-roi": 0}, occurred_at=_now())
    second = state.absorb_roi_counts(
        roi_counts={"entry-roi": 1}, occurred_at=_now() + timedelta(seconds=1)
    )
    assert len(second) == 1
    assert second[0].total_count == 1
    assert second[0].debug["emit_reason"] == "change"


def test_roi_stability_3_does_not_emit_on_first_changed_frame() -> None:
    """With debounce=3, the first frame at the new count only updates
    pending state. No event yet."""
    state = RoiOccupancyState(
        stream_session_id=uuid4(),
        emit_on_change=True,
        min_interval_seconds=None,
        change_stability_frames=3,
    )
    state.absorb_roi_counts(roi_counts={"entry-roi": 0}, occurred_at=_now())
    events = state.absorb_roi_counts(
        roi_counts={"entry-roi": 1}, occurred_at=_now() + timedelta(seconds=1)
    )
    assert events == []


def test_roi_stability_3_emits_on_third_consecutive_frame() -> None:
    """With debounce=3, the third consecutive frame at the new count
    commits the change. ``pending_seen_frames_by_roi`` is the
    accumulator; we don't expose it in the payload but the test asserts
    the emit fires exactly when the threshold is met."""
    state = RoiOccupancyState(
        stream_session_id=uuid4(),
        emit_on_change=True,
        min_interval_seconds=None,
        change_stability_frames=3,
    )
    state.absorb_roi_counts(roi_counts={"entry-roi": 0}, occurred_at=_now())
    assert (
        state.absorb_roi_counts(
            roi_counts={"entry-roi": 1}, occurred_at=_now() + timedelta(seconds=1)
        )
        == []
    )
    assert (
        state.absorb_roi_counts(
            roi_counts={"entry-roi": 1}, occurred_at=_now() + timedelta(seconds=2)
        )
        == []
    )
    events = state.absorb_roi_counts(
        roi_counts={"entry-roi": 1}, occurred_at=_now() + timedelta(seconds=3)
    )
    assert len(events) == 1
    assert events[0].total_count == 1
    assert events[0].debug["emit_reason"] == "change"
    # Sequence is 2 (first emit was sequence=1).
    assert events[0].sequence == 2


def test_roi_pending_resets_when_count_flips_back_before_threshold() -> None:
    """The 0 → 1 → 0 → 1 → … flap pattern from the P4-C live run.
    With debounce=3, the count returning to 0 mid-flap clears the
    pending candidate; the subsequent 1 starts over from 1, not 2."""
    state = RoiOccupancyState(
        stream_session_id=uuid4(),
        emit_on_change=True,
        min_interval_seconds=None,
        change_stability_frames=3,
    )
    state.absorb_roi_counts(roi_counts={"entry-roi": 0}, occurred_at=_now())
    # Two frames at 1 — pending=1, seen=2.
    state.absorb_roi_counts(roi_counts={"entry-roi": 1}, occurred_at=_now() + timedelta(seconds=1))
    state.absorb_roi_counts(roi_counts={"entry-roi": 1}, occurred_at=_now() + timedelta(seconds=2))
    # Count flips back to 0 (== last emitted) — pending clears.
    flip = state.absorb_roi_counts(
        roi_counts={"entry-roi": 0}, occurred_at=_now() + timedelta(seconds=3)
    )
    assert flip == []
    # Now 1 again — pending starts FROM 1, not from 3.
    again_1 = state.absorb_roi_counts(
        roi_counts={"entry-roi": 1}, occurred_at=_now() + timedelta(seconds=4)
    )
    again_2 = state.absorb_roi_counts(
        roi_counts={"entry-roi": 1}, occurred_at=_now() + timedelta(seconds=5)
    )
    assert again_1 == []
    assert again_2 == []
    # Third consecutive 1 should now commit.
    again_3 = state.absorb_roi_counts(
        roi_counts={"entry-roi": 1}, occurred_at=_now() + timedelta(seconds=6)
    )
    assert len(again_3) == 1
    assert again_3[0].total_count == 1
    assert again_3[0].debug["emit_reason"] == "change"


def test_roi_pending_switches_to_different_new_candidate() -> None:
    """Pending was 1 (seen 2x), then frame is 2 (different new value).
    Pending must reset to 2 with seen=1, NOT carry over the old 2x
    accumulation."""
    state = RoiOccupancyState(
        stream_session_id=uuid4(),
        emit_on_change=True,
        min_interval_seconds=None,
        change_stability_frames=3,
    )
    state.absorb_roi_counts(roi_counts={"entry-roi": 0}, occurred_at=_now())
    state.absorb_roi_counts(roi_counts={"entry-roi": 1}, occurred_at=_now() + timedelta(seconds=1))
    state.absorb_roi_counts(roi_counts={"entry-roi": 1}, occurred_at=_now() + timedelta(seconds=2))
    # Switch candidate to 2 — pending state replaces, doesn't increment.
    switch = state.absorb_roi_counts(
        roi_counts={"entry-roi": 2}, occurred_at=_now() + timedelta(seconds=3)
    )
    assert switch == []
    # Need TWO more consecutive 2s to hit the threshold.
    assert (
        state.absorb_roi_counts(
            roi_counts={"entry-roi": 2}, occurred_at=_now() + timedelta(seconds=4)
        )
        == []
    )
    final = state.absorb_roi_counts(
        roi_counts={"entry-roi": 2}, occurred_at=_now() + timedelta(seconds=5)
    )
    assert len(final) == 1
    assert final[0].total_count == 2
    assert final[0].debug["emit_reason"] == "change"


def test_roi_interval_heartbeat_reports_last_stable_count_during_pending() -> None:
    """Key invariant: when the interval elapses while a candidate is
    pending but not yet stable, the heartbeat must report the LAST
    STABLE count, not the unstable candidate. Consumers cannot mistake
    a heartbeat for a real state change."""
    state = RoiOccupancyState(
        stream_session_id=uuid4(),
        emit_on_change=True,
        min_interval_seconds=30.0,
        change_stability_frames=5,
    )
    # First emit at t=0 commits count=2.
    state.absorb_roi_counts(roi_counts={"entry-roi": 2}, occurred_at=_now())
    # Two frames at 0 — pending=0, seen=2. Threshold (5) not met.
    state.absorb_roi_counts(roi_counts={"entry-roi": 0}, occurred_at=_now() + timedelta(seconds=10))
    state.absorb_roi_counts(roi_counts={"entry-roi": 0}, occurred_at=_now() + timedelta(seconds=20))
    # 31 s elapsed — heartbeat fires. Must report the LAST STABLE count (2),
    # not the pending candidate (0).
    heartbeat = state.absorb_roi_counts(
        roi_counts={"entry-roi": 0}, occurred_at=_now() + timedelta(seconds=31)
    )
    assert len(heartbeat) == 1
    assert heartbeat[0].total_count == 2  # NOT 0
    assert heartbeat[0].debug["emit_reason"] == "interval"
    # Pending state survives the heartbeat: the seen counter must
    # have advanced (we just saw another 0).
    assert state.pending_count_by_roi.get("entry-roi") == {"__total": 0}
    assert state.pending_seen_frames_by_roi.get("entry-roi") == 3


def test_roi_pending_eventually_commits_after_heartbeat() -> None:
    """If pending count stays pending through a heartbeat AND
    eventually accumulates enough frames, the next emit is a regular
    'change' to the new value. Two distinct emits total: heartbeat
    (stable count) + change (new count)."""
    state = RoiOccupancyState(
        stream_session_id=uuid4(),
        emit_on_change=True,
        min_interval_seconds=30.0,
        change_stability_frames=10,  # threshold high enough that the
        # heartbeat fires BEFORE the pending count would commit.
    )
    state.absorb_roi_counts(roi_counts={"entry-roi": 2}, occurred_at=_now())  # first

    # Three frames at 0 over the interval window — pending=0, seen=3
    # (still well below threshold=10).
    state.absorb_roi_counts(roi_counts={"entry-roi": 0}, occurred_at=_now() + timedelta(seconds=10))
    state.absorb_roi_counts(roi_counts={"entry-roi": 0}, occurred_at=_now() + timedelta(seconds=20))
    state.absorb_roi_counts(roi_counts={"entry-roi": 0}, occurred_at=_now() + timedelta(seconds=29))

    # Heartbeat at 31s — must emit STABLE count=2, not the pending 0.
    hb = state.absorb_roi_counts(
        roi_counts={"entry-roi": 0}, occurred_at=_now() + timedelta(seconds=31)
    )
    assert len(hb) == 1
    assert hb[0].total_count == 2  # NOT 0
    assert hb[0].debug["emit_reason"] == "interval"
    # Pending state survived the heartbeat: count=0 was just seen 4 times
    # (frames at offsets 10, 20, 29, 31).
    assert state.pending_seen_frames_by_roi.get("entry-roi") == 4

    # Six more frames at 0 (4 + 6 = 10 = threshold) commit the change.
    for offset in (32, 33, 34, 35, 36):
        no_emit = state.absorb_roi_counts(
            roi_counts={"entry-roi": 0}, occurred_at=_now() + timedelta(seconds=offset)
        )
        assert no_emit == []
    final = state.absorb_roi_counts(
        roi_counts={"entry-roi": 0}, occurred_at=_now() + timedelta(seconds=37)
    )
    assert len(final) == 1
    assert final[0].total_count == 0
    assert final[0].debug["emit_reason"] == "change"
    assert final[0].sequence == 3  # first=1, interval=2, change=3


def test_roi_sequence_increments_only_on_actual_emits() -> None:
    """A flap that's suppressed by the debounce must NOT advance the
    sequence — sequences are emit counters, not frame counters."""
    state = RoiOccupancyState(
        stream_session_id=uuid4(),
        emit_on_change=True,
        min_interval_seconds=None,
        change_stability_frames=5,
    )
    e1 = state.absorb_roi_counts(roi_counts={"entry-roi": 0}, occurred_at=_now())
    # 4 frames at count=1, all suppressed by debounce.
    for i in range(1, 5):
        suppressed = state.absorb_roi_counts(
            roi_counts={"entry-roi": 1}, occurred_at=_now() + timedelta(seconds=i)
        )
        assert suppressed == []
    # 5th frame at 1 finally commits.
    e2 = state.absorb_roi_counts(
        roi_counts={"entry-roi": 1}, occurred_at=_now() + timedelta(seconds=5)
    )
    assert e1[0].sequence == 1
    assert e2[0].sequence == 2
    assert e1[0].idempotency_key.endswith(":1")
    assert e2[0].idempotency_key.endswith(":2")


def test_roi_idempotency_key_increments_per_emit() -> None:
    """The ``sequence`` field on the event must equal the sequence
    embedded in the idempotency key — keys are unique per emit, not
    per frame."""
    sid = uuid4()
    state = RoiOccupancyState(stream_session_id=sid, emit_on_change=True, min_interval_seconds=None)
    e1 = state.absorb_roi_counts(roi_counts={"entry-roi": 1}, occurred_at=_now())
    e2 = state.absorb_roi_counts(
        roi_counts={"entry-roi": 2}, occurred_at=_now() + timedelta(seconds=1)
    )
    k1 = e1[0].idempotency_key
    k2 = e2[0].idempotency_key
    assert k1 != k2
    assert k1.endswith(":1")
    assert k2.endswith(":2")
    assert e1[0].sequence == 1 and e2[0].sequence == 2


# --- source helper -----------------------------------------------------


def test_deepstream_source_block_has_stable_shape() -> None:
    src = deepstream_source_block(
        package_id="deepstream.analytics_probe.v0", tracker_algorithm="iou"
    )
    assert src == {
        "runtime": "deepstream",
        "package_id": "deepstream.analytics_probe.v0",
        "metadata_source": "pyds_pad_probe",
        "analytics_plugin": "nvdsanalytics",
        "tracker_algorithm": "iou",
    }


def test_deepstream_source_block_without_tracker() -> None:
    """The metadata probe (no tracker) could in theory hand the helper
    just a package_id and get a tracker-less source block back."""
    src = deepstream_source_block(package_id="deepstream.metadata_probe.v0")
    assert "tracker_algorithm" not in src
    assert src["runtime"] == "deepstream"


# --- malformed inputs are skipped, not crashing ------------------------


def test_line_crossing_state_handles_no_lines_dict_safely() -> None:
    """Empty dict from a frame with no analytics block must not crash."""
    state = LineCrossingState(stream_session_id=uuid4())
    assert state.absorb_cumulative(cumulative_by_line={}, occurred_at=_now()) == []


def test_roi_state_handles_missing_class_counts_safely() -> None:
    state = RoiOccupancyState(
        stream_session_id=uuid4(), emit_on_change=True, min_interval_seconds=None
    )
    events = state.absorb_roi_counts(
        roi_counts={"entry-roi": 1},
        class_counts_by_roi=None,
        occurred_at=_now(),
    )
    assert events[0].class_counts == {}


# --- regression: must not import observation writer at adapter import --


def test_adapter_module_does_not_import_writer_or_api_client() -> None:
    """The adapter is pure / dependency-free. Importing it must NOT
    pull in the http client or the writer — those belong to P4-B."""
    import sys

    # If the adapter accidentally imported the writer, that module would
    # appear in sys.modules after the import above. (writer is independently
    # importable, so we only assert what is reachable from THIS module's
    # explicit imports.)
    import app.observations.observation_adapter as adapter_mod

    # The module's own __dict__ must not contain the writer / api client.
    assert "ObservationWriter" not in vars(adapter_mod)
    assert "UpstreamApiClient" not in vars(adapter_mod)
    # And: sanity, the module is what we think it is.
    assert "LineCrossingState" in vars(adapter_mod)
    # Touch sys.modules to confirm the import worked at all.
    assert "app.observations.observation_adapter" in sys.modules


# --- value pass-through smoke -----------------------------------------


def test_line_crossing_payload_carries_session_in_idempotency_key() -> None:
    """The session UUID is what makes the key unique across cameras."""
    sid = uuid4()
    state = LineCrossingState(stream_session_id=sid)
    state.absorb_cumulative(cumulative_by_line={"entry-line": 0}, occurred_at=_now())
    events = state.absorb_cumulative(
        cumulative_by_line={"entry-line": 1},
        occurred_at=_now() + timedelta(seconds=1),
    )
    assert str(sid) in events[0].idempotency_key


@pytest.mark.parametrize("cum", [10, 99, 1_000_000])
def test_line_crossing_handles_large_cumulative_counts(cum: int) -> None:
    """Counter can be anything the SDK returns; the key just stringifies
    the int."""
    state = LineCrossingState(stream_session_id=uuid4())
    events = state.absorb_cumulative(cumulative_by_line={"entry-line": cum}, occurred_at=_now())
    assert events[-1].cumulative_count == cum
    assert events[-1].idempotency_key.endswith(f":{cum}")
