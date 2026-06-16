"""P4-D — ``ZoneMembershipState`` adapter unit tests.

Pure / dependency-free. NO Docker / GPU / pyds / api-server / worker.
The state machine is small (per-(track, zone) committed/pending), so
the tests cover every transition the adapter can take: missing
tracker_id, missing/malformed labels, first-stable-inside,
first-observed-outside (no-op), repeated inside, stable exit,
single-frame flap suppression, re-enter sequence,
disappearance-doesn't-emit, multi-track / multi-zone independence,
class_id / class_label preservation, debug shape, and idempotency
key format.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from app.observations.lifecycle_policy import LifecyclePolicy
from app.observations.observation_adapter import (
    CANDIDATE_ZONE_ENTERED_TYPE,
    CANDIDATE_ZONE_EXITED_TYPE,
    ZoneMembershipState,
    ZoneTransitionEvent,
    build_zone_entered_idempotency_key,
    build_zone_exited_idempotency_key,
)


def _now() -> datetime:
    return datetime(2026, 5, 21, 12, 0, 0, tzinfo=UTC)


# --- validation -------------------------------------------------------


def test_state_rejects_membership_stability_frames_below_one() -> None:
    """A typo in env config that gives a 0 / negative threshold must
    fail at adapter construction, not silently disable the debounce."""
    with pytest.raises(ValueError, match="membership_stability_frames must be >= 1"):
        ZoneMembershipState(stream_session_id=uuid4(), membership_stability_frames=0)
    with pytest.raises(ValueError, match="membership_stability_frames must be >= 1"):
        ZoneMembershipState(stream_session_id=uuid4(), membership_stability_frames=-7)


def test_state_default_stability_is_one() -> None:
    """Adapter dataclass default = legacy immediate-transition behavior
    so direct unit-test callers don't have to specify; the production
    default of 5 lives in ``Settings``."""
    s = ZoneMembershipState(stream_session_id=uuid4())
    assert s.membership_stability_frames == 1


# --- core semantics ---------------------------------------------------


def _absorb(
    state: ZoneMembershipState,
    track_id: str,
    *,
    zone_set: set[str],
    roi_labels: set[str],
    occurred_at: datetime,
    class_id: int | None = 2,
    class_label: str | None = "person",
) -> list[ZoneTransitionEvent]:
    return state.absorb_object(
        track_id=track_id,
        zone_set=zone_set,
        roi_labels=roi_labels,
        occurred_at=occurred_at,
        class_id=class_id,
        class_label=class_label,
        source={"runtime": "deepstream"},
        debug={"frame_num": 1},
    )


def test_first_observed_outside_is_no_op() -> None:
    """A track observed in this zone-set but NOT inside any zone has
    no committed prior state; we don't track 'outside' until we've
    committed an 'inside' first. So nothing emits, and the state is
    unchanged (no spurious zone_exited)."""
    s = ZoneMembershipState(stream_session_id=uuid4(), membership_stability_frames=3)
    events = _absorb(s, track_id="t0", zone_set={"entry-roi"}, roi_labels=set(), occurred_at=_now())
    assert events == []
    # No committed state — track was never seen inside.
    assert ("t0", "entry-roi") not in s.stable_membership
    assert ("t0", "entry-roi") not in s.pending_membership


def test_first_stable_inside_emits_zone_entered_after_n_frames() -> None:
    """N=3: first two inside frames pend; the third commits + emits."""
    sid = uuid4()
    s = ZoneMembershipState(stream_session_id=sid, membership_stability_frames=3)
    e1 = _absorb(
        s, track_id="t0", zone_set={"entry-roi"}, roi_labels={"entry-roi"}, occurred_at=_now()
    )
    e2 = _absorb(
        s,
        track_id="t0",
        zone_set={"entry-roi"},
        roi_labels={"entry-roi"},
        occurred_at=_now() + timedelta(seconds=1),
    )
    e3 = _absorb(
        s,
        track_id="t0",
        zone_set={"entry-roi"},
        roi_labels={"entry-roi"},
        occurred_at=_now() + timedelta(seconds=2),
    )
    assert e1 == [] and e2 == []
    assert len(e3) == 1
    ev = e3[0]
    assert ev.type == CANDIDATE_ZONE_ENTERED_TYPE
    assert ev.zone_id == "entry-roi"
    assert ev.track_id == "t0"
    assert ev.sequence == 1
    assert ev.idempotency_key == f"sess:{sid}:processor.zone_entered:entry-roi:t0:1"


def test_stability_one_emits_immediately() -> None:
    """Backward-compat path: with threshold=1 the first inside frame
    commits + emits straight away (matches the dataclass default)."""
    s = ZoneMembershipState(stream_session_id=uuid4(), membership_stability_frames=1)
    events = _absorb(
        s, track_id="t0", zone_set={"entry-roi"}, roi_labels={"entry-roi"}, occurred_at=_now()
    )
    assert len(events) == 1
    assert events[0].type == CANDIDATE_ZONE_ENTERED_TYPE


def test_repeated_inside_does_not_duplicate() -> None:
    """After committing zone_entered, additional stable-inside frames
    must NOT re-emit."""
    s = ZoneMembershipState(stream_session_id=uuid4(), membership_stability_frames=1)
    _absorb(s, track_id="t0", zone_set={"entry-roi"}, roi_labels={"entry-roi"}, occurred_at=_now())
    for offset in range(1, 6):
        again = _absorb(
            s,
            track_id="t0",
            zone_set={"entry-roi"},
            roi_labels={"entry-roi"},
            occurred_at=_now() + timedelta(seconds=offset),
        )
        assert again == [], offset


def test_stable_outside_after_inside_emits_zone_exited() -> None:
    """Inside committed → N frames at outside emits zone_exited."""
    sid = uuid4()
    s = ZoneMembershipState(stream_session_id=sid, membership_stability_frames=2)
    _absorb(s, track_id="t0", zone_set={"entry-roi"}, roi_labels={"entry-roi"}, occurred_at=_now())
    _absorb(
        s,
        track_id="t0",
        zone_set={"entry-roi"},
        roi_labels={"entry-roi"},
        occurred_at=_now() + timedelta(seconds=1),
    )
    # Two more frames at outside to meet threshold=2.
    a = _absorb(
        s,
        track_id="t0",
        zone_set={"entry-roi"},
        roi_labels=set(),
        occurred_at=_now() + timedelta(seconds=2),
    )
    b = _absorb(
        s,
        track_id="t0",
        zone_set={"entry-roi"},
        roi_labels=set(),
        occurred_at=_now() + timedelta(seconds=3),
    )
    assert a == []
    assert len(b) == 1
    ev = b[0]
    assert ev.type == CANDIDATE_ZONE_EXITED_TYPE
    assert ev.zone_id == "entry-roi"
    assert ev.track_id == "t0"
    assert ev.sequence == 1
    assert ev.idempotency_key == f"sess:{sid}:processor.zone_exited:entry-roi:t0:1"


def test_one_frame_outside_flap_does_not_emit_exit() -> None:
    """Inside committed, then a single outside frame, then back inside.
    Pending should clear; no zone_exited."""
    s = ZoneMembershipState(stream_session_id=uuid4(), membership_stability_frames=3)
    # Threshold=3 so this is unambiguous.
    _absorb(s, "t0", zone_set={"entry-roi"}, roi_labels={"entry-roi"}, occurred_at=_now())
    _absorb(
        s,
        "t0",
        zone_set={"entry-roi"},
        roi_labels={"entry-roi"},
        occurred_at=_now() + timedelta(seconds=1),
    )
    _absorb(
        s,
        "t0",
        zone_set={"entry-roi"},
        roi_labels={"entry-roi"},
        occurred_at=_now() + timedelta(seconds=2),
    )
    # The third inside frame commits zone_entered.
    # Single outside flap.
    flap = _absorb(
        s, "t0", zone_set={"entry-roi"}, roi_labels=set(), occurred_at=_now() + timedelta(seconds=3)
    )
    # Back inside — pending should clear because the count returns to
    # the committed value.
    back = _absorb(
        s,
        "t0",
        zone_set={"entry-roi"},
        roi_labels={"entry-roi"},
        occurred_at=_now() + timedelta(seconds=4),
    )
    assert flap == []
    assert back == []
    # Committed is still True; no spurious exit emitted.
    assert s.stable_membership[("t0", "entry-roi")] is True


def test_inside_outside_inside_short_flap_does_not_emit_false_exit() -> None:
    """The exact pattern P4-C cleanup targeted but at the zone level."""
    s = ZoneMembershipState(stream_session_id=uuid4(), membership_stability_frames=5)
    # Commit inside via 5 stable frames.
    for off in range(5):
        _absorb(
            s,
            "t0",
            zone_set={"entry-roi"},
            roi_labels={"entry-roi"},
            occurred_at=_now() + timedelta(seconds=off),
        )
    # Flap: 2 outside frames (below threshold), then back inside for 2.
    for off in range(5, 7):
        evs = _absorb(
            s,
            "t0",
            zone_set={"entry-roi"},
            roi_labels=set(),
            occurred_at=_now() + timedelta(seconds=off),
        )
        assert evs == []
    for off in range(7, 9):
        evs = _absorb(
            s,
            "t0",
            zone_set={"entry-roi"},
            roi_labels={"entry-roi"},
            occurred_at=_now() + timedelta(seconds=off),
        )
        # Returning to committed clears pending; no event.
        assert evs == []


def test_different_track_ids_tracked_independently() -> None:
    """t0 inside doesn't affect t1's state for the same zone."""
    s = ZoneMembershipState(stream_session_id=uuid4(), membership_stability_frames=1)
    e1 = _absorb(s, "t0", zone_set={"entry-roi"}, roi_labels={"entry-roi"}, occurred_at=_now())
    e2 = _absorb(
        s,
        "t1",
        zone_set={"entry-roi"},
        roi_labels={"entry-roi"},
        occurred_at=_now() + timedelta(seconds=1),
    )
    assert len(e1) == 1 and e1[0].track_id == "t0"
    assert len(e2) == 1 and e2[0].track_id == "t1"


def test_different_zones_tracked_independently() -> None:
    """One track in two zones: each (track, zone) pair has its own
    committed/pending/sequence state."""
    s = ZoneMembershipState(stream_session_id=uuid4(), membership_stability_frames=1)
    events = _absorb(
        s,
        "t0",
        zone_set={"zone-a", "zone-b"},
        roi_labels={"zone-a"},
        occurred_at=_now(),
    )
    # Only one event: zone-a entered. zone-b is observed-outside-with-
    # no-prior-inside → no-op (Case A).
    assert len(events) == 1
    assert events[0].zone_id == "zone-a"


def test_re_enter_after_exit_increments_zone_entered_sequence() -> None:
    """Per-(track, zone, type) counter — zone_entered sequence advances
    on every re-enter; zone_exited sequence is independent."""
    sid = uuid4()
    s = ZoneMembershipState(stream_session_id=sid, membership_stability_frames=1)
    # First enter.
    e1 = _absorb(s, "t0", zone_set={"entry-roi"}, roi_labels={"entry-roi"}, occurred_at=_now())
    # First exit.
    x1 = _absorb(
        s,
        "t0",
        zone_set={"entry-roi"},
        roi_labels=set(),
        occurred_at=_now() + timedelta(seconds=1),
    )
    # Re-enter.
    e2 = _absorb(
        s,
        "t0",
        zone_set={"entry-roi"},
        roi_labels={"entry-roi"},
        occurred_at=_now() + timedelta(seconds=2),
    )
    # Re-exit.
    x2 = _absorb(
        s,
        "t0",
        zone_set={"entry-roi"},
        roi_labels=set(),
        occurred_at=_now() + timedelta(seconds=3),
    )
    assert e1[0].sequence == 1
    assert x1[0].sequence == 1
    assert e2[0].sequence == 2
    assert x2[0].sequence == 2
    # All four idempotency keys distinct.
    keys = {
        e1[0].idempotency_key,
        x1[0].idempotency_key,
        e2[0].idempotency_key,
        x2[0].idempotency_key,
    }
    assert len(keys) == 4
    # Distinct namespaces by transition type.
    assert e1[0].idempotency_key.startswith(f"sess:{sid}:processor.zone_entered:")
    assert x1[0].idempotency_key.startswith(f"sess:{sid}:processor.zone_exited:")


def test_disappearance_does_not_emit_zone_exited() -> None:
    """Track was stably inside, then the camera/detector simply stops
    seeing it. We don't call ``absorb_object`` for missing tracks, so
    no state change → no event. This test simulates that pattern."""
    s = ZoneMembershipState(stream_session_id=uuid4(), membership_stability_frames=1)
    _absorb(s, "t0", zone_set={"entry-roi"}, roi_labels={"entry-roi"}, occurred_at=_now())
    assert s.stable_membership[("t0", "entry-roi")] is True
    # ``t0`` is missing from frames 1..20. Worker simply doesn't call
    # ``absorb_object`` for it. State should stay committed inside;
    # zero emissions on the missing-frame range.
    # (No-op iteration — we explicitly DON'T call absorb_object.)
    assert s.stable_membership[("t0", "entry-roi")] is True
    # When ``t0`` reappears still inside, no new event.
    revisit = _absorb(
        s,
        "t0",
        zone_set={"entry-roi"},
        roi_labels={"entry-roi"},
        occurred_at=_now() + timedelta(seconds=30),
    )
    assert revisit == []


def test_class_id_and_label_preserved_in_payload() -> None:
    """``class_id`` and ``class_label`` ride into the event when supplied
    so downstream consumers can filter on object class."""
    s = ZoneMembershipState(stream_session_id=uuid4(), membership_stability_frames=1)
    events = s.absorb_object(
        track_id="t0",
        zone_set={"entry-roi"},
        roi_labels={"entry-roi"},
        occurred_at=_now(),
        class_id=2,
        class_label="person",
    )
    body = events[0].to_payload()
    assert body["class_id"] == 2
    assert body["class_label"] == "person"


def test_optional_class_fields_omitted_when_none() -> None:
    """A producer without classifier metadata gets a clean payload —
    no ``"class_id": null`` noise."""
    s = ZoneMembershipState(stream_session_id=uuid4(), membership_stability_frames=1)
    events = s.absorb_object(
        track_id="t0",
        zone_set={"entry-roi"},
        roi_labels={"entry-roi"},
        occurred_at=_now(),
        class_id=None,
        class_label=None,
    )
    body = events[0].to_payload()
    assert "class_id" not in body
    assert "class_label" not in body


def test_debug_transition_field_is_correct() -> None:
    """``debug.transition`` distinguishes outside_to_inside (zone_entered)
    from inside_to_outside (zone_exited) at a glance."""
    s = ZoneMembershipState(stream_session_id=uuid4(), membership_stability_frames=1)
    enter = _absorb(s, "t0", zone_set={"entry-roi"}, roi_labels={"entry-roi"}, occurred_at=_now())
    exit_ev = _absorb(
        s,
        "t0",
        zone_set={"entry-roi"},
        roi_labels=set(),
        occurred_at=_now() + timedelta(seconds=1),
    )
    assert enter[0].debug["transition"] == "outside_to_inside"
    assert exit_ev[0].debug["transition"] == "inside_to_outside"
    # And: stability_frames_required survives.
    assert enter[0].debug["stability_frames_required"] == 1
    assert exit_ev[0].debug["stability_frames_required"] == 1


def test_payload_top_level_is_runtime_neutral() -> None:
    """No DeepStream / nvdsanalytics vocabulary at the top of the
    payload — provenance under ``source``, raw frame data + transition
    under ``debug``."""
    s = ZoneMembershipState(stream_session_id=uuid4(), membership_stability_frames=1)
    events = s.absorb_object(
        track_id="t0",
        zone_set={"entry-roi"},
        roi_labels={"entry-roi"},
        occurred_at=_now(),
        class_id=2,
        class_label="person",
        source={"runtime": "deepstream", "package_id": "x"},
        debug={"frame_num": 42, "geometry_source": "probe_default"},
    )
    body = events[0].to_payload()
    # Top-level keys: only neutral.
    for forbidden in (
        "runtime",
        "package_id",
        "metadata_source",
        "analytics_plugin",
        "tracker_algorithm",
        "frame_num",
        "geometry_source",
        "transition",
        "stability_frames_required",
    ):
        assert forbidden not in body, forbidden
    # Provenance under source.
    assert body["source"]["runtime"] == "deepstream"
    # Raw under debug.
    assert body["debug"]["frame_num"] == 42
    assert body["debug"]["geometry_source"] == "probe_default"
    assert body["debug"]["transition"] == "outside_to_inside"


def test_idempotency_key_builders_match_event_keys() -> None:
    """The helpers and the state machine must build identical keys —
    so a future server-side dedupe migration can re-derive them
    without coordinating with the runtime."""
    sid = uuid4()
    s = ZoneMembershipState(stream_session_id=sid, membership_stability_frames=1)
    enter = _absorb(s, "t0", zone_set={"entry-roi"}, roi_labels={"entry-roi"}, occurred_at=_now())
    exit_ev = _absorb(
        s,
        "t0",
        zone_set={"entry-roi"},
        roi_labels=set(),
        occurred_at=_now() + timedelta(seconds=1),
    )
    assert enter[0].idempotency_key == build_zone_entered_idempotency_key(
        stream_session_id=sid, zone_id="entry-roi", track_id="t0", sequence=1
    )
    assert exit_ev[0].idempotency_key == build_zone_exited_idempotency_key(
        stream_session_id=sid, zone_id="entry-roi", track_id="t0", sequence=1
    )
    # Distinct prefixes for the two transition types.
    assert enter[0].idempotency_key != exit_ev[0].idempotency_key


def test_pending_resets_when_candidate_returns_to_committed() -> None:
    """Inside committed; outside seen once; back inside — pending
    cleared, sequence-by-zone_exited never advances."""
    s = ZoneMembershipState(stream_session_id=uuid4(), membership_stability_frames=3)
    for off in range(3):
        _absorb(
            s,
            "t0",
            zone_set={"entry-roi"},
            roi_labels={"entry-roi"},
            occurred_at=_now() + timedelta(seconds=off),
        )
    # Two outside frames (below threshold=3).
    _absorb(
        s, "t0", zone_set={"entry-roi"}, roi_labels=set(), occurred_at=_now() + timedelta(seconds=3)
    )
    _absorb(
        s, "t0", zone_set={"entry-roi"}, roi_labels=set(), occurred_at=_now() + timedelta(seconds=4)
    )
    # Back inside — pending should clear.
    _absorb(
        s,
        "t0",
        zone_set={"entry-roi"},
        roi_labels={"entry-roi"},
        occurred_at=_now() + timedelta(seconds=5),
    )
    assert ("t0", "entry-roi") not in s.pending_membership
    # And the per-(track, zone, type) zone_exited sequence stays at 0.
    assert (
        s.sequence_by_track_zone_transition.get(("t0", "entry-roi", CANDIDATE_ZONE_EXITED_TYPE))
        is None
    )


# --- P5-Gap: disappearance-based inferred zone_exited -----------------


def _commit_inside_with_presence(
    s: ZoneMembershipState, track: str = "t0", zone: str = "entry-roi"
) -> None:
    """Commit ``track`` stably inside ``zone`` with >= the default
    min-presence (0.5s): one frame at t=0, another at t=0.6."""
    _absorb(s, track, zone_set={zone}, roi_labels={zone}, occurred_at=_now())
    _absorb(
        s, track, zone_set={zone}, roi_labels={zone},
        occurred_at=_now() + timedelta(seconds=0.6),
    )
    assert s.stable_membership[(track, zone)] is True


def test_detect_disappearances_infers_exit_after_grace() -> None:
    s = ZoneMembershipState(stream_session_id=uuid4(), membership_stability_frames=1)
    _commit_inside_with_presence(s)
    # Track absent; 2.4s since last seen (>= 2.0 grace) -> inferred exit.
    events = s.detect_disappearances(
        active_track_ids=set(), occurred_at=_now() + timedelta(seconds=3.0)
    )
    assert len(events) == 1
    ev = events[0]
    assert ev.type == CANDIDATE_ZONE_EXITED_TYPE
    assert ev.inferred is True
    assert ev.exit_reason == "track_missing_timeout"
    assert ev.missing_duration_seconds is not None and ev.missing_duration_seconds >= 2.0
    body = ev.to_payload()
    assert body["inferred"] is True
    assert body["exit_reason"] == "track_missing_timeout"
    assert "last_seen_at" in body and "missing_duration_seconds" in body
    # Now committed outside; a second tick does not re-emit.
    assert s.stable_membership[("t0", "entry-roi")] is False
    again = s.detect_disappearances(
        active_track_ids=set(), occurred_at=_now() + timedelta(seconds=6.0)
    )
    assert again == []


def test_detect_disappearances_no_exit_within_grace() -> None:
    s = ZoneMembershipState(stream_session_id=uuid4(), membership_stability_frames=1)
    _commit_inside_with_presence(s)
    # Only 1.0s missing (< 2.0 grace) -> nothing yet.
    events = s.detect_disappearances(
        active_track_ids=set(), occurred_at=_now() + timedelta(seconds=1.6)
    )
    assert events == []
    assert s.stable_membership[("t0", "entry-roi")] is True


def test_detect_disappearances_reappear_before_grace_no_exit() -> None:
    s = ZoneMembershipState(stream_session_id=uuid4(), membership_stability_frames=1)
    _commit_inside_with_presence(s)
    # Track present this tick (within grace) -> timer effectively resets;
    # absorb_object would also update last_seen. No inferred exit.
    events = s.detect_disappearances(
        active_track_ids={"t0"}, occurred_at=_now() + timedelta(seconds=3.0)
    )
    assert events == []
    assert s.stable_membership[("t0", "entry-roi")] is True


def test_detect_disappearances_requires_committed_inside() -> None:
    s = ZoneMembershipState(stream_session_id=uuid4(), membership_stability_frames=1)
    # Never observed inside -> no state -> no inferred exit.
    events = s.detect_disappearances(
        active_track_ids=set(), occurred_at=_now() + timedelta(seconds=10)
    )
    assert events == []


def test_detect_disappearances_single_frame_commit_still_emits() -> None:
    # A committed-inside track (even via a single stability=1 frame) is a
    # confirmed presence; on disappearance past grace it emits an inferred exit
    # (no min-presence suppression -> no dwell leak).
    s = ZoneMembershipState(stream_session_id=uuid4(), membership_stability_frames=1)
    _absorb(s, "t0", zone_set={"entry-roi"}, roi_labels={"entry-roi"}, occurred_at=_now())
    events = s.detect_disappearances(
        active_track_ids=set(), occurred_at=_now() + timedelta(seconds=3.0)
    )
    assert len(events) == 1 and events[0].inferred is True
    assert s.stable_membership[("t0", "entry-roi")] is False


def test_detect_disappearances_with_production_stability_frames() -> None:
    # Regression for the min-presence bug: with the production default of 5,
    # commit fires on the LAST inside frame. The inferred exit must still fire
    # on disappearance (it used to be silently dropped).
    s = ZoneMembershipState(stream_session_id=uuid4(), membership_stability_frames=5)
    for i in range(5):  # inside frames t=0..4 -> commits zone_entered at t=4
        _absorb(
            s, "t0", zone_set={"entry-roi"}, roi_labels={"entry-roi"},
            occurred_at=_now() + timedelta(seconds=i),
        )
    assert s.stable_membership[("t0", "entry-roi")] is True
    events = s.detect_disappearances(
        active_track_ids=set(), occurred_at=_now() + timedelta(seconds=7.0)
    )
    assert len(events) == 1
    assert events[0].inferred is True and events[0].type == CANDIDATE_ZONE_EXITED_TYPE


def test_detect_disappearances_committed_then_outside_present_then_gone() -> None:
    # Committed inside, then observed OUTSIDE-but-present for a few frames
    # (below the outside-commit threshold so it stays committed), then vanishes.
    # The inferred exit must still fire (last_seen stays at the last inside
    # frame; the missing window is measured from there).
    s = ZoneMembershipState(stream_session_id=uuid4(), membership_stability_frames=5)
    for i in range(5):
        _absorb(
            s, "t0", zone_set={"entry-roi"}, roi_labels={"entry-roi"},
            occurred_at=_now() + timedelta(seconds=i),
        )
    for i in range(5, 8):  # present but outside, below threshold -> stays committed
        _absorb(
            s, "t0", zone_set={"entry-roi"}, roi_labels=set(),
            occurred_at=_now() + timedelta(seconds=i),
        )
    assert s.stable_membership[("t0", "entry-roi")] is True
    events = s.detect_disappearances(
        active_track_ids=set(), occurred_at=_now() + timedelta(seconds=11.0)
    )
    assert len(events) == 1 and events[0].inferred is True


def test_detect_disappearances_dedup_then_real_exit_distinct() -> None:
    s = ZoneMembershipState(stream_session_id=uuid4(), membership_stability_frames=1)
    _commit_inside_with_presence(s)
    inferred = s.detect_disappearances(
        active_track_ids=set(), occurred_at=_now() + timedelta(seconds=3.0)
    )
    assert len(inferred) == 1 and inferred[0].sequence == 1
    # Track returns and is later observed exiting normally -> sequence 2.
    _commit_inside_with_presence(s)  # re-enter (enter seq bumps, exit unaffected)
    real_exit = _absorb(
        s, "t0", zone_set={"entry-roi"}, roi_labels=set(),
        occurred_at=_now() + timedelta(seconds=10),
    )
    assert len(real_exit) == 1
    assert real_exit[0].type == CANDIDATE_ZONE_EXITED_TYPE
    assert real_exit[0].sequence == 2  # distinct from the inferred exit (1)
    assert real_exit[0].inferred is False
    assert inferred[0].idempotency_key != real_exit[0].idempotency_key


def test_detect_disappearances_disabled_by_policy() -> None:
    off = LifecyclePolicy(infer_zone_exit_on_disappearance=False)
    s = ZoneMembershipState(
        stream_session_id=uuid4(), membership_stability_frames=1, lifecycle_policy=off
    )
    _commit_inside_with_presence(s)
    events = s.detect_disappearances(
        active_track_ids=set(), occurred_at=_now() + timedelta(seconds=30)
    )
    assert events == []
    assert s.stable_membership[("t0", "entry-roi")] is True  # untouched when off


def test_detect_disappearances_independent_per_track_and_zone() -> None:
    s = ZoneMembershipState(stream_session_id=uuid4(), membership_stability_frames=1)
    _commit_inside_with_presence(s, "t0", "zone-a")
    _commit_inside_with_presence(s, "t1", "zone-a")
    # Only t0 disappears; t1 still present.
    events = s.detect_disappearances(
        active_track_ids={"t1"}, occurred_at=_now() + timedelta(seconds=3.0)
    )
    assert len(events) == 1
    assert events[0].track_id == "t0"
    assert s.stable_membership[("t1", "zone-a")] is True
