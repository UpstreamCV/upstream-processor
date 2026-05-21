"""P4-E — ``DwellState`` adapter unit tests.

Pure / dependency-free. NO Docker / GPU / pyds / api-server / worker.
The state machine is small (per-(track, zone) active dwell + tick
clock) so tests cover every transition: validation, dwell_started
on zone_entered, duplicate-enter guard, dwell_updated cadence,
dwell_exceeded single-shot, session close on zone_exited, no
updates after exit, re-enter starts fresh dwell_session_id,
payload shape (top-level neutral, optional fields, rounded
duration), idempotency key format.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from app.observations.observation_adapter import (
    CANDIDATE_DWELL_EXCEEDED_TYPE,
    CANDIDATE_DWELL_STARTED_TYPE,
    CANDIDATE_DWELL_UPDATED_TYPE,
    CANDIDATE_ZONE_ENTERED_TYPE,
    CANDIDATE_ZONE_EXITED_TYPE,
    DwellEvent,
    DwellState,
    ZoneTransitionEvent,
    build_dwell_exceeded_idempotency_key,
    build_dwell_started_idempotency_key,
    build_dwell_updated_idempotency_key,
)

_T0 = datetime(2026, 5, 21, 12, 0, 0, tzinfo=UTC)


def _entered(
    *,
    track_id: str = "t0",
    zone_id: str = "entry-roi",
    sequence: int = 1,
    occurred_at: datetime = _T0,
    class_id: int | None = 2,
    class_label: str | None = "person",
    stream_session_id=None,
) -> ZoneTransitionEvent:
    sid = stream_session_id or uuid4()
    return ZoneTransitionEvent(
        type=CANDIDATE_ZONE_ENTERED_TYPE,
        stream_session_id=sid,
        occurred_at=occurred_at,
        zone_id=zone_id,
        zone_label=zone_id,
        track_id=track_id,
        sequence=sequence,
        idempotency_key=f"sess:{sid}:processor.zone_entered:{zone_id}:{track_id}:{sequence}",
        class_id=class_id,
        class_label=class_label,
    )


def _exited(
    *,
    track_id: str = "t0",
    zone_id: str = "entry-roi",
    sequence: int = 1,
    occurred_at: datetime = _T0 + timedelta(seconds=5),
    stream_session_id=None,
) -> ZoneTransitionEvent:
    sid = stream_session_id or uuid4()
    return ZoneTransitionEvent(
        type=CANDIDATE_ZONE_EXITED_TYPE,
        stream_session_id=sid,
        occurred_at=occurred_at,
        zone_id=zone_id,
        zone_label=zone_id,
        track_id=track_id,
        sequence=sequence,
        idempotency_key=f"sess:{sid}:processor.zone_exited:{zone_id}:{track_id}:{sequence}",
    )


# --- validation -------------------------------------------------------


def test_state_rejects_zero_or_negative_update_interval() -> None:
    with pytest.raises(ValueError, match="update_interval_seconds must be > 0"):
        DwellState(stream_session_id=uuid4(), update_interval_seconds=0)
    with pytest.raises(ValueError, match="update_interval_seconds must be > 0"):
        DwellState(stream_session_id=uuid4(), update_interval_seconds=-1.5)


def test_state_rejects_zero_or_negative_exceeded_threshold() -> None:
    with pytest.raises(ValueError, match="exceeded_threshold_seconds must be > 0"):
        DwellState(stream_session_id=uuid4(), exceeded_threshold_seconds=0)
    with pytest.raises(ValueError, match="exceeded_threshold_seconds must be > 0"):
        DwellState(stream_session_id=uuid4(), exceeded_threshold_seconds=-30)


# --- dwell_started semantics ------------------------------------------


def test_zone_entered_emits_dwell_started() -> None:
    sid = uuid4()
    s = DwellState(stream_session_id=sid)
    events = s.absorb_zone_event(_entered(stream_session_id=sid))
    assert len(events) == 1
    e = events[0]
    assert e.type == CANDIDATE_DWELL_STARTED_TYPE
    assert e.zone_id == "entry-roi"
    assert e.track_id == "t0"
    assert e.dwell_session_id == "entry-roi:t0:1"
    assert e.duration_seconds == 0.0
    # Idempotency key matches the builder.
    expected_key = build_dwell_started_idempotency_key(
        stream_session_id=sid,
        zone_id="entry-roi",
        track_id="t0",
        dwell_session_id="entry-roi:t0:1",
    )
    assert e.idempotency_key == expected_key


def test_duplicate_zone_entered_does_not_emit_second_started() -> None:
    """Defensive guard: ZoneMembershipState should never emit two
    zone_entered events back-to-back for the same (track, zone), but
    if a bug ever did, DwellState must not duplicate dwell_started."""
    s = DwellState(stream_session_id=uuid4())
    s.absorb_zone_event(_entered())
    again = s.absorb_zone_event(_entered(sequence=1))
    assert again == []
    assert s.active_session_count() == 1


# --- dwell_updated semantics ------------------------------------------


def test_dwell_updated_does_not_emit_before_interval() -> None:
    s = DwellState(
        stream_session_id=uuid4(),
        update_interval_seconds=30.0,
        exceeded_threshold_seconds=60.0,
    )
    s.absorb_zone_event(_entered(occurred_at=_T0))
    # Tick at +10s: well below the 30s interval.
    events = s.tick(occurred_at=_T0 + timedelta(seconds=10))
    assert events == []


def test_dwell_updated_emits_at_interval() -> None:
    sid = uuid4()
    s = DwellState(
        stream_session_id=sid,
        update_interval_seconds=30.0,
        exceeded_threshold_seconds=600.0,
    )
    s.absorb_zone_event(_entered(occurred_at=_T0, stream_session_id=sid))
    # Tick at +30s exactly: should fire one dwell_updated.
    events = s.tick(occurred_at=_T0 + timedelta(seconds=30))
    assert len(events) == 1
    e = events[0]
    assert e.type == CANDIDATE_DWELL_UPDATED_TYPE
    assert e.update_sequence == 1
    assert e.duration_seconds == 30.0
    assert e.idempotency_key == build_dwell_updated_idempotency_key(
        stream_session_id=sid,
        zone_id="entry-roi",
        track_id="t0",
        dwell_session_id="entry-roi:t0:1",
        update_sequence=1,
    )


def test_dwell_updated_emits_repeatedly_on_subsequent_intervals() -> None:
    s = DwellState(
        stream_session_id=uuid4(),
        update_interval_seconds=10.0,
        exceeded_threshold_seconds=600.0,
    )
    s.absorb_zone_event(_entered(occurred_at=_T0))
    # Tick at +10, +20, +30 — three update events, sequence 1, 2, 3.
    seqs: list[int] = []
    for offset in (10, 20, 30):
        events = s.tick(occurred_at=_T0 + timedelta(seconds=offset))
        assert len(events) == 1, offset
        assert events[0].type == CANDIDATE_DWELL_UPDATED_TYPE
        assert events[0].update_sequence is not None
        seqs.append(events[0].update_sequence)
    assert seqs == [1, 2, 3]


def test_dwell_updated_skips_when_interval_not_yet_elapsed_since_last() -> None:
    """update_sequence advances per emit, not per tick — a tick at
    +35s emits sequence=1; the next tick at +50s should NOT emit
    (only 15s since the last emit; threshold is 30s)."""
    s = DwellState(
        stream_session_id=uuid4(),
        update_interval_seconds=30.0,
        exceeded_threshold_seconds=600.0,
    )
    s.absorb_zone_event(_entered(occurred_at=_T0))
    first = s.tick(occurred_at=_T0 + timedelta(seconds=35))
    assert len(first) == 1
    second = s.tick(occurred_at=_T0 + timedelta(seconds=50))
    assert second == []


# --- dwell_exceeded semantics -----------------------------------------


def test_dwell_exceeded_does_not_emit_before_threshold() -> None:
    s = DwellState(
        stream_session_id=uuid4(),
        update_interval_seconds=600.0,
        exceeded_threshold_seconds=60.0,
    )
    s.absorb_zone_event(_entered(occurred_at=_T0))
    events = s.tick(occurred_at=_T0 + timedelta(seconds=30))
    assert events == []


def test_dwell_exceeded_emits_once_at_threshold() -> None:
    sid = uuid4()
    s = DwellState(
        stream_session_id=sid,
        update_interval_seconds=600.0,
        exceeded_threshold_seconds=60.0,
    )
    s.absorb_zone_event(_entered(occurred_at=_T0, stream_session_id=sid))
    events = s.tick(occurred_at=_T0 + timedelta(seconds=60))
    assert len(events) == 1
    e = events[0]
    assert e.type == CANDIDATE_DWELL_EXCEEDED_TYPE
    assert e.threshold_seconds == 60.0
    assert e.duration_seconds == 60.0
    assert e.idempotency_key == build_dwell_exceeded_idempotency_key(
        stream_session_id=sid,
        zone_id="entry-roi",
        track_id="t0",
        dwell_session_id="entry-roi:t0:1",
        threshold_seconds=60.0,
    )


def test_dwell_exceeded_does_not_duplicate_after_threshold() -> None:
    s = DwellState(
        stream_session_id=uuid4(),
        update_interval_seconds=600.0,
        exceeded_threshold_seconds=60.0,
    )
    s.absorb_zone_event(_entered(occurred_at=_T0))
    s.tick(occurred_at=_T0 + timedelta(seconds=60))
    later = s.tick(occurred_at=_T0 + timedelta(seconds=120))
    assert later == []


# --- zone_exited closes the session -----------------------------------


def test_zone_exited_closes_dwell_session() -> None:
    s = DwellState(stream_session_id=uuid4())
    s.absorb_zone_event(_entered(occurred_at=_T0))
    assert s.active_session_count() == 1
    s.absorb_zone_event(_exited(occurred_at=_T0 + timedelta(seconds=5)))
    assert s.active_session_count() == 0


def test_no_dwell_updated_after_zone_exit() -> None:
    s = DwellState(
        stream_session_id=uuid4(),
        update_interval_seconds=10.0,
        exceeded_threshold_seconds=600.0,
    )
    s.absorb_zone_event(_entered(occurred_at=_T0))
    s.absorb_zone_event(_exited(occurred_at=_T0 + timedelta(seconds=5)))
    # Tick well past the update interval — no events because session
    # was closed.
    events = s.tick(occurred_at=_T0 + timedelta(seconds=60))
    assert events == []


def test_no_dwell_exceeded_after_zone_exit() -> None:
    s = DwellState(
        stream_session_id=uuid4(),
        update_interval_seconds=600.0,
        exceeded_threshold_seconds=10.0,
    )
    s.absorb_zone_event(_entered(occurred_at=_T0))
    s.absorb_zone_event(_exited(occurred_at=_T0 + timedelta(seconds=5)))
    events = s.tick(occurred_at=_T0 + timedelta(seconds=60))
    assert events == []


def test_re_enter_after_exit_starts_new_dwell_session_id() -> None:
    """Per-(track, zone, enter-sequence) dwell_session_id. After exit,
    re-entering with zone sequence=2 produces a new dwell_session_id
    that includes ``:2`` instead of ``:1`` — the exceeded threshold
    fires again for the new session."""
    s = DwellState(
        stream_session_id=uuid4(),
        update_interval_seconds=600.0,
        exceeded_threshold_seconds=30.0,
    )
    # Session 1: enter, hit threshold, exit.
    s.absorb_zone_event(_entered(occurred_at=_T0, sequence=1))
    s.tick(occurred_at=_T0 + timedelta(seconds=30))
    s.absorb_zone_event(_exited(occurred_at=_T0 + timedelta(seconds=40), sequence=1))
    # Session 2: re-enter, dwell_session_id has sequence=2, fresh
    # exceeded path.
    start = s.absorb_zone_event(_entered(occurred_at=_T0 + timedelta(seconds=50), sequence=2))
    assert start[0].dwell_session_id == "entry-roi:t0:2"
    # New threshold fire because the new session has a fresh
    # exceeded_emitted=False.
    events = s.tick(occurred_at=_T0 + timedelta(seconds=80))
    assert any(e.type == CANDIDATE_DWELL_EXCEEDED_TYPE for e in events)


# --- multi-track / multi-zone independence ----------------------------


def test_independent_sessions_per_track_zone() -> None:
    s = DwellState(stream_session_id=uuid4())
    s.absorb_zone_event(_entered(track_id="t0", zone_id="entry-roi"))
    s.absorb_zone_event(_entered(track_id="t1", zone_id="entry-roi"))
    s.absorb_zone_event(_entered(track_id="t0", zone_id="other-roi"))
    assert s.active_session_count() == 3
    # Closing one session leaves the others.
    s.absorb_zone_event(_exited(track_id="t0", zone_id="entry-roi"))
    assert s.active_session_count() == 2


# --- payload + idempotency invariants ---------------------------------


def test_class_id_and_label_preserved_in_payload() -> None:
    s = DwellState(stream_session_id=uuid4())
    events = s.absorb_zone_event(_entered(class_id=2, class_label="person", occurred_at=_T0))
    body = events[0].to_payload()
    assert body["class_id"] == 2
    assert body["class_label"] == "person"


def test_optional_class_fields_omitted_when_none() -> None:
    s = DwellState(stream_session_id=uuid4())
    events = s.absorb_zone_event(_entered(class_id=None, class_label=None, occurred_at=_T0))
    body = events[0].to_payload()
    assert "class_id" not in body
    assert "class_label" not in body


def test_duration_seconds_rounded_to_two_decimals() -> None:
    s = DwellState(
        stream_session_id=uuid4(),
        update_interval_seconds=1.0,
        exceeded_threshold_seconds=600.0,
    )
    s.absorb_zone_event(_entered(occurred_at=_T0))
    # Tick at +1.234567s — duration should round to 1.23.
    events = s.tick(occurred_at=_T0 + timedelta(seconds=1.234567))
    assert len(events) == 1
    assert events[0].to_payload()["duration_seconds"] == 1.23


def test_idempotency_keys_dont_include_frame_num() -> None:
    """Sequences and dwell_session_id are the only sources of
    uniqueness in the key — frame_num must NOT appear (it lives in
    payload.debug only)."""
    sid = uuid4()
    s = DwellState(
        stream_session_id=sid,
        update_interval_seconds=10.0,
        exceeded_threshold_seconds=30.0,
    )
    started = s.absorb_zone_event(_entered(occurred_at=_T0, stream_session_id=sid))
    updated = s.tick(occurred_at=_T0 + timedelta(seconds=10))
    exceeded = s.tick(occurred_at=_T0 + timedelta(seconds=30))
    for batch in (started, updated, exceeded):
        for e in batch:
            assert "frame_num" not in e.idempotency_key, e.idempotency_key
            # Keys end with the session-specific suffix, never a
            # frame number.
            assert e.idempotency_key.startswith(f"sess:{sid}:")


def test_payload_top_level_is_runtime_neutral() -> None:
    """No DeepStream / nvdsanalytics vocabulary at the top level —
    provenance under ``source``, raw frame data + transition under
    ``debug``."""
    s = DwellState(stream_session_id=uuid4(), update_interval_seconds=10.0)
    events = s.absorb_zone_event(_entered(occurred_at=_T0))
    body = events[0].to_payload()
    for forbidden in (
        "runtime",
        "package_id",
        "metadata_source",
        "analytics_plugin",
        "tracker_algorithm",
        "frame_num",
        "geometry_source",
        "trigger",
    ):
        assert forbidden not in body, forbidden
    # source / debug carry the DeepStream provenance.
    assert "source" in body
    assert "debug" in body
    # debug.trigger is set by the adapter.
    assert body["debug"]["trigger"] == "zone_entered"


def test_dwell_updated_does_not_carry_threshold_field() -> None:
    """``threshold_seconds`` is an exceeded-only field. A dwell_updated
    payload must NOT include it (and vice versa for update_sequence)."""
    s = DwellState(
        stream_session_id=uuid4(),
        update_interval_seconds=10.0,
        exceeded_threshold_seconds=600.0,
    )
    s.absorb_zone_event(_entered(occurred_at=_T0))
    events = s.tick(occurred_at=_T0 + timedelta(seconds=10))
    body = events[0].to_payload()
    assert "threshold_seconds" not in body
    assert body["update_sequence"] == 1


def test_dwell_exceeded_does_not_carry_update_sequence_field() -> None:
    s = DwellState(
        stream_session_id=uuid4(),
        update_interval_seconds=600.0,
        exceeded_threshold_seconds=10.0,
    )
    s.absorb_zone_event(_entered(occurred_at=_T0))
    events = s.tick(occurred_at=_T0 + timedelta(seconds=10))
    body = events[0].to_payload()
    assert "update_sequence" not in body
    assert body["threshold_seconds"] == 10.0


def test_dwell_started_carries_neither_optional_field() -> None:
    s = DwellState(stream_session_id=uuid4())
    events = s.absorb_zone_event(_entered(occurred_at=_T0))
    body = events[0].to_payload()
    assert "update_sequence" not in body
    assert "threshold_seconds" not in body
    assert body["duration_seconds"] == 0.0


# --- DwellEvent dataclass invariants ----------------------------------


def test_dwell_event_type_field_matches_obs_type() -> None:
    """to_payload doesn't include the ``type`` (lives at the
    ObservationPayload top-level), but the DwellEvent itself must
    carry it so the worker can pick the right writer obs_type."""
    s = DwellState(stream_session_id=uuid4(), update_interval_seconds=5.0)
    started = s.absorb_zone_event(_entered(occurred_at=_T0))[0]
    updated = s.tick(occurred_at=_T0 + timedelta(seconds=5))[0]
    assert isinstance(started, DwellEvent)
    assert started.type == CANDIDATE_DWELL_STARTED_TYPE
    assert updated.type == CANDIDATE_DWELL_UPDATED_TYPE
