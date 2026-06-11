"""P4-A — runtime-neutral observation adapter contract (probe only).

This module is **contract-level**. It defines:

- the two first observation payload shapes the P4 milestone will emit
  (``processor.line_crossing`` and ``processor.roi_occupancy``),
- the deterministic idempotency keys that let the worker dedupe
  duplicate events even though the deployed api-server does NOT have
  an ``idempotency_key`` column yet (verified at
  ``upstream-api-server`` commit ``7a6057a`` /
  ``supabase/migrations/20260518140000_processor_status_observations.sql``),
- the per-session adapter state (``LineCrossingState`` /
  ``RoiOccupancyState``) the worker keeps between metadata frames so a
  cumulative counter delta turns into N discrete events and an ROI
  count snapshot only emits on change or interval.

It deliberately does **NOT** post anything yet. P4-B (next milestone)
will wire ``LineCrossingAdapter.absorb_frame()`` into the analytics
worker's consumer loop and emit through ``ObservationWriter`` once the
api-server's ``ProcessorObservationType`` literal gains the
``processor.*`` types (see P4-A report for the exact one-string change).

Why a probe rather than full implementation:

- ``schemas.ObservationType`` is a ``Literal`` restricted to the three
  ``processor.mock_*`` types both client- and server-side. Until the
  server's whitelist widens, a write would 422; we want to lock in the
  payload shape + idempotency story first so the api-server PR is
  one strict diff, not a moving target.
- The pad probe runs in the GLib thread and the worker reads its
  stdout — neither has any free-text "did I already emit this?" budget.
  The deterministic idempotency key + bounded last-cumulative state
  make the dedupe O(1) and trivially testable.

Everything in this module is pure (no I/O, no clock, no API). Callers
pass ``now`` / writer references in. That keeps the adapter unit-
testable without Docker / pyds / network.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any
from uuid import UUID

# Candidate observation type strings the P4 adapter will emit. Defined
# here (NOT in ``app/observations/schemas.py``) until the server-side
# literal is widened — keeping them out of the strict-validated payload
# until then prevents accidental production POSTs. P4-B will move these
# into ``schemas.ObservationType`` in lock-step with the api-server PR.
CANDIDATE_LINE_CROSSING_TYPE = "processor.line_crossing"
# Side task (per-object line crossing): per-OBJECT crossing evidence from
# nvdsanalytics' per-object ``lcStatus``, distinct from the aggregate
# cumulative-counter type above. Both flow side by side: the aggregate
# type keeps its count semantics; this type carries track/class/confidence
# attribution for workflow conditions.
CANDIDATE_LINE_CROSSING_OBJECT_TYPE = "processor.line_crossing_object"
CANDIDATE_ROI_OCCUPANCY_TYPE = "processor.roi_occupancy"
# P4-D zone transition types. The two are kept as separate constants
# because their event payloads share zone_id / track_id / sequence but
# the obs_type Literal is distinct (the api-server checks each on the
# wire) and the idempotency-key namespace is per-transition-type.
CANDIDATE_ZONE_ENTERED_TYPE = "processor.zone_entered"
CANDIDATE_ZONE_EXITED_TYPE = "processor.zone_exited"
# P4-E dwell observation types. Layered on top of P4-D's
# ZoneMembershipState — DwellState ingests zone_entered / zone_exited
# events and emits the three dwell observations.
CANDIDATE_DWELL_STARTED_TYPE = "processor.dwell_started"
CANDIDATE_DWELL_UPDATED_TYPE = "processor.dwell_updated"
CANDIDATE_DWELL_EXCEEDED_TYPE = "processor.dwell_exceeded"

CANDIDATE_OBSERVATION_TYPES: tuple[str, ...] = (
    CANDIDATE_LINE_CROSSING_TYPE,
    CANDIDATE_LINE_CROSSING_OBJECT_TYPE,
    CANDIDATE_ROI_OCCUPANCY_TYPE,
    CANDIDATE_ZONE_ENTERED_TYPE,
    CANDIDATE_ZONE_EXITED_TYPE,
    CANDIDATE_DWELL_STARTED_TYPE,
    CANDIDATE_DWELL_UPDATED_TYPE,
    CANDIDATE_DWELL_EXCEEDED_TYPE,
)


# --- line-crossing -----------------------------------------------------


@dataclass(frozen=True)
class LineCrossingEvent:
    """One ``processor.line_crossing`` event, ready to hand to
    ``ObservationWriter.emit(...)`` once the api-server accepts the type.

    Top-level fields are runtime-neutral (no ``deepstream`` /
    ``nvdsanalytics`` / pyds vocabulary). DeepStream-specific details
    live under ``source`` (stable provenance) and ``debug`` (raw counters
    + geometry source). The intent is that a future Triton / OpenCV
    runtime could emit the SAME observation type with a different
    ``source.runtime`` and the consumer side wouldn't care."""

    type: str
    stream_session_id: UUID
    occurred_at: datetime
    line_id: str
    line_label: str
    count_delta: int
    cumulative_count: int
    idempotency_key: str
    # Optional fields (None if the producer didn't have them).
    track_id: str | None = None
    class_id: int | None = None
    class_label: str | None = None
    direction: str | None = None
    source: dict[str, Any] = field(default_factory=dict)
    debug: dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        """Body for the ``payload`` field on ``ObservationPayload``. The
        outer ``ObservationPayload`` already carries
        ``stream_session_id``, ``type``, ``observed_at``, ``processor_id``,
        and ``package_id`` — this is the runtime-specific body. The
        ``idempotency_key`` lives here (in ``payload``) until the server
        gains a first-class column; client-side state in
        ``LineCrossingState`` does the actual dedupe today."""
        body: dict[str, Any] = {
            "line_id": self.line_id,
            "line_label": self.line_label,
            "count_delta": int(self.count_delta),
            "cumulative_count": int(self.cumulative_count),
            "idempotency_key": self.idempotency_key,
            "source": dict(self.source),
            "debug": dict(self.debug),
        }
        if self.track_id is not None:
            body["track_id"] = self.track_id
        if self.class_id is not None:
            body["class_id"] = int(self.class_id)
        if self.class_label is not None:
            body["class_label"] = self.class_label
        if self.direction is not None:
            body["direction"] = self.direction
        return body


def build_line_crossing_idempotency_key(
    *,
    stream_session_id: UUID,
    line_id: str,
    cumulative_count: int,
) -> str:
    """Deterministic dedupe key for a single line-crossing event.

    Format: ``sess:<session_id>:processor.line_crossing:<line_id>:<cum>``.
    Cumulative count is per-line / per-stream — two cameras crossing the
    same line label produce distinct keys because ``session_id`` differs.

    ``track_id`` is intentionally NOT in the key: nvdsanalytics' frame-
    level ``objLCCumCnt`` is a per-LINE aggregate, not a per-track one.
    The per-object ``lcStatus`` IS per-track but uses string labels with
    no count; we treat the frame-level cumulative as the source of truth
    for ``count_delta`` and attach ``track_id`` as optional context.

    Keep this function pure / dependency-free so it can be re-derived
    server-side if/when the api-server adds an ``idempotency_key``
    column without coordinating with the processor."""
    return (
        f"sess:{stream_session_id}:{CANDIDATE_LINE_CROSSING_TYPE}:{line_id}:{int(cumulative_count)}"
    )


@dataclass
class LineCrossingState:
    """Per-stream-per-worker adapter state for the line-crossing path.

    ``last_cumulative_by_line`` is the only required state for dedupe:
    when ``nvdsanalytics``' frame-level ``objLCCumCnt[<label>]`` goes from
    ``N`` to ``N + k`` between consecutive frames, we emit ``k`` events
    with cumulative counts ``N + 1 .. N + k`` (each one a distinct
    idempotency key). When the count is unchanged across frames, nothing
    is emitted. Resets when the supervisor restarts the worker (each
    bounded probe cycle gets a fresh state) — this is acceptable for
    P4-B because the server is currently non-deduping anyway, and the
    cumulative counter itself resets on container restart.

    ``recently_emitted_keys`` is a tiny bounded set as a belt-and-
    suspenders against the (theoretical) case where the producer rewinds
    the cumulative counter inside a single state. Bounded so an attacker
    / runaway producer can never grow it without bound."""

    stream_session_id: UUID
    last_cumulative_by_line: dict[str, int] = field(default_factory=dict)
    recently_emitted_keys: set[str] = field(default_factory=set)
    recently_emitted_keys_max: int = 256

    def absorb_cumulative(
        self,
        *,
        cumulative_by_line: dict[str, int],
        occurred_at: datetime,
        track_id: str | None = None,
        class_id: int | None = None,
        class_label: str | None = None,
        direction: str | None = None,
        source: dict[str, Any] | None = None,
        debug: dict[str, Any] | None = None,
    ) -> list[LineCrossingEvent]:
        """Compute the discrete events implied by a frame's cumulative
        counter snapshot. Returns ``[]`` if nothing changed. The caller
        is responsible for handing each event to the writer (and for
        passing ``occurred_at`` from the frame's ingest clock — the
        adapter does NOT call ``now()``).

        ``cumulative_by_line`` is exactly the dict
        ``frame.source["analytics"]["line_crossing_cumulative"]``
        landed in P3.5 — same shape, no translation needed."""
        events: list[LineCrossingEvent] = []
        for line_id, cum in cumulative_by_line.items():
            prev = self.last_cumulative_by_line.get(line_id, 0)
            if cum <= prev:
                # No new crossing — counter unchanged (or, defensively,
                # appears to have rewound). Do not emit, do not lower
                # the watermark.
                continue
            delta = cum - prev
            # Emit one event per increment so each crossing has a
            # distinct idempotency key. Most of the time delta == 1.
            for step in range(1, delta + 1):
                cumulative = prev + step
                key = build_line_crossing_idempotency_key(
                    stream_session_id=self.stream_session_id,
                    line_id=line_id,
                    cumulative_count=cumulative,
                )
                if key in self.recently_emitted_keys:
                    continue  # belt-and-suspenders, see class docstring.
                events.append(
                    LineCrossingEvent(
                        type=CANDIDATE_LINE_CROSSING_TYPE,
                        stream_session_id=self.stream_session_id,
                        occurred_at=occurred_at,
                        line_id=line_id,
                        line_label=line_id,
                        count_delta=1,
                        cumulative_count=cumulative,
                        idempotency_key=key,
                        track_id=track_id,
                        class_id=class_id,
                        class_label=class_label,
                        direction=direction,
                        source=dict(source or {}),
                        debug=dict(debug or {}),
                    )
                )
                self._remember(key)
            self.last_cumulative_by_line[line_id] = cum
        return events

    def _remember(self, key: str) -> None:
        if len(self.recently_emitted_keys) >= self.recently_emitted_keys_max:
            # Drop oldest — tiny set, O(1) eviction by re-creating. The
            # cumulative_by_line dict is the load-bearing dedupe path;
            # this set only catches a rewind/replay, so dropping the
            # oldest entries cannot cause a false positive in practice.
            self.recently_emitted_keys = set(
                list(self.recently_emitted_keys)[-(self.recently_emitted_keys_max - 1) :]
            )
        self.recently_emitted_keys.add(key)


# --- per-object line crossing (side task) -------------------------------


@dataclass(frozen=True)
class LineCrossingObjectEvent:
    """One ``processor.line_crossing_object`` event — per-OBJECT crossing
    evidence, distinct from the aggregate cumulative-counter
    ``processor.line_crossing`` type.

    ``track_id`` is REQUIRED: this type exists to attribute a crossing to
    a tracked object; untracked objects are skipped by the state (their
    crossings are still counted by the aggregate type). ``event_mode`` is
    stamped into the payload so consumers (the api-server rule matcher)
    can partition per-object vs aggregate events without relying on the
    type string alone.

    ``analytics_direction_label`` is the raw nvdsanalytics direction-RULE
    name (``dirStatus``) when present — deliberately NOT named
    ``direction`` because it is not a geometry-relative a_to_b/b_to_a
    crossing direction. The generated analytics config doesn't define
    direction rules today, so in production this is absent."""

    type: str  # CANDIDATE_LINE_CROSSING_OBJECT_TYPE
    stream_session_id: UUID
    occurred_at: datetime
    line_id: str
    line_label: str
    track_id: str
    sequence: int
    idempotency_key: str
    class_id: int | None = None
    class_label: str | None = None
    confidence: float | None = None
    analytics_direction_label: str | None = None
    source: dict[str, Any] = field(default_factory=dict)
    debug: dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        body: dict[str, Any] = {
            "line_id": self.line_id,
            "line_label": self.line_label,
            "track_id": self.track_id,
            "event_mode": "object_crossing",
            "sequence": int(self.sequence),
            "idempotency_key": self.idempotency_key,
            "source": dict(self.source),
            "debug": dict(self.debug),
        }
        if self.class_id is not None:
            body["class_id"] = int(self.class_id)
        if self.class_label is not None:
            body["class_label"] = self.class_label
        if self.confidence is not None:
            body["confidence"] = round(float(self.confidence), 4)
        if self.analytics_direction_label is not None:
            body["analytics_direction_label"] = self.analytics_direction_label
        return body


def build_line_crossing_object_idempotency_key(
    *,
    stream_session_id: UUID,
    line_id: str,
    track_id: str,
    sequence: int,
) -> str:
    """``sess:<sid>:processor.line_crossing_object:<line>:<track>:<seq>``.

    ``sequence`` is the per-(track, line) crossing counter from
    ``LineCrossingObjectState`` — a re-crossing by the same track gets
    sequence 2 while the first crossing stays 1, so keys never collide.
    Pure / dependency-free so the server can re-derive it later."""
    return (
        f"sess:{stream_session_id}:{CANDIDATE_LINE_CROSSING_OBJECT_TYPE}:"
        f"{line_id}:{track_id}:{int(sequence)}"
    )


@dataclass
class LineCrossingObjectState:
    """Per-stream-per-worker dedupe state for per-object line crossings.

    Rising-edge semantics: nvdsanalytics sets a per-object ``lcStatus``
    label when the object crosses that line. We emit exactly once when a
    (track, line) label transitions absent → present, and we do NOT
    re-emit while the label stays present on consecutive sightings
    (defensive — lcStatus is typically momentary, but some producer
    versions hold it). The label clearing (object sighted WITHOUT the
    label) re-arms the pair, so a genuine re-crossing emits again with a
    bumped sequence.

    Known limitation (documented, accepted): if a track disappears while
    a label is active and reappears later already carrying the same
    label, that sighting does not re-emit — we cannot distinguish a
    re-crossing from a held label across the gap. The aggregate
    cumulative counter still counts such crossings.

    Memory bound: state is keyed per (track, line) and pruned when the
    map exceeds ``max_entries`` — entries not seen for
    ``stale_after_seconds`` are dropped (frame-time based, not
    wall-clock). A pruned-then-reappearing pair behaves like a fresh
    track (one emit on its next rising edge); sequences restart, but
    idempotency keys stay unique in practice because pruning only
    happens after ``stale_after_seconds`` of absence."""

    stream_session_id: UUID
    max_entries: int = 4096
    stale_after_seconds: float = 600.0
    # (track_id, line_id) -> label active on the most recent sighting.
    active_by_track_line: dict[tuple[str, str], bool] = field(default_factory=dict)
    # (track_id, line_id) -> last frame-time the pair was updated.
    last_seen_by_track_line: dict[tuple[str, str], datetime] = field(default_factory=dict)
    # (track_id, line_id) -> crossings emitted (idempotency sequence).
    sequence_by_track_line: dict[tuple[str, str], int] = field(default_factory=dict)

    def absorb_object(
        self,
        *,
        track_id: str,
        line_labels: set[str] | frozenset[str],
        known_lines: set[str] | frozenset[str],
        occurred_at: datetime,
        class_id: int | None = None,
        class_label: str | None = None,
        confidence: float | None = None,
        analytics_direction_label: str | None = None,
        source: dict[str, Any] | None = None,
        debug: dict[str, Any] | None = None,
    ) -> list[LineCrossingObjectEvent]:
        """Update crossing state for one tracked object's sighting.

        ``line_labels`` is the object's ``lcStatus`` set for this frame;
        ``known_lines`` is the frame's full line universe (typically the
        keys of ``analytics.line_crossing_cumulative``, falling back to
        ``line_labels`` itself) — pairs in ``known_lines`` but not in
        ``line_labels`` are cleared (re-armed). Caller is responsible for
        filtering out objects without a ``tracker_id``."""
        events: list[LineCrossingObjectEvent] = []
        for line_id in known_lines | set(line_labels):
            key = (track_id, line_id)
            crossing_now = line_id in line_labels
            was_active = self.active_by_track_line.get(key, False)
            self.active_by_track_line[key] = crossing_now
            self.last_seen_by_track_line[key] = occurred_at
            if not crossing_now or was_active:
                continue
            seq = self.sequence_by_track_line.get(key, 0) + 1
            self.sequence_by_track_line[key] = seq
            events.append(
                LineCrossingObjectEvent(
                    type=CANDIDATE_LINE_CROSSING_OBJECT_TYPE,
                    stream_session_id=self.stream_session_id,
                    occurred_at=occurred_at,
                    line_id=line_id,
                    line_label=line_id,
                    track_id=track_id,
                    sequence=seq,
                    idempotency_key=build_line_crossing_object_idempotency_key(
                        stream_session_id=self.stream_session_id,
                        line_id=line_id,
                        track_id=track_id,
                        sequence=seq,
                    ),
                    class_id=class_id,
                    class_label=class_label,
                    confidence=confidence,
                    analytics_direction_label=analytics_direction_label,
                    source=dict(source or {}),
                    debug=dict(debug or {}),
                )
            )
        self._prune(occurred_at)
        return events

    def _prune(self, occurred_at: datetime) -> None:
        if len(self.active_by_track_line) <= self.max_entries:
            return
        stale_before = occurred_at - timedelta(seconds=self.stale_after_seconds)
        for key, last_seen in list(self.last_seen_by_track_line.items()):
            if last_seen < stale_before:
                self.active_by_track_line.pop(key, None)
                self.last_seen_by_track_line.pop(key, None)
                self.sequence_by_track_line.pop(key, None)


# --- ROI occupancy -----------------------------------------------------


@dataclass(frozen=True)
class RoiOccupancyEvent:
    """One ``processor.roi_occupancy`` event. Unlike line crossing this is
    a *snapshot* of an ROI's count, not a delta — emitting one per frame
    floods the api-server, so the adapter only emits on change OR after
    ``min_interval_seconds``. The idempotency key combines a stable
    sequence number with the session + ROI."""

    type: str
    stream_session_id: UUID
    occurred_at: datetime
    roi_id: str
    roi_label: str
    total_count: int
    class_counts: dict[str, int]
    sequence: int
    idempotency_key: str
    source: dict[str, Any] = field(default_factory=dict)
    debug: dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        return {
            "roi_id": self.roi_id,
            "roi_label": self.roi_label,
            "total_count": int(self.total_count),
            "class_counts": dict(self.class_counts),
            "sequence": int(self.sequence),
            "idempotency_key": self.idempotency_key,
            "source": dict(self.source),
            "debug": dict(self.debug),
        }


def build_roi_occupancy_idempotency_key(
    *,
    stream_session_id: UUID,
    roi_id: str,
    sequence: int,
) -> str:
    """Deterministic ROI occupancy key. ``sequence`` is the adapter's
    per-ROI emit counter (one per change/interval), NOT the raw frame
    number — using frame number would generate too many duplicates if
    the producer ever re-emits the same count and we'd lose the dedupe.
    Sequence is monotonically increasing within an adapter state."""
    return f"sess:{stream_session_id}:{CANDIDATE_ROI_OCCUPANCY_TYPE}:{roi_id}:{int(sequence)}"


@dataclass
class RoiOccupancyState:
    """Per-stream-per-worker ROI occupancy adapter state.

    Three emit conditions:

    1. **First-seen** — the first observation of an ROI always emits
       (so consumers learn the ROI exists). ``debug.emit_reason="first"``.
    2. **Change (stable)** — total count for an ROI differs from the
       last emitted count AND the new count has been seen on at least
       ``change_stability_frames`` consecutive frames. ``debug.emit_reason="change"``.
       Single-frame flaps below the threshold do NOT emit (P4-C cleanup).
    3. **Interval heartbeat** — at least ``min_interval_seconds`` have
       passed since the last emit. ``debug.emit_reason="interval"``.

    Important: if the interval elapses while a candidate count is
    pending (i.e., the count differs from last-emitted but hasn't yet
    been seen N times), the heartbeat reports the **last stable
    count**, NOT the unstable candidate. The pending state survives
    the heartbeat. Choice rationale: heartbeats are supposed to mean
    "this ROI is still alive and here's its committed state"; emitting
    a flapping count in a heartbeat would look indistinguishable from
    a real change to a consumer.

    Backward compat: ``change_stability_frames`` defaults to ``1``,
    which makes the change branch behave exactly like P4-C did
    (emit immediately on any difference). Production wiring sets it
    higher (default 5 via ``Settings.deepstream_roi_change_stability_frames``).

    Both ``emit_on_change`` and ``min_interval_seconds`` can be
    disabled (set to False / None) for tests."""

    stream_session_id: UUID
    emit_on_change: bool = True
    min_interval_seconds: float | None = 30.0
    # P4-C cleanup: number of consecutive frames a new (non-last-emitted)
    # count must hold before the change branch fires. 1 = legacy P4-C
    # behavior. >=2 suppresses short flaps. <1 is rejected up front.
    change_stability_frames: int = 1
    last_counts_by_roi: dict[str, dict[str, int]] = field(default_factory=dict)
    last_emit_at_by_roi: dict[str, datetime] = field(default_factory=dict)
    sequence_by_roi: dict[str, int] = field(default_factory=dict)
    # P4-C cleanup: per-ROI pending-candidate tracking. ``pending_count_by_roi``
    # is the snapshot under consideration; ``pending_seen_frames_by_roi``
    # counts how many consecutive frames it has been observed. Cleared
    # when the count returns to the last emitted value, or replaced when
    # a different new value arrives.
    pending_count_by_roi: dict[str, dict[str, int]] = field(default_factory=dict)
    pending_seen_frames_by_roi: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.change_stability_frames < 1:
            raise ValueError(
                f"change_stability_frames must be >= 1; got {self.change_stability_frames}"
            )

    def absorb_roi_counts(
        self,
        *,
        roi_counts: dict[str, int],
        class_counts_by_roi: dict[str, dict[str, int]] | None = None,
        occurred_at: datetime,
        source: dict[str, Any] | None = None,
        debug: dict[str, Any] | None = None,
    ) -> list[RoiOccupancyEvent]:
        """One ROI counts snapshot in, zero-to-many discrete events out.

        ``roi_counts`` maps ROI label → total object count in the ROI
        for this frame (the shape of ``NvDsAnalyticsFrameMeta.objInROIcnt``,
        as captured in P3.5). ``class_counts_by_roi`` is an optional
        finer breakdown; when not supplied, ``class_counts`` is left
        empty in the emitted event.

        Side effects: ``last_counts_by_roi`` / ``last_emit_at_by_roi`` /
        ``sequence_by_roi`` are updated only when an event is actually
        emitted. The pending state (``pending_count_by_roi`` /
        ``pending_seen_frames_by_roi``) is updated on every frame
        whether or not an event is emitted — that's what implements
        the debounce."""
        events: list[RoiOccupancyEvent] = []
        for roi_id, total in roi_counts.items():
            class_counts = (class_counts_by_roi or {}).get(roi_id, {})
            snapshot: dict[str, int] = {
                "__total": int(total),
                **{str(k): int(v) for k, v in class_counts.items()},
            }
            decision = self._classify_emit(roi_id, snapshot, occurred_at)
            if decision is None:
                continue
            reason, emit_snapshot, emit_total, emit_class_counts = decision
            seq = self.sequence_by_roi.get(roi_id, 0) + 1
            self.sequence_by_roi[roi_id] = seq
            self.last_counts_by_roi[roi_id] = emit_snapshot
            self.last_emit_at_by_roi[roi_id] = occurred_at
            # An emitted change resets the debounce — the new value is
            # now the committed last-emitted; pending is cleared.
            if reason == "change":
                self.pending_count_by_roi.pop(roi_id, None)
                self.pending_seen_frames_by_roi.pop(roi_id, None)
            key = build_roi_occupancy_idempotency_key(
                stream_session_id=self.stream_session_id,
                roi_id=roi_id,
                sequence=seq,
            )
            # P4-C: merge the emit reason into the event's debug dict so
            # consumers can tell first / change / interval apart without
            # re-deriving it. The caller's debug payload takes precedence
            # if it sets ``emit_reason`` explicitly (no-op in practice;
            # the worker never sets it).
            merged_debug: dict[str, Any] = {"emit_reason": reason, **(debug or {})}
            events.append(
                RoiOccupancyEvent(
                    type=CANDIDATE_ROI_OCCUPANCY_TYPE,
                    stream_session_id=self.stream_session_id,
                    occurred_at=occurred_at,
                    roi_id=roi_id,
                    roi_label=roi_id,
                    total_count=int(emit_total),
                    class_counts={str(k): int(v) for k, v in emit_class_counts.items()},
                    sequence=seq,
                    idempotency_key=key,
                    source=dict(source or {}),
                    debug=merged_debug,
                )
            )
        return events

    def _classify_emit(
        self,
        roi_id: str,
        snapshot: dict[str, int],
        occurred_at: datetime,
    ) -> tuple[str, dict[str, int], int, dict[str, int]] | None:
        """Decide whether to emit for this ROI and what to emit.

        Returns ``(reason, emit_snapshot, emit_total, emit_class_counts)``
        when emitting, where ``emit_*`` describe the count to put into
        the observation payload. Returns ``None`` to skip.

        The values are NOT always the input snapshot: during a pending
        change, a heartbeat reports the LAST EMITTED count, not the
        unstable candidate. The change branch updates pending state as
        a side effect even when it skips emission (that's the debounce
        accumulator)."""
        last = self.last_counts_by_roi.get(roi_id)

        # Case 1: first observation of this ROI — always emit it as-is.
        if last is None:
            return ("first", dict(snapshot), snapshot["__total"], _strip_total(snapshot))

        same_as_last = last == snapshot

        if same_as_last:
            # The count is back to the committed last-emitted value.
            # Clear any pending candidate (the flap is over).
            self.pending_count_by_roi.pop(roi_id, None)
            self.pending_seen_frames_by_roi.pop(roi_id, None)
            # Maybe heartbeat. The emit count IS the last (== snapshot).
            if self._interval_elapsed(roi_id, occurred_at):
                return ("interval", dict(snapshot), snapshot["__total"], _strip_total(snapshot))
            return None

        # Case 3: count differs from last-emitted. Update the pending
        # accumulator first; THEN decide whether to emit change or fall
        # through to a possible heartbeat-with-stable-count.
        if self.emit_on_change:
            pending = self.pending_count_by_roi.get(roi_id)
            if pending == snapshot:
                self.pending_seen_frames_by_roi[roi_id] = (
                    self.pending_seen_frames_by_roi.get(roi_id, 0) + 1
                )
            else:
                # New (or different) candidate — start counting from 1.
                self.pending_count_by_roi[roi_id] = dict(snapshot)
                self.pending_seen_frames_by_roi[roi_id] = 1
            if self.pending_seen_frames_by_roi.get(roi_id, 0) >= self.change_stability_frames:
                # Stable threshold met: emit a change with the new count.
                return (
                    "change",
                    dict(snapshot),
                    snapshot["__total"],
                    _strip_total(snapshot),
                )

        # Change is pending but not yet stable. Heartbeat may still fire
        # — but it MUST report the last stable count, not the unstable
        # candidate, so a downstream consumer can't mistake the heartbeat
        # for a real state change.
        if self._interval_elapsed(roi_id, occurred_at):
            return (
                "interval",
                dict(last),
                last["__total"],
                _strip_total(last),
            )
        return None

    def _interval_elapsed(self, roi_id: str, occurred_at: datetime) -> bool:
        if self.min_interval_seconds is None:
            return False
        last_at = self.last_emit_at_by_roi.get(roi_id)
        if last_at is None:
            return True
        return (occurred_at - last_at).total_seconds() >= float(self.min_interval_seconds)


def _strip_total(snapshot: dict[str, int]) -> dict[str, int]:
    """Return the per-class breakdown without the synthetic
    ``__total`` sentinel that ``_classify_emit`` uses internally to
    compare last-vs-new snapshots without giving the total special
    treatment."""
    return {k: v for k, v in snapshot.items() if k != "__total"}


# --- P4-D: per-track zone transitions ---------------------------------


@dataclass(frozen=True)
class ZoneTransitionEvent:
    """One ``processor.zone_entered`` or ``processor.zone_exited`` event.

    Distinct from ``LineCrossingEvent`` / ``RoiOccupancyEvent`` because
    a zone transition is keyed per ``(track_id, zone_id)`` — the same
    person walking in and out twice produces (enter, exit, enter, exit)
    with monotonically-increasing per-(track, zone, type) sequence
    numbers.

    Top-level payload is runtime-neutral. DeepStream-specific provenance
    sits under ``source``; raw frame numbers + the
    ``"outside_to_inside"`` / ``"inside_to_outside"`` transition string
    + ``stability_frames_required`` sit under ``debug``."""

    type: str  # CANDIDATE_ZONE_ENTERED_TYPE | CANDIDATE_ZONE_EXITED_TYPE
    stream_session_id: UUID
    occurred_at: datetime
    zone_id: str
    zone_label: str
    track_id: str
    sequence: int
    idempotency_key: str
    class_id: int | None = None
    class_label: str | None = None
    # Side task (workflow condition payload enrichment): the object's
    # detection confidence in the frame that COMMITTED the transition (the
    # Nth stable frame). None when the producer didn't have it — omitted
    # from the payload, never guessed.
    confidence: float | None = None
    source: dict[str, Any] = field(default_factory=dict)
    debug: dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        """Body for the ``payload`` field of ``ObservationPayload``.
        Optional fields (``class_id`` / ``class_label`` /
        ``confidence``) are omitted when None so an analytics frame
        without classifier metadata doesn't produce ``"class_id":
        null`` noise."""
        body: dict[str, Any] = {
            "zone_id": self.zone_id,
            "zone_label": self.zone_label,
            "track_id": self.track_id,
            "sequence": int(self.sequence),
            "idempotency_key": self.idempotency_key,
            "source": dict(self.source),
            "debug": dict(self.debug),
        }
        if self.class_id is not None:
            body["class_id"] = int(self.class_id)
        if self.class_label is not None:
            body["class_label"] = self.class_label
        if self.confidence is not None:
            body["confidence"] = round(float(self.confidence), 4)
        return body


def build_zone_entered_idempotency_key(
    *,
    stream_session_id: UUID,
    zone_id: str,
    track_id: str,
    sequence: int,
) -> str:
    """Deterministic dedupe key for a single ``zone_entered`` event.

    Format:
    ``sess:<session_id>:processor.zone_entered:<zone_id>:<track_id>:<sequence>``.
    The sequence is per-(track_id, zone_id, transition-type) so a
    person who walks in, out, in, out produces enter:1, exit:1,
    enter:2, exit:2 — the two transition types share zone/track
    but namespace their sequences independently."""
    return (
        f"sess:{stream_session_id}:{CANDIDATE_ZONE_ENTERED_TYPE}:"
        f"{zone_id}:{track_id}:{int(sequence)}"
    )


def build_zone_exited_idempotency_key(
    *,
    stream_session_id: UUID,
    zone_id: str,
    track_id: str,
    sequence: int,
) -> str:
    """Pairs with ``build_zone_entered_idempotency_key`` — same key
    structure, different type segment."""
    return (
        f"sess:{stream_session_id}:{CANDIDATE_ZONE_EXITED_TYPE}:"
        f"{zone_id}:{track_id}:{int(sequence)}"
    )


@dataclass
class ZoneMembershipState:
    """Per-stream-per-worker adapter state for zone transition events.

    Two emit conditions:

    1. **Stable inside**: a tracked object whose ``roi_labels`` contains
       a given zone for ``membership_stability_frames`` consecutive
       observed frames → emit ``processor.zone_entered``.
    2. **Stable outside** (after a stable inside): the same tracked
       object observed WITHOUT the zone label for
       ``membership_stability_frames`` consecutive frames →
       emit ``processor.zone_exited``.

    Important invariants:

    - **Object disappearance does NOT emit zone_exited.** A tracker
      dropping a frame can't be distinguished from a real exit, so we
      only act on observed frames. Pending state survives an arbitrary
      number of missing frames.
    - **First-observed-outside is a no-op.** If we never saw the track
      in this zone, we don't track an "outside" state for it; no
      ``zone_exited`` event can fire until we've committed an inside.
    - **Per-(track, zone, transition-type) sequence.** Re-entering a
      zone after exiting it produces ``zone_entered`` with sequence=2
      while ``zone_exited`` is still at sequence=1 — the two transition
      types share zone/track but namespace their sequences
      independently.

    Backward compat: ``membership_stability_frames`` defaults to ``1``
    so direct unit-test callers get immediate-transition behavior.
    Production wiring sets it from
    ``Settings.deepstream_zone_membership_stability_frames`` (default
    5). Values < 1 are rejected at ``__post_init__``."""

    stream_session_id: UUID
    membership_stability_frames: int = 1
    # Committed state per (track_id, zone_id). Missing key = never
    # observed inside this zone for this track. True = stably inside.
    # False = stably outside (only reachable after a prior True).
    stable_membership: dict[tuple[str, str], bool] = field(default_factory=dict)
    # Per-(track, zone) accumulator for the debounce.
    pending_membership: dict[tuple[str, str], bool] = field(default_factory=dict)
    pending_seen_frames: dict[tuple[str, str], int] = field(default_factory=dict)
    # Per-(track_id, zone_id, obs_type) emit counter — idempotency-key
    # sequence. The two transition types each carry their own counter
    # so a re-enter is :2 and a re-exit is also :2 (distinct keys
    # because the obs_type segment of the key differs).
    sequence_by_track_zone_transition: dict[tuple[str, str, str], int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.membership_stability_frames < 1:
            raise ValueError(
                f"membership_stability_frames must be >= 1; got {self.membership_stability_frames}"
            )

    def absorb_object(
        self,
        *,
        track_id: str,
        zone_set: set[str] | frozenset[str],
        roi_labels: set[str] | frozenset[str],
        occurred_at: datetime,
        class_id: int | None = None,
        class_label: str | None = None,
        confidence: float | None = None,
        source: dict[str, Any] | None = None,
        debug: dict[str, Any] | None = None,
    ) -> list[ZoneTransitionEvent]:
        """Update membership state for one observed object across the
        currently-known zone set; emit transition events for any
        membership change that has been stable for
        ``membership_stability_frames`` frames.

        Caller is responsible for filtering out objects without a
        ``tracker_id``; this method assumes ``track_id`` is a non-empty
        string. Caller also decides what counts as "the known zone
        set" for this frame (typically the keys of
        ``frame.source.analytics.roi_counts``)."""
        events: list[ZoneTransitionEvent] = []
        for zone_id in zone_set:
            observed = zone_id in roi_labels
            key = (track_id, zone_id)
            committed = self.stable_membership.get(key)

            # Case A: never observed inside this zone for this track,
            # and not observed inside now. No state to track.
            if committed is None and observed is False:
                continue

            # Case B: matches committed. Clear any pending state.
            if observed == committed:
                self.pending_membership.pop(key, None)
                self.pending_seen_frames.pop(key, None)
                continue

            # Case C: differs from committed (or first-ever sighting
            # in this zone). Update the pending accumulator.
            pending = self.pending_membership.get(key)
            if pending == observed:
                self.pending_seen_frames[key] = self.pending_seen_frames.get(key, 0) + 1
            else:
                self.pending_membership[key] = observed
                self.pending_seen_frames[key] = 1

            if self.pending_seen_frames.get(key, 0) < self.membership_stability_frames:
                continue

            # Threshold met: commit + emit.
            self.stable_membership[key] = observed
            self.pending_membership.pop(key, None)
            self.pending_seen_frames.pop(key, None)
            obs_type = CANDIDATE_ZONE_ENTERED_TYPE if observed else CANDIDATE_ZONE_EXITED_TYPE
            seq_key = (track_id, zone_id, obs_type)
            seq = self.sequence_by_track_zone_transition.get(seq_key, 0) + 1
            self.sequence_by_track_zone_transition[seq_key] = seq
            if observed:
                idem = build_zone_entered_idempotency_key(
                    stream_session_id=self.stream_session_id,
                    zone_id=zone_id,
                    track_id=track_id,
                    sequence=seq,
                )
                transition = "outside_to_inside"
            else:
                idem = build_zone_exited_idempotency_key(
                    stream_session_id=self.stream_session_id,
                    zone_id=zone_id,
                    track_id=track_id,
                    sequence=seq,
                )
                transition = "inside_to_outside"
            merged_debug: dict[str, Any] = {
                "transition": transition,
                "stability_frames_required": self.membership_stability_frames,
                **(debug or {}),
            }
            events.append(
                ZoneTransitionEvent(
                    type=obs_type,
                    stream_session_id=self.stream_session_id,
                    occurred_at=occurred_at,
                    zone_id=zone_id,
                    zone_label=zone_id,
                    track_id=track_id,
                    sequence=seq,
                    idempotency_key=idem,
                    class_id=class_id,
                    class_label=class_label,
                    confidence=confidence,
                    source=dict(source or {}),
                    debug=merged_debug,
                )
            )
        return events


# --- P4-E: dwell observations ----------------------------------------


@dataclass(frozen=True)
class DwellEvent:
    """One ``processor.dwell_started`` / ``dwell_updated`` /
    ``dwell_exceeded`` event.

    A single dataclass covers all three transition variants because
    the payloads share zone_id / track_id / dwell_session_id /
    started_at / duration_seconds — only ``update_sequence`` (for
    ``dwell_updated``) and ``threshold_seconds`` (for
    ``dwell_exceeded``) are type-specific, and both are optional /
    omitted in the other variants' payloads. ``type`` selects the
    Literal at the writer."""

    type: str  # CANDIDATE_DWELL_* — must be one of the three.
    stream_session_id: UUID
    occurred_at: datetime
    zone_id: str
    zone_label: str
    track_id: str
    dwell_session_id: str
    started_at: datetime
    duration_seconds: float
    idempotency_key: str
    class_id: int | None = None
    class_label: str | None = None
    # Side task (workflow condition payload enrichment): the object's
    # detection confidence in the frame that TRIGGERED this event — the
    # zone-entry commit frame for dwell_started, the tick frame for
    # dwell_updated / dwell_exceeded. None (omitted from the payload) when
    # the track wasn't detected in that frame; a stale entry-time value is
    # never re-stamped onto later events.
    confidence: float | None = None
    # Only set for ``dwell_updated``; the rolling per-session sequence.
    update_sequence: int | None = None
    # Only set for ``dwell_exceeded``; the threshold that was crossed.
    threshold_seconds: float | None = None
    source: dict[str, Any] = field(default_factory=dict)
    debug: dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        """Body for ``ObservationPayload.payload``. Top-level fields are
        runtime-neutral; ``source`` / ``debug`` carry DeepStream
        provenance + raw frame numbers. Optional fields
        (``class_id`` / ``class_label`` / ``update_sequence`` /
        ``threshold_seconds``) are omitted when None so the
        ``dwell_started`` payload doesn't carry ``"update_sequence":
        null`` noise."""
        body: dict[str, Any] = {
            "zone_id": self.zone_id,
            "zone_label": self.zone_label,
            "track_id": self.track_id,
            "dwell_session_id": self.dwell_session_id,
            "started_at": self.started_at.isoformat(),
            # Round to 2 dp so payloads are stable across small clock
            # jitter (the rounded value still encodes ~10 ms).
            "duration_seconds": round(float(self.duration_seconds), 2),
            "idempotency_key": self.idempotency_key,
            "source": dict(self.source),
            "debug": dict(self.debug),
        }
        if self.class_id is not None:
            body["class_id"] = int(self.class_id)
        if self.class_label is not None:
            body["class_label"] = self.class_label
        if self.confidence is not None:
            body["confidence"] = round(float(self.confidence), 4)
        if self.update_sequence is not None:
            body["update_sequence"] = int(self.update_sequence)
        if self.threshold_seconds is not None:
            body["threshold_seconds"] = round(float(self.threshold_seconds), 2)
        return body


def build_dwell_started_idempotency_key(
    *,
    stream_session_id: UUID,
    zone_id: str,
    track_id: str,
    dwell_session_id: str,
) -> str:
    """``sess:<id>:processor.dwell_started:<zone>:<track>:<dwell_session_id>``.

    ``dwell_session_id`` already contains
    ``<zone>:<track>:<enter_sequence>`` (so the prefix duplicates
    ``<zone>:<track>``); the duplication is intentional — it makes the
    key independently parseable as a single string without re-parsing
    ``dwell_session_id``, and keeps the prefix consistent with the
    other dwell event types."""
    return (
        f"sess:{stream_session_id}:{CANDIDATE_DWELL_STARTED_TYPE}:"
        f"{zone_id}:{track_id}:{dwell_session_id}"
    )


def build_dwell_updated_idempotency_key(
    *,
    stream_session_id: UUID,
    zone_id: str,
    track_id: str,
    dwell_session_id: str,
    update_sequence: int,
) -> str:
    """Adds ``:<update_sequence>`` after the dwell_session_id so each
    periodic update inside a session has a unique deterministic key."""
    return (
        f"sess:{stream_session_id}:{CANDIDATE_DWELL_UPDATED_TYPE}:"
        f"{zone_id}:{track_id}:{dwell_session_id}:{int(update_sequence)}"
    )


def build_dwell_exceeded_idempotency_key(
    *,
    stream_session_id: UUID,
    zone_id: str,
    track_id: str,
    dwell_session_id: str,
    threshold_seconds: float,
) -> str:
    """Adds ``:<threshold_seconds>`` (as int) after the dwell_session_id.
    Casting to int avoids floating-point key drift between calls
    while still uniquely identifying configurable thresholds (e.g.
    60 vs 300)."""
    return (
        f"sess:{stream_session_id}:{CANDIDATE_DWELL_EXCEEDED_TYPE}:"
        f"{zone_id}:{track_id}:{dwell_session_id}:{int(threshold_seconds)}"
    )


@dataclass
class _ActiveDwell:
    """Per-(track, zone) dwell session bookkeeping. Created on
    zone_entered; closed (removed from the map) on zone_exited.

    ``update_sequence`` starts at 0 and increments to 1, 2, 3, … on
    each ``dwell_updated`` emit. ``exceeded_emitted`` is the
    single-shot guard for ``dwell_exceeded``."""

    dwell_session_id: str
    started_at: datetime
    last_update_at: datetime
    update_sequence: int = 0
    exceeded_emitted: bool = False
    class_id: int | None = None
    class_label: str | None = None
    zone_label: str = ""


@dataclass
class DwellState:
    """Per-stream-per-worker adapter state for dwell observations.

    Inputs come from two channels:
      - ``absorb_zone_event(zone_event)`` — start or close a dwell
        session when ZoneMembershipState commits a transition.
      - ``tick(occurred_at)`` — periodic check on each frame to fire
        ``dwell_updated`` (when ``update_interval_seconds`` elapsed)
        and ``dwell_exceeded`` (when ``exceeded_threshold_seconds``
        is crossed).

    Important invariants (mirror P4-D's disappearance guard):
      - A track disappearing from a frame does NOT close the dwell
        session. The session stays active until the matching
        ``zone_exited`` arrives, or until the supervisor restarts the
        worker.
      - After a dwell session is closed by ``zone_exited``, no more
        ``dwell_updated`` / ``dwell_exceeded`` events fire for that
        session. A subsequent ``zone_entered`` (re-enter) starts a NEW
        ``dwell_session_id`` with a fresh threshold-emitted state.
      - ``dwell_exceeded`` is single-shot per session — even if the
        track stays in the zone forever, the event fires exactly once.
      - ``track_id`` and ``zone_id`` are required (already guaranteed
        by ZoneMembershipState's contract)."""

    stream_session_id: UUID
    # Default 30s matches Settings.deepstream_dwell_update_interval_seconds.
    update_interval_seconds: float = 30.0
    # Default 60s matches Settings.deepstream_dwell_exceeded_threshold_seconds.
    exceeded_threshold_seconds: float = 60.0
    # Active dwell sessions per (track_id, zone_id). Created on
    # zone_entered; popped on zone_exited.
    active_dwells: dict[tuple[str, str], _ActiveDwell] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.update_interval_seconds <= 0:
            raise ValueError(
                f"update_interval_seconds must be > 0; got {self.update_interval_seconds}"
            )
        if self.exceeded_threshold_seconds <= 0:
            raise ValueError(
                f"exceeded_threshold_seconds must be > 0; got {self.exceeded_threshold_seconds}"
            )

    def absorb_zone_event(
        self,
        zone_event: ZoneTransitionEvent,
        *,
        source: dict[str, Any] | None = None,
        debug: dict[str, Any] | None = None,
    ) -> list[DwellEvent]:
        """Process a single ``zone_entered`` / ``zone_exited`` event:
          - ``zone_entered`` → start a new session (idempotent — if an
            active session already exists for this (track, zone), do
            NOT emit another ``dwell_started``; the duplicate guard
            mirrors P4-D's stability invariant).
          - ``zone_exited`` → close the session silently (no
            ``dwell_ended`` in P4-E).
        Returns at most one ``dwell_started`` event."""
        events: list[DwellEvent] = []
        key = (zone_event.track_id, zone_event.zone_id)
        if zone_event.type == CANDIDATE_ZONE_ENTERED_TYPE:
            if key in self.active_dwells:
                # Defensive: ZoneMembershipState should never emit two
                # zone_entered events back-to-back for the same
                # (track, zone), but if it does, don't duplicate the
                # dwell_started.
                return events
            dwell_session_id = f"{zone_event.zone_id}:{zone_event.track_id}:{zone_event.sequence}"
            active = _ActiveDwell(
                dwell_session_id=dwell_session_id,
                started_at=zone_event.occurred_at,
                last_update_at=zone_event.occurred_at,
                class_id=zone_event.class_id,
                class_label=zone_event.class_label,
                zone_label=zone_event.zone_label,
            )
            self.active_dwells[key] = active
            idem = build_dwell_started_idempotency_key(
                stream_session_id=self.stream_session_id,
                zone_id=zone_event.zone_id,
                track_id=zone_event.track_id,
                dwell_session_id=dwell_session_id,
            )
            merged_debug: dict[str, Any] = {"trigger": "zone_entered", **(debug or {})}
            events.append(
                DwellEvent(
                    type=CANDIDATE_DWELL_STARTED_TYPE,
                    stream_session_id=self.stream_session_id,
                    occurred_at=zone_event.occurred_at,
                    zone_id=zone_event.zone_id,
                    zone_label=zone_event.zone_label,
                    track_id=zone_event.track_id,
                    dwell_session_id=dwell_session_id,
                    started_at=zone_event.occurred_at,
                    duration_seconds=0.0,
                    idempotency_key=idem,
                    class_id=zone_event.class_id,
                    class_label=zone_event.class_label,
                    # Same frame as the committed zone entry — provable.
                    confidence=zone_event.confidence,
                    source=dict(source or {}),
                    debug=merged_debug,
                )
            )
        elif zone_event.type == CANDIDATE_ZONE_EXITED_TYPE:
            # Close the session silently. dwell_ended is intentionally
            # deferred — when the matching zone_exited has stable
            # semantics from ZoneMembershipState we know the operator
            # is gone; further updates would be noise.
            self.active_dwells.pop(key, None)
        # Other zone event types (none currently) are no-ops.
        return events

    def tick(
        self,
        *,
        occurred_at: datetime,
        confidence_by_track: dict[str, float] | None = None,
        source: dict[str, Any] | None = None,
        debug: dict[str, Any] | None = None,
    ) -> list[DwellEvent]:
        """Per-frame check: emit dwell_updated (if the update interval
        has elapsed since the last update / start) and dwell_exceeded
        (if the total duration crosses the threshold for the first
        time). Updates come BEFORE the exceeded check inside the same
        tick — operationally identical because both are idempotent
        per session but the order makes the sequence numbering
        deterministic in tests.

        ``confidence_by_track`` is the CURRENT tick frame's per-track
        detection confidence (side task: workflow condition payload
        enrichment). A track absent from the map — e.g. temporarily
        undetected, which deliberately does NOT close its session —
        emits with no ``confidence`` field rather than a stale value."""
        events: list[DwellEvent] = []
        confidences = confidence_by_track or {}
        for (track_id, zone_id), active in list(self.active_dwells.items()):
            duration = (occurred_at - active.started_at).total_seconds()
            since_last = (occurred_at - active.last_update_at).total_seconds()
            # dwell_updated: emits at every full update_interval since
            # the last emit (NOT since session start). After
            # dwell_started, this is the next emission this session
            # produces. Update_sequence increments per emit.
            if since_last >= self.update_interval_seconds:
                active.update_sequence += 1
                active.last_update_at = occurred_at
                idem = build_dwell_updated_idempotency_key(
                    stream_session_id=self.stream_session_id,
                    zone_id=zone_id,
                    track_id=track_id,
                    dwell_session_id=active.dwell_session_id,
                    update_sequence=active.update_sequence,
                )
                merged_debug: dict[str, Any] = {"trigger": "interval", **(debug or {})}
                events.append(
                    DwellEvent(
                        type=CANDIDATE_DWELL_UPDATED_TYPE,
                        stream_session_id=self.stream_session_id,
                        occurred_at=occurred_at,
                        zone_id=zone_id,
                        zone_label=active.zone_label,
                        track_id=track_id,
                        dwell_session_id=active.dwell_session_id,
                        started_at=active.started_at,
                        duration_seconds=duration,
                        idempotency_key=idem,
                        class_id=active.class_id,
                        class_label=active.class_label,
                        confidence=confidences.get(track_id),
                        update_sequence=active.update_sequence,
                        source=dict(source or {}),
                        debug=merged_debug,
                    )
                )
            # dwell_exceeded: single-shot per session. Once emitted,
            # never re-emitted even if the track stays inside forever
            # (re-entering produces a NEW session_id with a fresh
            # exceeded_emitted=False).
            if not active.exceeded_emitted and duration >= self.exceeded_threshold_seconds:
                active.exceeded_emitted = True
                idem_exc = build_dwell_exceeded_idempotency_key(
                    stream_session_id=self.stream_session_id,
                    zone_id=zone_id,
                    track_id=track_id,
                    dwell_session_id=active.dwell_session_id,
                    threshold_seconds=self.exceeded_threshold_seconds,
                )
                merged_debug_exc: dict[str, Any] = {
                    "trigger": "threshold",
                    **(debug or {}),
                }
                events.append(
                    DwellEvent(
                        type=CANDIDATE_DWELL_EXCEEDED_TYPE,
                        stream_session_id=self.stream_session_id,
                        occurred_at=occurred_at,
                        zone_id=zone_id,
                        zone_label=active.zone_label,
                        track_id=track_id,
                        dwell_session_id=active.dwell_session_id,
                        started_at=active.started_at,
                        duration_seconds=duration,
                        idempotency_key=idem_exc,
                        class_id=active.class_id,
                        class_label=active.class_label,
                        confidence=confidences.get(track_id),
                        threshold_seconds=self.exceeded_threshold_seconds,
                        source=dict(source or {}),
                        debug=merged_debug_exc,
                    )
                )
        return events

    def active_session_count(self) -> int:
        """For the status payload — count of currently-open dwell
        sessions. Helpful triage signal: if this stays 0 but
        zone_entered events are flowing, that's a DwellState bug."""
        return len(self.active_dwells)


# --- ergonomic helper --------------------------------------------------


def deepstream_source_block(
    *,
    package_id: str,
    tracker_algorithm: str | None = None,
    geometry_source: str | None = None,
    geometry_revision: int | None = None,
    applied_processing_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Stable provenance block for the ``source`` field on events from
    the DeepStream analytics path. Pulled out so worker code does NOT
    inline these literals at the emit site.

    P6-C: ``geometry_source`` / ``geometry_revision`` are flat (consistent
    with the existing ``runtime`` / ``analytics_plugin`` / etc keys) so
    downstream consumers can index on the lineage without parsing nested
    objects. Omitted when None — keeps probe-only test fixtures unchanged.

    M8: ``applied_processing_config`` (the resolver's ``status_payload()`` —
    the SAME requested-vs-applied lineage stamped on processor status by
    M7B/C/D) is ADDITIVE metadata; it changes no count. The honest
    ``selected_classes_analytics_counts_not_filtered`` caveat already rides
    inside that block's ``warnings`` when selected_classes is applied — it is
    NOT stripped here. Omitted when None (mirrors the fields above) so
    existing fixtures stay byte-identical.
    """
    src: dict[str, Any] = {
        "runtime": "deepstream",
        "package_id": package_id,
        "metadata_source": "pyds_pad_probe",
        "analytics_plugin": "nvdsanalytics",
    }
    if tracker_algorithm is not None:
        src["tracker_algorithm"] = tracker_algorithm
    if geometry_source is not None:
        src["geometry_source"] = geometry_source
    if geometry_revision is not None:
        src["geometry_revision"] = geometry_revision
    if applied_processing_config is not None:
        src["applied_processing_config"] = applied_processing_config
    return src


def iter_class_label_counts(label_counts: Iterable[tuple[str, int]]) -> dict[str, int]:
    """Helper for callers building ``class_counts_by_roi`` from worker
    counters — accepts any iterable of ``(label, count)`` and returns a
    plain dict. Stays out of the adapter's hot path; lets tests assert
    a stable shape."""
    return {str(k): int(v) for k, v in label_counts}
