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
from datetime import datetime
from typing import Any
from uuid import UUID

# Candidate observation type strings the P4 adapter will emit. Defined
# here (NOT in ``app/observations/schemas.py``) until the server-side
# literal is widened — keeping them out of the strict-validated payload
# until then prevents accidental production POSTs. P4-B will move these
# into ``schemas.ObservationType`` in lock-step with the api-server PR.
CANDIDATE_LINE_CROSSING_TYPE = "processor.line_crossing"
CANDIDATE_ROI_OCCUPANCY_TYPE = "processor.roi_occupancy"

CANDIDATE_OBSERVATION_TYPES: tuple[str, ...] = (
    CANDIDATE_LINE_CROSSING_TYPE,
    CANDIDATE_ROI_OCCUPANCY_TYPE,
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


# --- ergonomic helper --------------------------------------------------


def deepstream_source_block(
    *,
    package_id: str,
    tracker_algorithm: str | None = None,
) -> dict[str, Any]:
    """Stable provenance block for the ``source`` field on events from
    the DeepStream analytics path. Pulled out so worker code does NOT
    inline these literals at the emit site."""
    src: dict[str, Any] = {
        "runtime": "deepstream",
        "package_id": package_id,
        "metadata_source": "pyds_pad_probe",
        "analytics_plugin": "nvdsanalytics",
    }
    if tracker_algorithm is not None:
        src["tracker_algorithm"] = tracker_algorithm
    return src


def iter_class_label_counts(label_counts: Iterable[tuple[str, int]]) -> dict[str, int]:
    """Helper for callers building ``class_counts_by_roi`` from worker
    counters — accepts any iterable of ``(label, count)`` and returns a
    plain dict. Stays out of the adapter's hot path; lets tests assert
    a stable shape."""
    return {str(k): int(v) for k, v in label_counts}
