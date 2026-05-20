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

    Two emit conditions, OR'd together:

    1. **On change** — total count for an ROI changed since the last
       emit. Catches "person stepped into ROI" and "person left ROI".
    2. **On interval** — at least ``min_interval_seconds`` have passed
       since the last emit, even if the count is unchanged. Keeps the
       observation stream visible during long steady periods.

    Both can be disabled (``emit_on_change=False``,
    ``min_interval_seconds=None``) for tests. v0 recommends
    ``emit_on_change=True`` + a generous interval (e.g. 30 s) — we want
    state, not noise."""

    stream_session_id: UUID
    emit_on_change: bool = True
    min_interval_seconds: float | None = 30.0
    last_counts_by_roi: dict[str, dict[str, int]] = field(default_factory=dict)
    last_emit_at_by_roi: dict[str, datetime] = field(default_factory=dict)
    sequence_by_roi: dict[str, int] = field(default_factory=dict)

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
        empty in the emitted event."""
        events: list[RoiOccupancyEvent] = []
        for roi_id, total in roi_counts.items():
            class_counts = (class_counts_by_roi or {}).get(roi_id, {})
            reason = self._classify_emit(roi_id, total, class_counts, occurred_at)
            if reason is None:
                continue
            seq = self.sequence_by_roi.get(roi_id, 0) + 1
            self.sequence_by_roi[roi_id] = seq
            self.last_counts_by_roi[roi_id] = {
                "__total": int(total),
                **{str(k): int(v) for k, v in class_counts.items()},
            }
            self.last_emit_at_by_roi[roi_id] = occurred_at
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
                    total_count=int(total),
                    class_counts={str(k): int(v) for k, v in class_counts.items()},
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
        total: int,
        class_counts: dict[str, int],
        occurred_at: datetime,
    ) -> str | None:
        """Decide whether to emit for this ROI, and why.

        Returns one of ``"first"`` / ``"change"`` / ``"interval"`` to
        emit (the value rides in the event's ``debug.emit_reason`` so
        operators can tell heartbeats from state changes at a glance),
        or ``None`` to skip."""
        last = self.last_counts_by_roi.get(roi_id)
        snapshot: dict[str, int] = {
            "__total": int(total),
            **{str(k): int(v) for k, v in class_counts.items()},
        }
        if last is None:
            # First observation of this ROI in this state — always emit so
            # consumers know it exists.
            return "first"
        if self.emit_on_change and last != snapshot:
            return "change"
        if self.min_interval_seconds is None:
            return None
        last_at = self.last_emit_at_by_roi.get(roi_id)
        if last_at is None:
            return "interval"
        elapsed = (occurred_at - last_at).total_seconds()
        return "interval" if elapsed >= float(self.min_interval_seconds) else None


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
