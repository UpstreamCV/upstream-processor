"""P4-A.5 — internal processor observation taxonomy + emission policy.

A small, type-checked table of every ``processor.*`` observation type
the design admits, along with:

- **category** (event / state / telemetry) — drives the emission
  policy class (event = one-shot, state = on-change + interval,
  telemetry = sampled / test-mode).
- **status** (api_supported / proposed_first_api_add / deferred) —
  drives the api-server widening order. **Only the first
  ``processor.*`` type to land is marked ``proposed_first_api_add``**;
  everything else stays ``deferred`` until its own milestone.
- **requirements** (tracker_id, line/zone geometry, dashboard-driven
  config) — drives "can we even emit this today?" The probe shipped
  in P3.5 owns its own geometry, so probe-shipped types do not need
  ``requires_dashboard_geometry``; future operator-authored zones do.
- **idempotency key pattern** — the exact deterministic key shape
  the adapter (P4-A) builds, repeated here so future api-server
  dedupe migrations can re-derive it without grepping the runtime
  code.

This module is **pure data + helpers**. It does NOT:

- widen ``schemas.OBSERVATION_TYPES`` (that gate stays narrow until
  the api-server matching PR lands; the taxonomy is allowed to
  describe types the server doesn't accept yet).
- call ``ObservationWriter`` or any API.
- depend on DeepStream / pyds / gi.

Future agents touching this file: the rule is "describe the type
*before* implementing it" — adding a row here is cheap, opens a
conversation, and makes the deferred list grep-able. Removing or
renaming a row is breaking; if the type was ever emitted, treat the
rename as an api-server schema change too.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

ObservationCategory = Literal["event", "state", "telemetry"]
"""Three emission policy classes. See module docstring."""

ObservationStatus = Literal[
    "api_supported",  # type is in the deployed api-server's Literal whitelist today
    "proposed_first_api_add",  # next type to widen the api-server whitelist with
    "deferred",  # designed; deliberately not in the api-server whitelist yet
]
"""Drives the api-server widening order. P4-A.5 marks exactly one type
as ``proposed_first_api_add`` (processor.line_crossing); P4-B advances
that one to ``api_supported`` once both ends of the wire are widened."""


@dataclass(frozen=True)
class ObservationSpec:
    """One row in the taxonomy table.

    ``description`` is a single sentence so the table can be skimmed.
    ``payload_keys`` lists the runtime-neutral keys the body will have
    (excluding ``source`` / ``debug`` / ``idempotency_key`` which every
    type carries — the goal is to make the SHAPE of each type obvious
    at a glance without reading the adapter)."""

    type: str
    category: ObservationCategory
    status: ObservationStatus
    description: str
    payload_keys: tuple[str, ...]
    idempotency_key_pattern: str
    source_signal: str
    requires_tracker_id: bool = False
    requires_line_geometry: bool = False
    requires_zone_geometry: bool = False
    requires_dashboard_geometry: bool = False
    dependencies: tuple[str, ...] = field(default_factory=tuple)


# ----------------------------------------------------------------------
# Initial taxonomy table (P4-A.5).
#
# Order matters for human reading: events first (lowest-volume / highest
# signal), then state observations (deduped via change-detection /
# interval), then telemetry (sampled). Within each category, the type
# most likely to ship next is listed first.
# ----------------------------------------------------------------------

INITIAL_TAXONOMY: tuple[ObservationSpec, ...] = (
    # --- event-style: one-shot, dedupe via deterministic key ----------
    ObservationSpec(
        type="processor.line_crossing",
        category="event",
        # P4-B widened the api-server ``ProcessorObservationType`` Literal
        # AND this repo's ``schemas.OBSERVATION_TYPES`` in lock-step. The
        # invariant test ``test_no_processor_type_is_already_in_observation_types``
        # would have fired if either side moved without the other.
        status="api_supported",
        description=(
            "One discrete crossing of a configured line. Source is the "
            "nvdsanalytics frame-level objLCCumCnt counter; the adapter "
            "turns each integer increment into one event."
        ),
        payload_keys=(
            "line_id",
            "line_label",
            "count_delta",
            "cumulative_count",
            "track_id",  # optional
            "class_id",  # optional
            "class_label",  # optional
            "direction",  # optional
        ),
        idempotency_key_pattern=(
            "sess:<session_id>:processor.line_crossing:<line_id>:<cumulative_count>"
        ),
        source_signal="nvdsanalytics frame_user_meta_list objLCCumCnt delta",
        requires_line_geometry=True,
        # P3.5 ships the line geometry hard-coded in build_analytics_config.
        # Dashboard-driven geometry is a later milestone; not blocking P4-B.
    ),
    ObservationSpec(
        type="processor.zone_entered",
        category="event",
        status="deferred",
        description=(
            "Tracked object's first frame INSIDE a zone polygon. "
            "Distinct event per (zone, tracker_id, enter sequence) so "
            "a person leaving and re-entering emits two events."
        ),
        payload_keys=("zone_id", "zone_label", "track_id", "enter_sequence"),
        idempotency_key_pattern=(
            "sess:<session_id>:processor.zone_entered:<zone_id>:<track_id>:<enter_sequence>"
        ),
        source_signal="per-object NvDsAnalyticsObjInfo.roiStatus transition (absent -> present)",
        requires_tracker_id=True,
        requires_zone_geometry=True,
        # ROI membership transition tracking needs a per-(track, zone)
        # last-state map that P3.5's per-frame roi_labels list does
        # not currently maintain; that's a P4-C+ adapter addition.
    ),
    ObservationSpec(
        type="processor.zone_exited",
        category="event",
        status="deferred",
        description=(
            "Tracked object's first frame OUTSIDE a zone after having "
            "been inside it. Pairs with zone_entered."
        ),
        payload_keys=("zone_id", "zone_label", "track_id", "exit_sequence"),
        idempotency_key_pattern=(
            "sess:<session_id>:processor.zone_exited:<zone_id>:<track_id>:<exit_sequence>"
        ),
        source_signal="per-object NvDsAnalyticsObjInfo.roiStatus transition (present -> absent)",
        requires_tracker_id=True,
        requires_zone_geometry=True,
        dependencies=("processor.zone_entered",),
    ),
    ObservationSpec(
        type="processor.dwell_exceeded",
        category="event",
        status="deferred",
        description=(
            "A tracked object has been inside a zone continuously for "
            "longer than the configured threshold (e.g. 30s / 5min). "
            "One event per (track, zone, threshold) — re-entering "
            "starts a new dwell session and resets the thresholds."
        ),
        payload_keys=(
            "zone_id",
            "zone_label",
            "track_id",
            "threshold_seconds",
            "dwell_session_id",
            "dwell_seconds",
        ),
        idempotency_key_pattern=(
            "sess:<session_id>:processor.dwell_exceeded:"
            "<zone_id>:<track_id>:<dwell_session_id>:<threshold_seconds>"
        ),
        source_signal="DwellState.update(...) crossing a configured threshold",
        requires_tracker_id=True,
        requires_zone_geometry=True,
        dependencies=("processor.zone_entered", "processor.zone_exited"),
    ),
    ObservationSpec(
        type="processor.direction_detected",
        category="event",
        status="deferred",
        description=(
            "Tracked object moved in a configured cardinal/named "
            "direction. One event per (track, direction-label) within "
            "a debounce window so a person walking back-and-forth does "
            "not flood."
        ),
        payload_keys=("direction_label", "track_id", "debounce_window_seconds"),
        idempotency_key_pattern=(
            "sess:<session_id>:processor.direction_detected:"
            "<direction_label>:<track_id>:<bucket_id>"
        ),
        source_signal="per-object NvDsAnalyticsObjInfo.dirStatus",
        requires_tracker_id=True,
    ),
    # --- state-style: emit on change OR every N seconds ---------------
    ObservationSpec(
        type="processor.roi_occupancy",
        category="state",
        status="deferred",
        description=(
            "Snapshot of how many tracked objects are currently inside "
            "an ROI, broken down by class. Emitted on change AND "
            "periodically every min_interval_seconds; never per-frame."
        ),
        payload_keys=("roi_id", "roi_label", "total_count", "class_counts", "sequence"),
        idempotency_key_pattern=("sess:<session_id>:processor.roi_occupancy:<roi_id>:<sequence>"),
        source_signal="nvdsanalytics frame_user_meta_list objInROIcnt",
    ),
    ObservationSpec(
        type="processor.zone_occupancy",
        category="state",
        status="deferred",
        description=(
            "Like roi_occupancy but for operator-named zones (the same "
            "polygon, different naming intent). May fold into "
            "roi_occupancy once dashboard zone geometry exists."
        ),
        payload_keys=("zone_id", "zone_label", "total_count", "class_counts", "sequence"),
        idempotency_key_pattern=("sess:<session_id>:processor.zone_occupancy:<zone_id>:<sequence>"),
        source_signal="nvdsanalytics + per-track zone membership",
        requires_zone_geometry=True,
        requires_dashboard_geometry=True,
    ),
    ObservationSpec(
        type="processor.queue_length",
        category="state",
        status="deferred",
        description=(
            "Snapshot of how many tracked objects are in a configured "
            "queue zone (a specialised ROI). Same emission policy as "
            "roi_occupancy; distinct type so consumers can tell a "
            "queue from a generic occupancy state."
        ),
        payload_keys=("queue_id", "queue_label", "queue_length", "sequence"),
        idempotency_key_pattern=("sess:<session_id>:processor.queue_length:<queue_id>:<sequence>"),
        source_signal="nvdsanalytics ROI + queue config",
        requires_zone_geometry=True,
        requires_dashboard_geometry=True,
    ),
    ObservationSpec(
        type="processor.track_state",
        category="state",
        status="deferred",
        description=(
            "Per-track lifecycle: appeared / tracked-N-frames / "
            "disappeared. Bounded by sampling (every K frames) and by "
            "a per-track interval so 30 fps times N tracks does not flood."
        ),
        payload_keys=(
            "track_id",
            "class_label",
            "lifecycle_state",
            "frames_tracked",
            "sequence",
        ),
        idempotency_key_pattern=("sess:<session_id>:processor.track_state:<track_id>:<sequence>"),
        source_signal="nvtracker NvDsObjectMeta.object_id + frame age",
        requires_tracker_id=True,
    ),
    # --- telemetry-style: sampled / debug; opt-in by default ----------
    ObservationSpec(
        type="processor.track_update",
        category="telemetry",
        status="deferred",
        description=(
            "Periodic per-track snapshot (position, bbox, class, "
            "confidence). High volume — must be sampled or debug-only. "
            "Useful for diagnostics, not as a primary signal."
        ),
        payload_keys=("track_id", "class_label", "bbox_pixels", "confidence", "bucket_id"),
        idempotency_key_pattern=("sess:<session_id>:processor.track_update:<track_id>:<bucket_id>"),
        source_signal="every Nth pyds NvDsObjectMeta",
        requires_tracker_id=True,
    ),
    ObservationSpec(
        type="processor.detection_seen",
        category="telemetry",
        status="deferred",
        description=(
            "Coarse-grained per-class detection count over a time "
            "bucket. Cheap proxy for 'pipeline is alive and seeing "
            "objects'. Default off in production."
        ),
        payload_keys=("class_label", "count", "bucket_id"),
        idempotency_key_pattern=(
            "sess:<session_id>:processor.detection_seen:<class_label>:<bucket_id>"
        ),
        source_signal="per-frame class_counts aggregated over an interval",
    ),
    ObservationSpec(
        type="processor.frame_summary",
        category="telemetry",
        status="deferred",
        description=(
            "One observation per (sampled) frame with frames_seen / "
            "frames_with_objects / fps. Likely never used in production "
            "(the same numbers live in the status payload), but useful "
            "for offline analysis."
        ),
        payload_keys=("frame_num", "frames_seen", "frames_with_objects", "bucket_id"),
        idempotency_key_pattern=("sess:<session_id>:processor.frame_summary:<bucket_id>"),
        source_signal="pyds pad probe summary counters",
    ),
    ObservationSpec(
        type="processor.scene_quality_warning",
        category="telemetry",
        status="deferred",
        description=(
            "Probe-detected scene anomaly (engine rebuild storm, no "
            "frames received for N seconds, model confidence falling "
            "below floor, etc.). Event-shaped but treated as telemetry "
            "because it carries operator-visible health signal."
        ),
        payload_keys=("warning_code", "warning_message", "since_seconds"),
        idempotency_key_pattern=(
            "sess:<session_id>:processor.scene_quality_warning:<warning_code>:<bucket_id>"
        ),
        source_signal="package validation or processor heuristics",
    ),
)


# Quick-access tuples by status (handy for tests + docs links). Kept
# derived rather than duplicated so a row's status edit propagates.

API_SUPPORTED_TYPES: tuple[str, ...] = tuple(
    spec.type for spec in INITIAL_TAXONOMY if spec.status == "api_supported"
)

PROPOSED_FIRST_API_ADD_TYPES: tuple[str, ...] = tuple(
    spec.type for spec in INITIAL_TAXONOMY if spec.status == "proposed_first_api_add"
)

DEFERRED_TYPES: tuple[str, ...] = tuple(
    spec.type for spec in INITIAL_TAXONOMY if spec.status == "deferred"
)


def spec_for(observation_type: str) -> ObservationSpec | None:
    """Look up the spec for a candidate observation type. Returns
    ``None`` if the type is not in the taxonomy (i.e. someone made it
    up without adding a row — usually a bug in the caller). Linear
    scan; the taxonomy is < 20 rows and isn't on a hot path."""
    for spec in INITIAL_TAXONOMY:
        if spec.type == observation_type:
            return spec
    return None


def types_in_category(category: ObservationCategory) -> tuple[str, ...]:
    """All taxonomy types for one emission-policy category. Useful for
    tests and for the docs example that lists 'all event types'."""
    return tuple(spec.type for spec in INITIAL_TAXONOMY if spec.category == category)
