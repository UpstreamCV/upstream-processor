"""P4-A.5 — taxonomy invariants.

These tests are the contract for the taxonomy table itself. They are
intentionally strict so a future careless edit to
``observation_taxonomy.py`` (e.g. adding a row but forgetting the
idempotency key, or accidentally flipping a deferred type to
``api_supported``) fails fast.

The taxonomy is the single source of truth that ``docs/architecture.md``
links to; these tests keep the two in sync."""

from __future__ import annotations

import re

from app.observations.observation_adapter import (
    CANDIDATE_LINE_CROSSING_TYPE,
    CANDIDATE_ROI_OCCUPANCY_TYPE,
)
from app.observations.observation_taxonomy import (
    API_SUPPORTED_TYPES,
    DEFERRED_TYPES,
    INITIAL_TAXONOMY,
    PROPOSED_FIRST_API_ADD_TYPES,
    ObservationCategory,
    ObservationStatus,
    spec_for,
    types_in_category,
)
from app.observations.schemas import OBSERVATION_TYPES

# --- shape / uniqueness invariants -------------------------------------


def test_taxonomy_is_non_empty() -> None:
    """A zero-row taxonomy means the docs lost the source of truth."""
    assert len(INITIAL_TAXONOMY) > 0


def test_every_type_string_is_unique() -> None:
    """No duplicates — a future row addition must use a distinct type."""
    types = [spec.type for spec in INITIAL_TAXONOMY]
    assert len(types) == len(set(types)), f"duplicates in taxonomy: {types}"


def test_every_type_uses_processor_namespace() -> None:
    """Initial taxonomy is internal-only: ``processor.<thing>``.
    Customer-facing names (``person.crossed_line`` etc.) come later."""
    for spec in INITIAL_TAXONOMY:
        assert spec.type.startswith("processor."), spec.type


def test_every_category_is_valid() -> None:
    """``ObservationCategory`` Literal must cover every row's category."""
    allowed: set[ObservationCategory] = {"event", "state", "telemetry"}
    for spec in INITIAL_TAXONOMY:
        assert spec.category in allowed, (spec.type, spec.category)


def test_every_status_is_valid() -> None:
    allowed: set[ObservationStatus] = {
        "api_supported",
        "proposed_first_api_add",
        "deferred",
    }
    for spec in INITIAL_TAXONOMY:
        assert spec.status in allowed, (spec.type, spec.status)


# --- P4-A.5's load-bearing decision ------------------------------------


def test_at_most_one_type_proposed_for_first_api_add() -> None:
    """The processing layer widens the api-server one type at a time;
    multiple ``proposed_first_api_add`` rows would mean a future
    milestone is doing more than one thing. After P4-B the count is
    zero (processor.line_crossing flipped to ``api_supported``); the
    next P4-C milestone may re-populate the slot with the next
    candidate (likely ``processor.roi_occupancy``)."""
    assert len(PROPOSED_FIRST_API_ADD_TYPES) <= 1, (
        "the processing layer adds one type to the api-server at a "
        f"time; got {list(PROPOSED_FIRST_API_ADD_TYPES)!r}"
    )


def test_line_crossing_is_now_api_supported() -> None:
    """P4-B flipped ``processor.line_crossing`` from
    ``proposed_first_api_add`` to ``api_supported`` in lock-step with
    the api-server widening. This test is the post-P4-B state lock —
    a future edit that demotes the type fails here."""
    assert "processor.line_crossing" in API_SUPPORTED_TYPES
    assert "processor.line_crossing" not in PROPOSED_FIRST_API_ADD_TYPES
    assert "processor.line_crossing" not in DEFERRED_TYPES


def test_no_proposed_type_is_already_supported() -> None:
    """``proposed_first_api_add`` and ``api_supported`` are mutually
    exclusive — once a type is supported it must be flipped to
    ``api_supported`` (P4-B's first step)."""
    proposed = set(PROPOSED_FIRST_API_ADD_TYPES)
    supported = set(API_SUPPORTED_TYPES)
    assert proposed.isdisjoint(supported)


def test_no_processor_type_is_already_in_observation_types() -> None:
    """The ``processor.*`` taxonomy lives in this file; the strict
    ``Literal`` whitelist in ``schemas.OBSERVATION_TYPES`` is owned by
    the deployed api-server. They are deliberately out of sync until
    P4-B. (Today the schema whitelist only contains the three mock
    types.) If a taxonomy entry ever shows up in the schema by
    accident, P4-A's "candidate types not in whitelist" regression
    will fire — this is the matching guard from the taxonomy side."""
    for spec in INITIAL_TAXONOMY:
        if spec.status == "api_supported":
            # Tightening this check would require a second source of
            # truth that the test runs against the real deployed
            # whitelist; for now just guarantee the local whitelist
            # is in lock-step with the taxonomy.
            assert spec.type in OBSERVATION_TYPES, (
                f"{spec.type} is api_supported in taxonomy but missing from "
                "schemas.OBSERVATION_TYPES — they must widen together"
            )
        else:
            assert spec.type not in OBSERVATION_TYPES, (
                f"{spec.type} is in schemas.OBSERVATION_TYPES but not "
                f"marked api_supported in taxonomy (status={spec.status})"
            )


# --- alignment with the P4-A adapter -----------------------------------


def test_adapter_candidate_types_present_in_taxonomy() -> None:
    """P4-A's adapter has two ``CANDIDATE_*`` constants; both must
    have rows in the taxonomy so anyone reading them can find the
    classification + idempotency key pattern."""
    types = {spec.type for spec in INITIAL_TAXONOMY}
    assert CANDIDATE_LINE_CROSSING_TYPE in types
    assert CANDIDATE_ROI_OCCUPANCY_TYPE in types


def test_line_crossing_is_event_category() -> None:
    spec = spec_for(CANDIDATE_LINE_CROSSING_TYPE)
    assert spec is not None
    assert spec.category == "event"
    assert spec.requires_line_geometry is True
    assert spec.requires_tracker_id is False


def test_roi_occupancy_is_state_category() -> None:
    spec = spec_for(CANDIDATE_ROI_OCCUPANCY_TYPE)
    assert spec is not None
    assert spec.category == "state"


# --- idempotency key patterns are well-formed --------------------------


_KEY_PREFIX_RE = re.compile(r"^sess:<session_id>:processor\.[a-z_]+:")


def test_every_idempotency_key_starts_with_session_pattern() -> None:
    """All keys share the same prefix so log scraping / DB indexing
    works uniformly."""
    for spec in INITIAL_TAXONOMY:
        assert _KEY_PREFIX_RE.match(spec.idempotency_key_pattern), (
            f"{spec.type} idempotency key does not start with "
            f"'sess:<session_id>:{spec.type}:' — got "
            f"{spec.idempotency_key_pattern!r}"
        )


def test_event_keys_include_a_uniqueness_token() -> None:
    """Event keys must encode WHICH event (cumulative count, sequence
    number, threshold, …) so two events of the same type on the same
    line/zone get distinct keys."""
    for spec in INITIAL_TAXONOMY:
        if spec.category != "event":
            continue
        tail = spec.idempotency_key_pattern.split(f"processor.{spec.type.split('.', 1)[1]}:", 1)[1]
        # Tail must reference at least one placeholder that disambiguates
        # repeated events — typically <cumulative_count>, <enter_sequence>,
        # <exit_sequence>, <dwell_session_id>, or <threshold_seconds>.
        uniqueness_tokens = (
            "<cumulative_count>",
            "<enter_sequence>",
            "<exit_sequence>",
            "<sequence>",
            "<dwell_session_id>",
            "<threshold_seconds>",
            "<bucket_id>",
        )
        assert any(tok in tail for tok in uniqueness_tokens), (
            f"event {spec.type} idempotency key lacks a uniqueness token: "
            f"{spec.idempotency_key_pattern!r}"
        )


def test_state_keys_use_a_sequence_not_frame_num() -> None:
    """State observations must NOT key on raw frame numbers (would
    balloon the dedupe set). Sequence numbers / bucket ids only."""
    for spec in INITIAL_TAXONOMY:
        if spec.category != "state":
            continue
        assert "<frame_num>" not in spec.idempotency_key_pattern, (
            f"{spec.type} state idempotency key uses <frame_num>; use "
            "<sequence> or a bucket id instead"
        )


# --- requirements consistency ------------------------------------------


def test_zone_dependent_types_require_zone_geometry() -> None:
    """zone_entered / zone_exited / dwell_exceeded all need zone
    geometry to even define an event."""
    for type_name in (
        "processor.zone_entered",
        "processor.zone_exited",
        "processor.dwell_exceeded",
    ):
        spec = spec_for(type_name)
        assert spec is not None
        assert spec.requires_zone_geometry is True, type_name
        assert spec.requires_tracker_id is True, type_name


def test_dwell_depends_on_zone_events() -> None:
    """Dwell can only fire once zone_entered / zone_exited are
    implemented; the taxonomy makes that dependency explicit so a
    future agent doesn't try to ship dwell standalone."""
    spec = spec_for("processor.dwell_exceeded")
    assert spec is not None
    assert "processor.zone_entered" in spec.dependencies
    assert "processor.zone_exited" in spec.dependencies


def test_track_types_require_tracker_id() -> None:
    """track_state / track_update / direction_detected all need a
    tracker_id to identify the subject of the observation."""
    for type_name in (
        "processor.track_state",
        "processor.track_update",
        "processor.direction_detected",
    ):
        spec = spec_for(type_name)
        assert spec is not None
        assert spec.requires_tracker_id is True, type_name


def test_no_payload_key_repeats_idempotency_or_source_or_debug() -> None:
    """``payload_keys`` lists the runtime-neutral keys ONLY. The
    universal keys (``idempotency_key``, ``source``, ``debug``) live on
    every type and must not be enumerated per-spec."""
    universals = {"idempotency_key", "source", "debug"}
    for spec in INITIAL_TAXONOMY:
        leaks = universals & set(spec.payload_keys)
        assert not leaks, (
            f"{spec.type} payload_keys redundantly includes universal "
            f"key(s) {leaks}; remove them — they're implicit"
        )


# --- category accessor + lookup ----------------------------------------


def test_types_in_category_partitions_the_taxonomy() -> None:
    """Every taxonomy row appears in exactly one category bucket."""
    events = set(types_in_category("event"))
    states = set(types_in_category("state"))
    telemetry = set(types_in_category("telemetry"))
    all_types = {spec.type for spec in INITIAL_TAXONOMY}
    assert events | states | telemetry == all_types
    assert events.isdisjoint(states)
    assert events.isdisjoint(telemetry)
    assert states.isdisjoint(telemetry)


def test_spec_for_unknown_returns_none() -> None:
    """spec_for is the public lookup; missing types return None (callers
    decide whether that's a bug)."""
    assert spec_for("processor.completely-not-real") is None
    assert spec_for("") is None


def test_deferred_list_covers_everything_not_yet_supported() -> None:
    """API_SUPPORTED + PROPOSED_FIRST_API_ADD + DEFERRED must sum to
    all taxonomy types — no rows can slip into a fourth state."""
    seen = set(API_SUPPORTED_TYPES) | set(PROPOSED_FIRST_API_ADD_TYPES) | set(DEFERRED_TYPES)
    all_types = {spec.type for spec in INITIAL_TAXONOMY}
    assert seen == all_types


# --- description hygiene -----------------------------------------------


def test_every_spec_has_a_non_empty_description() -> None:
    """A row without a description is undocumented and the docs link
    will look empty; force one sentence at minimum."""
    for spec in INITIAL_TAXONOMY:
        assert spec.description.strip(), spec.type
        assert len(spec.description.strip()) > 20, (
            f"{spec.type} description is too terse: {spec.description!r}"
        )


def test_observation_spec_is_frozen() -> None:
    """``ObservationSpec`` is dataclass(frozen=True). A future agent
    must not be able to mutate a spec at runtime."""
    spec = INITIAL_TAXONOMY[0]
    try:
        spec.type = "modified"  # type: ignore[misc]
    except Exception:
        return  # expected: FrozenInstanceError or AttributeError
    raise AssertionError("ObservationSpec is not frozen; mutation succeeded")
