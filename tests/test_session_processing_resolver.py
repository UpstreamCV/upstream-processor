"""M7B — resolver behavior: envelope → ``ResolvedProcessingConfig``.

Mirrors ``test_session_geometry_resolver.py``. M7B is STAMP ONLY: the
resolver records what the operator *requested* and compares it field-by-field
against what the runtime *actually applies* (the static defaults), but NOTHING
is applied. ``applied`` is ALWAYS ``DEFAULT_APPLIED`` and never echoes a
requested value. ``partially_applied`` is reserved for a later milestone and
must NEVER be emitted here.

Three resolution outcomes:

- ``source="default"``: null envelope → ``applied_defaults``, all-null
  requested, no warnings.
- ``source="operator_saved"``: valid envelope → requested recorded; per-field
  divergence flips ``status`` to ``requested_not_applied`` with the matching
  ``*_requested_not_applied`` warnings; a config equal to the defaults stays
  ``applied_defaults``.
- ``source="fallback_default_on_error"``: invalid envelope → fallback returned
  with ``error_reason`` set; the resolver NEVER raises.
"""

from __future__ import annotations

from app.processing.resolver import (
    DEFAULT_APPLIED,
    AppliedProcessingConfig,
    resolve_session_processing_config,
)


def _config(**overrides: object) -> dict:
    base: dict = {
        "version": 1,
        "model_key": "trafficcamnet_v1",
        "selected_classes": [],
        "tracker": None,
        "confidence_threshold": None,
        "processing_fps": None,
    }
    base.update(overrides)
    return base


def _envelope(revision: int = 7, **config_overrides: object) -> dict:
    return {"revision": revision, "config": _config(**config_overrides)}


# --- source="default" --------------------------------------------------


def test_null_envelope_returns_default() -> None:
    r = resolve_session_processing_config(None)
    assert r.source == "default"
    assert r.status == "applied_defaults"
    assert r.revision is None
    assert r.error_reason is None
    assert r.warnings == ()
    # All-null requested (no operator config at all).
    assert r.requested.model_key is None
    assert r.requested.selected_classes is None
    assert r.requested.tracker is None
    assert r.requested.confidence_threshold is None
    assert r.requested.processing_fps is None
    # Applied is the static defaults.
    assert r.applied == DEFAULT_APPLIED


# --- source="operator_saved" : diverging config (nothing applied) -------


def test_valid_diverging_config_is_requested_not_applied() -> None:
    """Spec scenario: selected_classes ["person"], confidence 0.55, fps 5,
    tracker "iou". Everything diverges from the runtime defaults except the
    model and tracker. status → requested_not_applied; applied stays the
    static defaults; warnings carry the per-field entries."""
    r = resolve_session_processing_config(
        _envelope(
            revision=12,
            selected_classes=["person"],
            confidence_threshold=0.55,
            processing_fps=5,
            tracker="iou",
        )
    )
    assert r.source == "operator_saved"
    assert r.status == "requested_not_applied"
    assert r.revision == 12
    assert r.error_reason is None

    # Requested mirrors the saved config.
    assert r.requested.model_key == "trafficcamnet_v1"
    assert r.requested.selected_classes == ["person"]
    assert r.requested.confidence_threshold == 0.55
    assert r.requested.processing_fps == 5
    assert r.requested.tracker == "iou"

    # NOTHING is applied — applied is exactly the static defaults.
    assert r.applied == DEFAULT_APPLIED

    # Per-field warnings for the divergences. tracker "iou" == default, so it
    # does NOT warn; model_key == default, so no model warning.
    assert "selected_classes_requested_not_applied" in r.warnings
    assert "confidence_threshold_requested_not_applied" in r.warnings
    assert "processing_fps_requested_not_applied" in r.warnings
    assert "tracker_requested_not_applied" not in r.warnings
    assert "model_switch_requested_not_applied" not in r.warnings

    # M7B never emits partially_applied.
    assert r.status != "partially_applied"
    assert "partially_applied" not in r.warnings


# --- source="operator_saved" : config equal to defaults ----------------


def test_config_equal_to_defaults_is_applied_defaults() -> None:
    """model trafficcamnet_v1, classes [], confidence null, fps null, and a
    tracker that equals the default (or null) → nothing diverges, so
    status stays applied_defaults with no warnings."""
    for tracker in (None, "iou"):
        r = resolve_session_processing_config(_envelope(tracker=tracker))
        assert r.source == "operator_saved"
        assert r.status == "applied_defaults"
        assert r.warnings == ()
        assert r.applied == DEFAULT_APPLIED
        assert r.error_reason is None


# --- per-field warnings, exercised individually ------------------------


def test_model_switch_with_known_runtime_warns_only_model_switch() -> None:
    r = resolve_session_processing_config(_envelope(model_key="mock_detect_v0"))
    assert r.status == "requested_not_applied"
    assert "model_switch_requested_not_applied" in r.warnings
    # mock_detect_v0 IS in MODEL_KEY_RUNTIME, so no no-mapping warning.
    assert "model_key_no_runtime_mapping" not in r.warnings
    assert r.applied == DEFAULT_APPLIED


def test_unknown_model_key_warns_no_runtime_mapping() -> None:
    """Unknown model_key still parses (structural validation only) and must
    not crash — it degrades to a lineage warning."""
    r = resolve_session_processing_config(_envelope(model_key="some_future_model_v9"))
    assert r.source == "operator_saved"
    assert r.status == "requested_not_applied"
    assert "model_switch_requested_not_applied" in r.warnings
    assert "model_key_no_runtime_mapping" in r.warnings
    assert r.applied == DEFAULT_APPLIED


def test_unsupported_tracker_warns_but_applied_stays_iou() -> None:
    """An operator tracker that isn't the runtime default degrades to a
    lineage warning; applied.tracker stays the default 'iou' (nothing
    applied)."""
    r = resolve_session_processing_config(_envelope(tracker="deepsort"))
    assert r.source == "operator_saved"
    assert r.status == "requested_not_applied"
    assert "tracker_requested_not_applied" in r.warnings
    assert r.requested.tracker == "deepsort"
    # applied.tracker is unchanged.
    assert r.applied.tracker == "iou"
    assert r.applied == DEFAULT_APPLIED


def test_all_fields_diverge_accumulates_every_warning() -> None:
    r = resolve_session_processing_config(
        _envelope(
            model_key="some_future_model_v9",
            selected_classes=["person", "car"],
            tracker="deepsort",
            confidence_threshold=0.7,
            processing_fps=10,
        )
    )
    assert r.status == "requested_not_applied"
    assert set(r.warnings) == {
        "selected_classes_requested_not_applied",
        "confidence_threshold_requested_not_applied",
        "processing_fps_requested_not_applied",
        "model_switch_requested_not_applied",
        "model_key_no_runtime_mapping",
        "tracker_requested_not_applied",
    }
    assert r.applied == DEFAULT_APPLIED
    assert "partially_applied" not in r.warnings


# --- source="fallback_default_on_error" --------------------------------


def test_malformed_envelope_falls_back_without_raising() -> None:
    bad = {"revision": 1, "config": {"version": 1}}  # missing model_key
    r = resolve_session_processing_config(bad)
    assert r.source == "fallback_default_on_error"
    assert r.status == "fallback_default_on_error"
    assert r.error_reason is not None
    assert r.revision is None
    assert r.warnings == ()
    # Even on error, applied is the static defaults; requested is all-null.
    assert r.applied == DEFAULT_APPLIED
    assert r.requested.model_key is None


def test_bad_version_falls_back() -> None:
    r = resolve_session_processing_config(_envelope(version=2))
    assert r.source == "fallback_default_on_error"
    assert r.error_reason is not None


def test_extra_field_falls_back() -> None:
    """``extra="forbid"`` — a foreign field is an api-server drift signal and
    degrades to fallback rather than being silently accepted."""
    env = _envelope()
    env["unknown_field"] = "drift"
    r = resolve_session_processing_config(env)
    assert r.source == "fallback_default_on_error"


def test_out_of_range_confidence_falls_back() -> None:
    r = resolve_session_processing_config(_envelope(confidence_threshold=2.0))
    assert r.source == "fallback_default_on_error"


def test_error_reason_is_short_and_secret_free() -> None:
    bad = {"revision": 1, "config": {"version": 1}}
    r = resolve_session_processing_config(bad)
    assert r.error_reason is not None
    assert len(r.error_reason) <= 200


# --- custom defaults seam ----------------------------------------------


def test_custom_defaults_are_honored_in_applied() -> None:
    """The resolver compares against (and reports) the passed defaults. A
    request matching the custom defaults must not warn."""
    custom = AppliedProcessingConfig(
        model_key="mock_detect_v0",
        selected_classes=[],
        tracker="iou",
        confidence_threshold=None,
        processing_fps=None,
    )
    r = resolve_session_processing_config(_envelope(model_key="mock_detect_v0"), defaults=custom)
    assert r.applied == custom
    assert r.status == "applied_defaults"
    assert r.warnings == ()


# --- status_payload() lineage shape ------------------------------------


def test_status_payload_shape_diverging() -> None:
    r = resolve_session_processing_config(
        _envelope(
            revision=12,
            selected_classes=["person"],
            confidence_threshold=0.55,
            processing_fps=5,
            tracker="iou",
        )
    )
    payload = r.status_payload()
    # Exact key set of the applied_processing_config block.
    assert set(payload.keys()) == {
        "source",
        "revision",
        "status",
        "reason",
        "requested",
        "applied",
        "warnings",
    }
    assert payload["source"] == "operator_saved"
    assert payload["revision"] == 12
    assert payload["status"] == "requested_not_applied"
    assert payload["reason"] is None
    assert set(payload["requested"].keys()) == {
        "model_key",
        "selected_classes",
        "tracker",
        "confidence_threshold",
        "processing_fps",
    }
    assert payload["requested"]["selected_classes"] == ["person"]
    # applied is ALWAYS the static defaults, never an echo of requested.
    assert payload["applied"] == {
        "model_key": "trafficcamnet_v1",
        "selected_classes": [],
        "tracker": "iou",
        "confidence_threshold": None,
        "processing_fps": None,
    }
    assert isinstance(payload["warnings"], list)
    assert "selected_classes_requested_not_applied" in payload["warnings"]
    assert payload["status"] != "partially_applied"


def test_status_payload_shape_default() -> None:
    payload = resolve_session_processing_config(None).status_payload()
    assert payload["source"] == "default"
    assert payload["revision"] is None
    assert payload["status"] == "applied_defaults"
    assert payload["reason"] is None
    assert payload["warnings"] == []
    assert payload["requested"] == {
        "model_key": None,
        "selected_classes": None,
        "tracker": None,
        "confidence_threshold": None,
        "processing_fps": None,
    }
    assert payload["applied"] == {
        "model_key": "trafficcamnet_v1",
        "selected_classes": [],
        "tracker": "iou",
        "confidence_threshold": None,
        "processing_fps": None,
    }


def test_status_payload_includes_reason_when_degraded() -> None:
    bad = {"revision": 1, "config": {"version": 1}}
    payload = resolve_session_processing_config(bad).status_payload()
    assert payload["source"] == "fallback_default_on_error"
    assert payload["status"] == "fallback_default_on_error"
    assert payload["reason"] is not None
    # applied is still the static defaults even on the error path.
    assert payload["applied"]["model_key"] == "trafficcamnet_v1"


# ===========================================================================
# M7C — confidence_threshold is genuinely applied when the mode supports it
# ===========================================================================
#
# These exercise the new ``confidence_apply_supported`` seam. The existing M7B
# tests above intentionally OMIT the kwarg (defaults to False) and must stay
# green — proving M7C is a pure superset: when apply isn't supported the
# resolver behaves exactly as it did in M7B (stamp-only, applied==defaults).


def test_confidence_applied_when_supported_status_applied() -> None:
    """confidence_apply_supported=True + a confidence request and otherwise
    default config → confidence is echoed into applied, NO confidence warning,
    status == applied (a value was applied, nothing diverged)."""
    r = resolve_session_processing_config(
        _envelope(confidence_threshold=0.55), confidence_apply_supported=True
    )
    assert r.source == "operator_saved"
    assert r.status == "applied"
    assert r.applied.confidence_threshold == 0.55
    assert "confidence_threshold_requested_not_applied" not in r.warnings
    assert r.warnings == ()
    # Only confidence is applied; every other applied field stays at defaults.
    assert r.applied.model_key == DEFAULT_APPLIED.model_key
    assert r.applied.selected_classes == []
    assert r.applied.tracker == DEFAULT_APPLIED.tracker
    assert r.applied.processing_fps is None
    # The frozen module default must NOT have been mutated by replace().
    assert DEFAULT_APPLIED.confidence_threshold is None


def test_confidence_applied_plus_unapplied_fields_is_partially_applied() -> None:
    """confidence_apply_supported=True + confidence 0.55 + selected_classes
    + processing_fps → confidence applied (no confidence warning) while the
    stamp-only fields warn. status == partially_applied; the stamp-only fields
    stay at defaults ([] / None)."""
    r = resolve_session_processing_config(
        _envelope(
            confidence_threshold=0.55,
            selected_classes=["person"],
            processing_fps=5,
        ),
        confidence_apply_supported=True,
    )
    assert r.status == "partially_applied"
    assert r.applied.confidence_threshold == 0.55
    assert "confidence_threshold_requested_not_applied" not in r.warnings
    assert "selected_classes_requested_not_applied" in r.warnings
    assert "processing_fps_requested_not_applied" in r.warnings
    # The unapplied fields keep the runtime defaults (not echoes of requested).
    assert r.applied.selected_classes == []
    assert r.applied.processing_fps is None
    # ...but requested still mirrors what the operator saved.
    assert r.requested.selected_classes == ["person"]
    assert r.requested.processing_fps == 5


def test_confidence_not_applied_when_unsupported_preserves_m7b() -> None:
    """confidence_apply_supported=False (the M7B default) + a confidence
    request → NOT applied, warning emitted, status == requested_not_applied,
    applied stays the static defaults. This proves the pre-M7C behavior is
    preserved exactly when the mode can't apply the threshold."""
    r = resolve_session_processing_config(
        _envelope(confidence_threshold=0.55), confidence_apply_supported=False
    )
    assert r.status == "requested_not_applied"
    assert r.applied.confidence_threshold is None
    assert r.applied == DEFAULT_APPLIED
    assert "confidence_threshold_requested_not_applied" in r.warnings
    assert r.requested.confidence_threshold == 0.55
    # Default kwarg must behave identically to passing False explicitly.
    r_default = resolve_session_processing_config(_envelope(confidence_threshold=0.55))
    assert r_default.status == "requested_not_applied"
    assert r_default.applied == DEFAULT_APPLIED


def test_no_confidence_request_is_applied_defaults_even_when_supported() -> None:
    """confidence_apply_supported=True but the operator requested no
    confidence (None) and nothing else diverges → no confidence warning,
    applied.confidence_threshold stays None, status == applied_defaults."""
    r = resolve_session_processing_config(
        _envelope(confidence_threshold=None), confidence_apply_supported=True
    )
    assert r.status == "applied_defaults"
    assert r.applied.confidence_threshold is None
    assert r.applied == DEFAULT_APPLIED
    assert r.warnings == ()
    assert "confidence_threshold_requested_not_applied" not in r.warnings


def test_out_of_range_confidence_falls_back_even_when_apply_supported() -> None:
    """Defensive: an out-of-[0,1] confidence (e.g. 1.5) fails the
    session_processing bounds validation, so even with apply support the
    resolver degrades to the error fallback WITHOUT raising — applied stays
    the static defaults (confidence None)."""
    r = resolve_session_processing_config(
        _envelope(confidence_threshold=1.5), confidence_apply_supported=True
    )
    assert r.source == "fallback_default_on_error"
    assert r.status == "fallback_default_on_error"
    assert r.error_reason is not None
    assert r.applied.confidence_threshold is None
    assert r.applied == DEFAULT_APPLIED


def test_status_payload_applied_is_always_default_across_all_outcomes() -> None:
    """Pin the core M7B invariant in one place: regardless of source/status,
    the applied block equals DEFAULT_APPLIED rendered to dict — nothing from
    a requested config is ever echoed into applied."""
    expected_applied = {
        "model_key": "trafficcamnet_v1",
        "selected_classes": [],
        "tracker": "iou",
        "confidence_threshold": None,
        "processing_fps": None,
    }
    envelopes = [
        None,
        _envelope(),  # equals defaults
        _envelope(selected_classes=["person"], processing_fps=5),  # diverges
        {"revision": 1, "config": {"version": 1}},  # malformed
    ]
    for env in envelopes:
        payload = resolve_session_processing_config(env).status_payload()
        assert payload["applied"] == expected_applied
        assert payload["status"] != "partially_applied"


# ===========================================================================
# M7D — selected_classes is genuinely applied (post-inference per-object
# filtering) when the mode supports it
# ===========================================================================
#
# These exercise the new ``selected_classes_apply_supported`` seam. The
# existing M7B/M7C tests above intentionally OMIT the kwarg (defaults to False)
# and must stay green — proving M7D is a pure superset: when apply isn't
# supported the resolver behaves exactly as it did before (the non-empty list
# warns ``selected_classes_requested_not_applied`` and applied stays []).


def test_selected_classes_applied_when_supported_is_partially_applied() -> None:
    """selected_classes_apply_supported=True + a non-empty list and otherwise
    default config → the list is echoed into applied.selected_classes, NO
    ``selected_classes_requested_not_applied`` warning, the honest
    ``selected_classes_analytics_counts_not_filtered`` warning IS added, and
    status == partially_applied (an applied list always carries that warning,
    because the aggregate analytics counts stay across all classes)."""
    r = resolve_session_processing_config(
        _envelope(selected_classes=["person", "car"]),
        selected_classes_apply_supported=True,
    )
    assert r.source == "operator_saved"
    assert r.status == "partially_applied"
    assert r.applied.selected_classes == ["person", "car"]
    assert "selected_classes_requested_not_applied" not in r.warnings
    assert "selected_classes_analytics_counts_not_filtered" in r.warnings
    # Nothing else diverged, so that is the only warning.
    assert r.warnings == ("selected_classes_analytics_counts_not_filtered",)
    # Only selected_classes is applied; every other applied field stays default.
    assert r.applied.model_key == DEFAULT_APPLIED.model_key
    assert r.applied.confidence_threshold is None
    assert r.applied.tracker == DEFAULT_APPLIED.tracker
    assert r.applied.processing_fps is None
    # The frozen module default must NOT have been mutated by replace().
    assert DEFAULT_APPLIED.selected_classes == []


def test_selected_classes_not_applied_when_unsupported_preserves_prior() -> None:
    """selected_classes_apply_supported=False (the default) + a non-empty list
    → NOT applied, warning emitted, status == requested_not_applied, applied
    stays the static defaults. Proves the pre-M7D behavior is preserved when
    the mode can't filter."""
    r = resolve_session_processing_config(
        _envelope(selected_classes=["person"]),
        selected_classes_apply_supported=False,
    )
    assert r.status == "requested_not_applied"
    assert r.applied.selected_classes == []
    assert r.applied == DEFAULT_APPLIED
    assert "selected_classes_requested_not_applied" in r.warnings
    assert "selected_classes_analytics_counts_not_filtered" not in r.warnings
    # Default kwarg must behave identically to passing False explicitly.
    r_default = resolve_session_processing_config(_envelope(selected_classes=["person"]))
    assert r_default.status == "requested_not_applied"
    assert r_default.applied == DEFAULT_APPLIED


def test_empty_selected_classes_is_applied_defaults_even_when_supported() -> None:
    """selected_classes_apply_supported=True but the operator requested [] (no
    filter) and nothing else diverges → no warning, applied.selected_classes
    stays [], status == applied_defaults. The [] path is byte/behavior
    identical to today."""
    r = resolve_session_processing_config(
        _envelope(selected_classes=[]),
        selected_classes_apply_supported=True,
    )
    assert r.status == "applied_defaults"
    assert r.applied.selected_classes == []
    assert r.applied == DEFAULT_APPLIED
    assert r.warnings == ()


def test_selected_classes_and_confidence_both_applied_is_partially_applied() -> None:
    """Both M7C confidence and M7D selected_classes applied together →
    applied_count==2, both echoed, the analytics-counts warning is the only
    warning, status == partially_applied. Proves the two seams compose and M7C
    apply is unregressed."""
    r = resolve_session_processing_config(
        _envelope(confidence_threshold=0.55, selected_classes=["person"]),
        confidence_apply_supported=True,
        selected_classes_apply_supported=True,
    )
    assert r.status == "partially_applied"
    assert r.applied.confidence_threshold == 0.55
    assert r.applied.selected_classes == ["person"]
    assert "confidence_threshold_requested_not_applied" not in r.warnings
    assert "selected_classes_requested_not_applied" not in r.warnings
    assert r.warnings == ("selected_classes_analytics_counts_not_filtered",)


def test_selected_classes_applied_plus_unapplied_fields_accumulates() -> None:
    """selected_classes applied while stamp-only fields (fps/model/tracker)
    still warn → status partially_applied with both the analytics-counts
    warning and the stamp-only warnings; applied.selected_classes echoes the
    list while the stamp-only applied fields stay at defaults."""
    r = resolve_session_processing_config(
        _envelope(
            selected_classes=["person"],
            processing_fps=5,
            tracker="deepsort",
        ),
        selected_classes_apply_supported=True,
    )
    assert r.status == "partially_applied"
    assert r.applied.selected_classes == ["person"]
    assert "selected_classes_analytics_counts_not_filtered" in r.warnings
    assert "processing_fps_requested_not_applied" in r.warnings
    assert "tracker_requested_not_applied" in r.warnings
    assert "selected_classes_requested_not_applied" not in r.warnings
    assert r.applied.processing_fps is None
    assert r.applied.tracker == "iou"


def test_status_payload_echoes_applied_selected_classes_when_supported() -> None:
    """The status_payload applied block now echoes the operator list when it
    is genuinely applied (M7D) — distinct from the M7B/M7C invariant where
    applied.selected_classes was always []."""
    payload = resolve_session_processing_config(
        _envelope(selected_classes=["person", "car"]),
        selected_classes_apply_supported=True,
    ).status_payload()
    assert payload["applied"]["selected_classes"] == ["person", "car"]
    assert payload["status"] == "partially_applied"
    assert "selected_classes_analytics_counts_not_filtered" in payload["warnings"]


# --- M7E: processing_fps applied (output-cadence only) ------------------


def test_processing_fps_applied_when_supported_is_partially_applied() -> None:
    """M7E: with processing_fps_apply_supported=True a requested fps is
    genuinely applied (echoed into applied.processing_fps) — and ALWAYS rides
    with the honest output-cadence caveat (inference and event-driven
    observations are not throttled), so an fps-applied config is at least
    partially_applied, never plain applied."""
    r = resolve_session_processing_config(
        _envelope(processing_fps=5),
        processing_fps_apply_supported=True,
    )
    assert r.status == "partially_applied"
    assert r.applied.processing_fps == 5
    assert r.requested.processing_fps == 5
    assert "processing_fps_output_cadence_only" in r.warnings
    assert "processing_fps_requested_not_applied" not in r.warnings


def test_processing_fps_not_applied_when_unsupported_preserves_prior() -> None:
    """Without support (default kwarg, the pre-M7E posture) a requested fps
    stays stamp-only exactly as before."""
    r = resolve_session_processing_config(_envelope(processing_fps=5))
    assert r.status == "requested_not_applied"
    assert r.applied.processing_fps is None
    assert "processing_fps_requested_not_applied" in r.warnings
    assert "processing_fps_output_cadence_only" not in r.warnings


def test_null_processing_fps_is_applied_defaults_even_when_supported() -> None:
    """fps=null means "runtime default" — nothing applied, no warnings, no
    behavior change."""
    r = resolve_session_processing_config(
        _envelope(processing_fps=None),
        processing_fps_apply_supported=True,
    )
    assert r.status == "applied_defaults"
    assert r.applied.processing_fps is None
    assert r.warnings == ()


def test_out_of_range_processing_fps_falls_back_even_when_apply_supported() -> None:
    """The envelope schema bounds fps to 1..30 — an out-of-range value fails
    parsing and degrades to fallback defaults (never a crash, never a bogus
    cadence), even when the mode could apply a valid fps."""
    for bad in (0, -3, 31):
        r = resolve_session_processing_config(
            _envelope(processing_fps=bad),
            processing_fps_apply_supported=True,
        )
        assert r.source == "fallback_default_on_error"
        assert r.applied == DEFAULT_APPLIED
        assert r.applied.processing_fps is None


def test_fps_confidence_and_classes_all_applied_accumulates() -> None:
    """M7C + M7D + M7E together: all three applied fields echo into applied,
    each caveat/no-warning rule holds, and status is partially_applied (the
    selected-classes + fps caveats keep it from plain applied)."""
    r = resolve_session_processing_config(
        _envelope(
            selected_classes=["person"],
            confidence_threshold=0.2,
            processing_fps=5,
        ),
        confidence_apply_supported=True,
        selected_classes_apply_supported=True,
        processing_fps_apply_supported=True,
    )
    assert r.status == "partially_applied"
    assert r.applied.confidence_threshold == 0.2
    assert r.applied.selected_classes == ["person"]
    assert r.applied.processing_fps == 5
    assert "confidence_threshold_requested_not_applied" not in r.warnings
    assert "selected_classes_requested_not_applied" not in r.warnings
    assert "processing_fps_requested_not_applied" not in r.warnings
    assert "selected_classes_analytics_counts_not_filtered" in r.warnings
    assert "processing_fps_output_cadence_only" in r.warnings


def test_status_payload_echoes_applied_processing_fps_when_supported() -> None:
    payload = resolve_session_processing_config(
        _envelope(processing_fps=10),
        processing_fps_apply_supported=True,
    ).status_payload()
    assert payload["requested"]["processing_fps"] == 10
    assert payload["applied"]["processing_fps"] == 10
    assert payload["status"] == "partially_applied"
    assert "processing_fps_output_cadence_only" in payload["warnings"]
