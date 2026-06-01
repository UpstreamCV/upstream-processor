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
