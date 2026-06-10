"""M7B — DeepStream worker processing-config resolution + lineage stamping.

Mirrors ``test_worker_geometry_resolution.py``. Covers two seams:

1. ``EligibleSession`` envelope tolerance — the wire field accepts ``null``
   AND a full ``{revision, config}`` envelope, with ``extra="ignore"`` so a
   forward-compatible api-server addition doesn't break parsing.
2. The worker resolves ``processing_config`` (STAMP ONLY) into
   ``_resolved_processing_config`` and ``_build(status)`` injects an
   ``applied_processing_config`` block into the status payload — with
   ``applied`` ALWAYS the runtime defaults and NO ``error_code`` even when the
   config diverges or is malformed (unlike geometry's ``geometry_invalid``).

The worker lifecycle is NOT started — we only inspect ``__init__`` resolution
and ``_build``.
"""

from __future__ import annotations

from app.observations.schemas import EligibleSession
from app.packages.resolver import resolve
from app.processing.resolver import DEFAULT_APPLIED
from app.runtimes.deepstream_runtime.metadata_probe import (
    DeepStreamMetadataProcessManager,
)
from app.runtimes.deepstream_runtime.worker import DeepStreamWorker

from tests._fakes import FakeApi
from tests.conftest import eligible_session_dict

_DEFAULT_APPLIED_DICT = {
    "model_key": "trafficcamnet_v1",
    "selected_classes": [],
    "tracker": "iou",
    "confidence_threshold": None,
    "processing_fps": None,
}


def _session(processing_config: dict | None = None) -> EligibleSession:
    return EligibleSession.model_validate(
        eligible_session_dict(processing_config=processing_config)
    )


def _envelope(*, revision: int = 4, **config_overrides: object) -> dict:
    config: dict = {
        "version": 1,
        "model_key": "trafficcamnet_v1",
        "selected_classes": [],
        "tracker": None,
        "confidence_threshold": None,
        "processing_fps": None,
    }
    config.update(config_overrides)
    return {"revision": revision, "config": config}


def _analytics_worker(session: EligibleSession, settings) -> DeepStreamWorker:  # type: ignore[no-untyped-def]
    return DeepStreamWorker(
        session=session,
        settings=settings,
        api=FakeApi(),
        package=resolve("deepstream.analytics_probe.v0"),
    )


def _worker(session: EligibleSession, settings, package_id: str) -> DeepStreamWorker:  # type: ignore[no-untyped-def]
    return DeepStreamWorker(
        session=session,
        settings=settings,
        api=FakeApi(),
        package=resolve(package_id),
    )


# ===========================================================================
# EligibleSession envelope tolerance
# ===========================================================================


def test_eligible_session_accepts_null_processing_config() -> None:
    s = EligibleSession.model_validate(eligible_session_dict(processing_config=None))
    assert s.processing_config is None


def test_eligible_session_defaults_processing_config_to_none() -> None:
    """The field is additive — omitting it entirely (older api-server) parses
    fine and defaults to None."""
    raw = eligible_session_dict()
    raw.pop("processing_config", None)
    s = EligibleSession.model_validate(raw)
    assert s.processing_config is None


def test_eligible_session_accepts_full_envelope() -> None:
    env = _envelope(revision=9, selected_classes=["person"], processing_fps=5)
    s = EligibleSession.model_validate(eligible_session_dict(processing_config=env))
    assert s.processing_config == env


def test_eligible_session_tolerates_unknown_nested_keys() -> None:
    """``extra="ignore"`` on EligibleSession — the field is loosely typed
    (``dict | None``), so a forward-compatible api-server addition inside the
    envelope is carried through opaquely (the resolver, not this model, owns
    strict validation)."""
    env = _envelope()
    env["config"]["some_future_field"] = "drift"
    s = EligibleSession.model_validate(eligible_session_dict(processing_config=env))
    assert s.processing_config is not None
    assert s.processing_config["config"]["some_future_field"] == "drift"


# ===========================================================================
# Worker resolution outcomes
# ===========================================================================


def test_null_processing_config_resolves_to_default(fast_settings) -> None:  # type: ignore[no-untyped-def]
    w = _analytics_worker(_session(None), fast_settings)
    assert w._resolved_processing_config.source == "default"
    assert w._resolved_processing_config.status == "applied_defaults"
    assert w._resolved_processing_config.revision is None
    assert w._resolved_processing_config.applied == DEFAULT_APPLIED


def test_diverging_config_resolves_to_requested_not_applied(fast_settings) -> None:  # type: ignore[no-untyped-def]
    # M7C applies confidence, M7D applies selected_classes, and M7E applies
    # processing_fps in analytics mode, so this stamp-only divergence case uses
    # ONLY a field that remains stamp-only in every milestone (tracker). The
    # confidence/selected-classes/fps applied paths are covered explicitly below.
    env = _envelope(revision=7, tracker="deepsort")
    w = _analytics_worker(_session(env), fast_settings)
    rc = w._resolved_processing_config
    assert rc.source == "operator_saved"
    assert rc.status == "requested_not_applied"
    assert rc.revision == 7
    # Nothing applied (tracker remains stamp-only in all milestones).
    assert rc.applied == DEFAULT_APPLIED


def test_malformed_config_resolves_to_fallback_without_raising(fast_settings) -> None:  # type: ignore[no-untyped-def]
    bad = {"revision": 1, "config": {"version": 1}}  # missing model_key
    w = _analytics_worker(_session(bad), fast_settings)
    rc = w._resolved_processing_config
    assert rc.source == "fallback_default_on_error"
    assert rc.error_reason is not None
    assert rc.applied == DEFAULT_APPLIED


# ===========================================================================
# Status payload lineage stamping
# ===========================================================================


def test_status_payload_carries_applied_processing_config(fast_settings) -> None:  # type: ignore[no-untyped-def]
    # Use only stamp-only divergences (model switch + tracker) so applied stays
    # the runtime defaults. (M7D applies selected_classes and M7E applies
    # processing_fps on an analytics worker — see the dedicated tests below.)
    env = _envelope(
        revision=11,
        model_key="some_future_model_v9",
        tracker="deepsort",
    )
    w = _analytics_worker(_session(env), fast_settings)
    status = w._build("attached")
    block = status.payload["applied_processing_config"]
    assert block["source"] == "operator_saved"
    assert block["revision"] == 11
    assert block["status"] == "requested_not_applied"
    # applied stays the runtime defaults — these fields are never applied.
    assert block["applied"] == _DEFAULT_APPLIED_DICT
    assert "model_switch_requested_not_applied" in block["warnings"]
    assert "tracker_requested_not_applied" in block["warnings"]
    # A diverging config must NOT flag a worker error_code.
    assert status.error_code is None


def test_status_payload_carries_applied_selected_classes(fast_settings) -> None:  # type: ignore[no-untyped-def]
    """M7D: on an analytics worker a saved non-empty selected_classes is
    genuinely applied (post-inference per-object filtering), so the status
    payload echoes the operator list into applied.selected_classes, flips
    status to partially_applied, and carries the honest analytics-counts
    warning rather than ``selected_classes_requested_not_applied``."""
    env = _envelope(revision=12, selected_classes=["person", "car"])
    w = _analytics_worker(_session(env), fast_settings)
    status = w._build("attached")
    block = status.payload["applied_processing_config"]
    assert block["source"] == "operator_saved"
    assert block["status"] == "partially_applied"
    assert block["requested"]["selected_classes"] == ["person", "car"]
    assert block["applied"]["selected_classes"] == ["person", "car"]
    assert "selected_classes_analytics_counts_not_filtered" in block["warnings"]
    assert "selected_classes_requested_not_applied" not in block["warnings"]
    assert status.error_code is None
    # The per-worker filter tuple is derived from the resolved APPLIED list.
    assert w._selected_classes == ("person", "car")


def test_status_payload_carries_default_processing_config(fast_settings) -> None:  # type: ignore[no-untyped-def]
    w = _analytics_worker(_session(None), fast_settings)
    status = w._build("attached")
    block = status.payload["applied_processing_config"]
    assert block["source"] == "default"
    assert block["status"] == "applied_defaults"
    assert block["revision"] is None
    assert block["applied"] == _DEFAULT_APPLIED_DICT
    assert status.error_code is None


def test_malformed_config_does_not_set_error_code(fast_settings) -> None:  # type: ignore[no-untyped-def]
    """Unlike geometry (which sets ``error_code="geometry_invalid"`` on a bad
    envelope), a malformed processing config in M7B applies nothing and so
    must NOT flag a worker error — the degradation lives only in the lineage
    block's ``source``/``reason``."""
    bad = {"revision": 3, "config": {"version": 99}}
    w = _analytics_worker(_session(bad), fast_settings)
    status = w._build("attached")
    block = status.payload["applied_processing_config"]
    assert block["source"] == "fallback_default_on_error"
    assert block["reason"] is not None
    assert status.error_code is None
    assert block["applied"] == _DEFAULT_APPLIED_DICT


def test_processing_config_stamped_on_non_analytics_worker(fast_settings) -> None:  # type: ignore[no-untyped-def]
    """``_resolved_processing_config`` is resolved unconditionally (not gated
    on analytics mode, unlike geometry), so the lineage block is stamped on
    every DeepStream worker's status."""
    w = DeepStreamWorker(
        session=_session(_envelope(revision=2, processing_fps=5)),
        settings=fast_settings,
        api=FakeApi(),
        package=resolve("deepstream.metadata_probe.v0"),
    )
    status = w._build("attached")
    assert "applied_processing_config" in status.payload
    assert status.payload["applied_processing_config"]["source"] == "operator_saved"
    assert status.error_code is None


def test_caller_payload_merges_with_applied_processing_config(fast_settings) -> None:  # type: ignore[no-untyped-def]
    """A caller passing extra payload fields must not clobber the
    processing-config lineage block, and vice versa."""
    w = _analytics_worker(_session(_envelope()), fast_settings)
    status = w._build("processing", payload={"fps_processed": 30, "frames_seen": 1234})
    assert status.payload["fps_processed"] == 30
    assert status.payload["frames_seen"] == 1234
    assert "applied_processing_config" in status.payload
    # Geometry lineage still flows alongside it on an analytics worker.
    assert "applied_geometry" in status.payload


# ===========================================================================
# M7C — confidence_threshold flows into the PGIE-rendering process manager
# ===========================================================================


def test_analytics_worker_applies_confidence_in_resolution(fast_settings) -> None:  # type: ignore[no-untyped-def]
    """An analytics worker computes confidence_apply_supported=True, so a
    saved confidence is genuinely applied. With a stamp-only field also
    diverging (tracker) the resolved status is partially_applied and
    applied.confidence_threshold echoes the operator value."""
    env = _envelope(revision=8, confidence_threshold=0.55, tracker="deepsort")
    w = _analytics_worker(_session(env), fast_settings)
    rc = w._resolved_processing_config
    assert rc.source == "operator_saved"
    assert rc.status == "partially_applied"
    assert rc.applied.confidence_threshold == 0.55
    assert "confidence_threshold_requested_not_applied" not in rc.warnings
    assert "tracker_requested_not_applied" in rc.warnings


def test_analytics_worker_applies_selected_classes_in_resolution(fast_settings) -> None:  # type: ignore[no-untyped-def]
    """M7D: an analytics worker computes selected_classes_apply_supported=True,
    so a saved non-empty list is genuinely applied (post-inference per-object
    filtering): applied.selected_classes echoes the list, the per-worker
    ``_selected_classes`` filter tuple is derived from it, and status is
    partially_applied (the honest analytics-counts warning always rides along).
    M7C confidence apply is unregressed when both are present."""
    env = _envelope(revision=9, confidence_threshold=0.4, selected_classes=["person"])
    w = _analytics_worker(_session(env), fast_settings)
    rc = w._resolved_processing_config
    assert rc.status == "partially_applied"
    assert rc.applied.selected_classes == ["person"]
    assert rc.applied.confidence_threshold == 0.4
    assert "selected_classes_requested_not_applied" not in rc.warnings
    assert "selected_classes_analytics_counts_not_filtered" in rc.warnings
    assert w._selected_classes == ("person",)


def test_decode_worker_does_not_apply_selected_classes(fast_settings) -> None:  # type: ignore[no-untyped-def]
    """Decode mode produces no per-object output paths, so
    selected_classes_apply_supported is False: a saved list is recorded as
    requested_not_applied, applied stays [], and the per-worker filter tuple is
    empty (byte-identical to today)."""
    env = _envelope(revision=5, selected_classes=["person"])
    w = _worker(_session(env), fast_settings, "deepstream.decode_probe.v0")
    rc = w._resolved_processing_config
    assert rc.status == "requested_not_applied"
    assert rc.applied.selected_classes == []
    assert "selected_classes_requested_not_applied" in rc.warnings
    assert "selected_classes_analytics_counts_not_filtered" not in rc.warnings
    assert w._selected_classes == ()


def test_metadata_worker_applies_confidence_clean_status_applied(fast_settings) -> None:  # type: ignore[no-untyped-def]
    """Metadata mode also supports applying confidence. With only confidence
    requested, status is the clean ``applied`` (no warnings)."""
    env = _envelope(revision=3, confidence_threshold=0.55)
    w = _worker(_session(env), fast_settings, "deepstream.metadata_probe.v0")
    rc = w._resolved_processing_config
    assert rc.status == "applied"
    assert rc.applied.confidence_threshold == 0.55
    assert rc.warnings == ()


def test_decode_worker_does_not_apply_confidence(fast_settings) -> None:  # type: ignore[no-untyped-def]
    """Decode mode renders no PGIE config, so confidence_apply_supported is
    False there: the same saved confidence is recorded as
    requested_not_applied and applied stays at the defaults (None)."""
    env = _envelope(revision=3, confidence_threshold=0.55)
    w = _worker(_session(env), fast_settings, "deepstream.decode_probe.v0")
    rc = w._resolved_processing_config
    assert rc.status == "requested_not_applied"
    assert rc.applied.confidence_threshold is None
    assert rc.applied == DEFAULT_APPLIED
    assert "confidence_threshold_requested_not_applied" in rc.warnings


def test_default_pm_factory_forwards_confidence_threshold(fast_settings) -> None:  # type: ignore[no-untyped-def]
    """``_default_pm_factory()`` on an analytics worker must hand the
    DeepStreamMetadataProcessManager the resolved APPLIED confidence (the
    single source of truth the live PGIE config and the stamped lineage
    share), mirroring how the geometry test inspects the forwarded value."""
    env = _envelope(revision=2, confidence_threshold=0.55)
    w = _analytics_worker(_session(env), fast_settings)
    # The applied value is what gets forwarded.
    assert w._resolved_processing_config.applied.confidence_threshold == 0.55
    pm = w._default_pm_factory()
    assert isinstance(pm, DeepStreamMetadataProcessManager)
    assert pm._confidence_threshold == 0.55


def test_default_pm_factory_forwards_none_when_no_override(fast_settings) -> None:  # type: ignore[no-untyped-def]
    """No operator confidence → the PM receives None, preserving the built-in
    pre-cluster-threshold defaults (M7B/byte-identical behavior)."""
    w = _analytics_worker(_session(None), fast_settings)
    pm = w._default_pm_factory()
    assert isinstance(pm, DeepStreamMetadataProcessManager)
    assert pm._confidence_threshold is None


def test_default_pm_factory_forwards_none_when_mode_cannot_apply(fast_settings) -> None:  # type: ignore[no-untyped-def]
    """A tracker-mode worker DOES render a PGIE config and DOES support
    applying confidence, so it forwards the resolved value. (Sanity that the
    PM-facing path matches the resolver gate.)"""
    env = _envelope(revision=4, confidence_threshold=0.4)
    w = _worker(_session(env), fast_settings, "deepstream.tracker_probe.v0")
    assert w._resolved_processing_config.applied.confidence_threshold == 0.4
    pm = w._default_pm_factory()
    assert isinstance(pm, DeepStreamMetadataProcessManager)
    assert pm._confidence_threshold == 0.4


# ===========================================================================
# M7E — processing_fps flows into the perception-snapshot cadence
# ===========================================================================


def test_analytics_worker_applies_processing_fps_in_resolution(fast_settings) -> None:  # type: ignore[no-untyped-def]
    """M7E: an analytics worker (perception snapshots enabled) computes
    processing_fps_apply_supported=True, so a saved fps is genuinely applied:
    applied.processing_fps echoes the request, the honest output-cadence
    caveat rides along, and the per-worker perception interval becomes 1/fps."""
    env = _envelope(revision=13, processing_fps=5)
    w = _analytics_worker(_session(env), fast_settings)
    rc = w._resolved_processing_config
    assert rc.status == "partially_applied"
    assert rc.applied.processing_fps == 5
    assert "processing_fps_output_cadence_only" in rc.warnings
    assert "processing_fps_requested_not_applied" not in rc.warnings
    assert w._perception_interval_seconds == 0.2


def test_analytics_worker_null_fps_keeps_global_interval(fast_settings) -> None:  # type: ignore[no-untyped-def]
    """No fps requested → byte-equivalent pre-M7E behavior: the cadence is the
    global perception_snapshot_interval_seconds and applied stays null."""
    w = _analytics_worker(_session(_envelope(processing_fps=None)), fast_settings)
    assert w._resolved_processing_config.applied.processing_fps is None
    assert w._perception_interval_seconds == float(
        fast_settings.perception_snapshot_interval_seconds
    )


def test_metadata_worker_does_not_apply_processing_fps(fast_settings) -> None:  # type: ignore[no-untyped-def]
    """Non-analytics modes have no perception-snapshot cadence, so fps stays
    stamp-only there (requested_not_applied) and the interval falls back to
    the global setting."""
    env = _envelope(processing_fps=5)
    w = _worker(_session(env), fast_settings, "deepstream.metadata_probe.v0")
    rc = w._resolved_processing_config
    assert rc.applied.processing_fps is None
    assert "processing_fps_requested_not_applied" in rc.warnings
    assert "processing_fps_output_cadence_only" not in rc.warnings
    assert w._perception_interval_seconds == float(
        fast_settings.perception_snapshot_interval_seconds
    )


def test_perception_disabled_does_not_apply_processing_fps(fast_settings) -> None:  # type: ignore[no-untyped-def]
    """Analytics mode with perception snapshots disabled has no cadence seam
    either — fps stays stamp-only and behavior is unchanged."""
    settings = fast_settings.model_copy(update={"perception_snapshot_enabled": False})
    env = _envelope(processing_fps=5)
    w = _analytics_worker(_session(env), settings)
    rc = w._resolved_processing_config
    assert rc.applied.processing_fps is None
    assert "processing_fps_requested_not_applied" in rc.warnings


def test_fps_interval_is_per_worker_no_cross_camera_leakage(fast_settings) -> None:  # type: ignore[no-untyped-def]
    """Two workers with different fps configs each derive their own interval;
    the shared Settings object is never mutated."""
    before = float(fast_settings.perception_snapshot_interval_seconds)
    w5 = _analytics_worker(_session(_envelope(processing_fps=5)), fast_settings)
    w1 = _analytics_worker(_session(_envelope(processing_fps=1)), fast_settings)
    w_none = _analytics_worker(_session(None), fast_settings)
    assert w5._perception_interval_seconds == 0.2
    assert w1._perception_interval_seconds == 1.0
    assert w_none._perception_interval_seconds == before
    assert float(fast_settings.perception_snapshot_interval_seconds) == before


def test_status_payload_carries_applied_processing_fps(fast_settings) -> None:  # type: ignore[no-untyped-def]
    """The M7B->M8 stamping path carries the new fps lineage unchanged: the
    status payload block echoes applied.processing_fps with the caveat, and a
    config combining all three applied fields accumulates honestly."""
    env = _envelope(
        revision=14,
        selected_classes=["person"],
        confidence_threshold=0.2,
        processing_fps=5,
    )
    w = _analytics_worker(_session(env), fast_settings)
    status = w._build("attached")
    block = status.payload["applied_processing_config"]
    assert block["status"] == "partially_applied"
    assert block["requested"]["processing_fps"] == 5
    assert block["applied"]["processing_fps"] == 5
    assert block["applied"]["confidence_threshold"] == 0.2
    assert block["applied"]["selected_classes"] == ["person"]
    assert "processing_fps_output_cadence_only" in block["warnings"]
    assert "selected_classes_analytics_counts_not_filtered" in block["warnings"]
    assert "processing_fps_requested_not_applied" not in block["warnings"]
    assert status.error_code is None
