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
    env = _envelope(
        revision=7, selected_classes=["person"], confidence_threshold=0.55, processing_fps=5
    )
    w = _analytics_worker(_session(env), fast_settings)
    rc = w._resolved_processing_config
    assert rc.source == "operator_saved"
    assert rc.status == "requested_not_applied"
    assert rc.revision == 7
    # Nothing applied.
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
    env = _envelope(revision=11, selected_classes=["person"], processing_fps=5)
    w = _analytics_worker(_session(env), fast_settings)
    status = w._build("attached")
    block = status.payload["applied_processing_config"]
    assert block["source"] == "operator_saved"
    assert block["revision"] == 11
    assert block["status"] == "requested_not_applied"
    assert block["requested"]["selected_classes"] == ["person"]
    # applied is ALWAYS the runtime defaults — never an echo of requested.
    assert block["applied"] == _DEFAULT_APPLIED_DICT
    assert "selected_classes_requested_not_applied" in block["warnings"]
    # STAMP ONLY: a diverging config must NOT flag a worker error_code.
    assert status.error_code is None
    # partially_applied is never emitted in M7B.
    assert block["status"] != "partially_applied"


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
