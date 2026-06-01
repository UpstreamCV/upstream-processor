"""M7B — pydantic validation of the eligible-session processing-config envelope.

Mirrors ``test_session_geometry_models.py``. The processor re-validates the
wire envelope locally (it arrives as an opaque JSON-decoded dict on
``EligibleSession.processing_config``) so a malformed config from a future
api-server drift surfaces as a clean ``ValidationError`` — which the resolver
maps to ``source="fallback_default_on_error"`` — instead of corrupting the
stamped lineage block downstream.

These models are a clone of the api-server's ``CameraProcessingConfig`` /
its envelope; ``extra="forbid"`` is deliberate (drift signal).
"""

from __future__ import annotations

import pytest
from app.processing.session_processing import (
    SessionProcessingConfig,
    SessionProcessingConfigEnvelope,
)
from pydantic import ValidationError


def _valid_config() -> dict:
    return {
        "version": 1,
        "model_key": "trafficcamnet_v1",
        "selected_classes": ["person", "car"],
        "tracker": "iou",
        "confidence_threshold": 0.55,
        "processing_fps": 5,
    }


def _valid_envelope() -> dict:
    return {"revision": 3, "config": _valid_config()}


# --- happy path --------------------------------------------------------


def test_config_accepts_canonical_shape() -> None:
    cfg = SessionProcessingConfig.model_validate(_valid_config())
    assert cfg.version == 1
    assert cfg.model_key == "trafficcamnet_v1"
    assert cfg.selected_classes == ["person", "car"]
    assert cfg.tracker == "iou"
    assert cfg.confidence_threshold == 0.55
    assert cfg.processing_fps == 5


def test_envelope_accepts_canonical_shape() -> None:
    env = SessionProcessingConfigEnvelope.model_validate(_valid_envelope())
    assert env.revision == 3
    assert env.config.model_key == "trafficcamnet_v1"
    assert env.config.processing_fps == 5


def test_config_defaults_for_optional_fields() -> None:
    """Only ``version`` + ``model_key`` are required; the rest default to the
    'no override' sentinels (empty list / None)."""
    cfg = SessionProcessingConfig.model_validate({"version": 1, "model_key": "trafficcamnet_v1"})
    assert cfg.selected_classes == []
    assert cfg.tracker is None
    assert cfg.confidence_threshold is None
    assert cfg.processing_fps is None


# --- structural rejection (extra="forbid" + bounds/types) --------------


def test_unknown_config_field_rejected() -> None:
    """``extra="forbid"`` — a forward-compatible api-server field we don't
    yet understand degrades instead of being silently ignored."""
    cfg = _valid_config()
    cfg["unexpected"] = "drift"
    with pytest.raises(ValidationError):
        SessionProcessingConfig.model_validate(cfg)


def test_unknown_envelope_field_rejected() -> None:
    env = _valid_envelope()
    env["unexpected"] = "drift"
    with pytest.raises(ValidationError):
        SessionProcessingConfigEnvelope.model_validate(env)


def test_version_must_be_one() -> None:
    cfg = _valid_config()
    cfg["version"] = 2
    with pytest.raises(ValidationError):
        SessionProcessingConfig.model_validate(cfg)


def test_model_key_required_and_non_empty() -> None:
    cfg = _valid_config()
    cfg["model_key"] = ""
    with pytest.raises(ValidationError):
        SessionProcessingConfig.model_validate(cfg)
    del cfg["model_key"]
    with pytest.raises(ValidationError):
        SessionProcessingConfig.model_validate(cfg)


def test_model_key_max_length() -> None:
    cfg = _valid_config()
    cfg["model_key"] = "x" * 129
    with pytest.raises(ValidationError):
        SessionProcessingConfig.model_validate(cfg)


def test_selected_classes_caps_at_64() -> None:
    cfg = _valid_config()
    cfg["selected_classes"] = [f"c{i}" for i in range(65)]
    with pytest.raises(ValidationError):
        SessionProcessingConfig.model_validate(cfg)


def test_confidence_threshold_unit_interval() -> None:
    for bad in (-0.01, 1.01):
        cfg = _valid_config()
        cfg["confidence_threshold"] = bad
        with pytest.raises(ValidationError):
            SessionProcessingConfig.model_validate(cfg)


def test_processing_fps_bounds() -> None:
    for bad in (0, 31):
        cfg = _valid_config()
        cfg["processing_fps"] = bad
        with pytest.raises(ValidationError):
            SessionProcessingConfig.model_validate(cfg)


def test_tracker_max_length() -> None:
    cfg = _valid_config()
    cfg["tracker"] = "x" * 65
    with pytest.raises(ValidationError):
        SessionProcessingConfig.model_validate(cfg)


def test_envelope_revision_must_be_non_negative() -> None:
    env = _valid_envelope()
    env["revision"] = -1
    with pytest.raises(ValidationError):
        SessionProcessingConfigEnvelope.model_validate(env)


def test_envelope_revision_must_be_int() -> None:
    env = _valid_envelope()
    env["revision"] = "not-an-int"
    with pytest.raises(ValidationError):
        SessionProcessingConfigEnvelope.model_validate(env)


def test_envelope_requires_config() -> None:
    with pytest.raises(ValidationError):
        SessionProcessingConfigEnvelope.model_validate({"revision": 1})
