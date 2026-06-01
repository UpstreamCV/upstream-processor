"""Pydantic models for the operator-saved processing-config envelope.

These mirror ``upstream-api-server``'s
``schemas/processing_config_camera.py`` (``CameraProcessingConfig``)
exactly — the api-server is the source of truth for the wire contract, and
the processor re-validates here because the eligible-session payload carries
``processing_config`` as an opaque dict that has only seen JSON-decode (no
schema validation) up to this point. Re-validating locally means a
malformed config from a future api drift surfaces as a clean fallback in the
processor instead of corrupting the lineage block downstream.

Drift policy: ``extra="forbid"`` here is deliberate (mirrors
``session_geometry.py``). The api-server enforces its schema on write, so an
unexpected key in the wire envelope means *either* an api-server bug *or* a
malicious row that bypassed validation. Either way, the resolver should
degrade to runtime defaults rather than silently accept the foreign field.

M7B is STAMP ONLY: these models exist so the worker can re-validate the
envelope and stamp a requested-vs-applied lineage block. Nothing parsed here
is applied to the runtime (model/tracker/threshold/fps stay at the
DeepStream worker's current defaults).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class SessionProcessingConfig(BaseModel):
    """The processing config payload stored in
    ``camera_processing_configs.config``.

    STRUCTURAL VALIDATION ONLY — bounds/types, no model-registry checks —
    matching the api-server's ``CameraProcessingConfig``.
    """

    # ``protected_namespaces=()`` so ``model_key`` doesn't collide with
    # Pydantic's reserved ``model_`` namespace.
    model_config = ConfigDict(extra="forbid", protected_namespaces=())
    version: Literal[1]
    model_key: str = Field(min_length=1, max_length=128)
    # [] means all supported classes.
    selected_classes: list[str] = Field(default_factory=list, max_length=64)
    tracker: str | None = Field(default=None, max_length=64)
    confidence_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    processing_fps: int | None = Field(default=None, ge=1, le=30)


class SessionProcessingConfigEnvelope(BaseModel):
    """The eligible-session ``processing_config`` field. ``revision`` is the
    api-server's monotonic save counter — surfaced on status lineage so
    consumers can correlate the stamped block to a specific save."""

    model_config = ConfigDict(extra="forbid")
    revision: int = Field(ge=0)
    config: SessionProcessingConfig
