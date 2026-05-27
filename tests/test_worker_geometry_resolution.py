"""P6-C — DeepStream analytics worker geometry resolution + lineage flow.

These tests exercise the seam where the eligible-session envelope becomes
the runtime's applied geometry. The worker's lifecycle is NOT started here
— we only need:

1. ``_resolved_geometry`` reflects the envelope (or default / fallback).
2. ``_build(status)`` injects ``applied_geometry`` into the status payload,
   with ``error_code="geometry_invalid"`` when the envelope was bad.
3. ``_default_pm_factory()`` forwards the resolved geometry to the
   ``DeepStreamMetadataProcessManager`` so the rendered INI matches what
   was saved.
"""

from __future__ import annotations

from app.observations.schemas import EligibleSession
from app.packages.resolver import resolve
from app.runtimes.deepstream_runtime.metadata_probe import (
    DeepStreamMetadataProcessManager,
)
from app.runtimes.deepstream_runtime.worker import DeepStreamWorker

from tests._fakes import FakeApi
from tests.conftest import eligible_session_dict


def _session(geometry_config: dict | None = None) -> EligibleSession:
    return EligibleSession.model_validate(eligible_session_dict(geometry_config=geometry_config))


def _operator_envelope(*, revision: int = 4) -> dict:
    return {
        "revision": revision,
        "config": {
            "version": 1,
            "frame": {"width": 1280, "height": 720, "coordinate_space": "normalized"},
            "rois": [
                {
                    "id": "main-room",
                    "label": "Main Room",
                    "type": "polygon",
                    "points": [
                        {"x": 0.1, "y": 0.1},
                        {"x": 0.9, "y": 0.1},
                        {"x": 0.9, "y": 0.9},
                        {"x": 0.1, "y": 0.9},
                    ],
                    "enabled": True,
                }
            ],
            "lines": [
                {
                    "id": "doorway",
                    "label": "Doorway",
                    "points": [{"x": 0.2, "y": 0.5}, {"x": 0.8, "y": 0.5}],
                    "direction": "in",
                    "enabled": True,
                }
            ],
        },
    }


def _analytics_worker(session: EligibleSession, settings) -> DeepStreamWorker:  # type: ignore[no-untyped-def]
    return DeepStreamWorker(
        session=session,
        settings=settings,
        api=FakeApi(),
        package=resolve("deepstream.analytics_probe.v0"),
    )


# --- resolution outcomes -----------------------------------------------


def test_null_geometry_resolves_to_default(fast_settings) -> None:  # type: ignore[no-untyped-def]
    """Camera with no saved geometry: ``source="default"`` and the probe
    fallback flows through unchanged. Existing camera deployments must
    keep behaving identically to pre-P6-C."""
    w = _analytics_worker(_session(geometry_config=None), fast_settings)
    assert w._resolved_geometry is not None
    assert w._resolved_geometry.source == "default"
    assert w._resolved_geometry.revision is None


def test_valid_operator_envelope_resolves_to_operator_saved(fast_settings) -> None:  # type: ignore[no-untyped-def]
    w = _analytics_worker(_session(_operator_envelope(revision=4)), fast_settings)
    assert w._resolved_geometry is not None
    assert w._resolved_geometry.source == "operator_saved"
    assert w._resolved_geometry.revision == 4
    assert w._resolved_geometry.line_count == 1
    assert w._resolved_geometry.roi_count == 1
    # Operator IDs survive resolution — observation rows will reference these.
    assert {r.id for r in w._resolved_geometry.geometry.rois} == {"main-room"}
    assert {line.id for line in w._resolved_geometry.geometry.lines} == {"doorway"}


def test_invalid_operator_envelope_degrades_with_fallback(fast_settings) -> None:  # type: ignore[no-untyped-def]
    """Bad envelope (wrong coordinate_space): worker stays up with probe
    defaults but flags the degradation. ``revision`` is still surfaced so
    operators can correlate the bad save."""
    bad = _operator_envelope(revision=9)
    bad["config"]["frame"]["coordinate_space"] = "absolute"
    w = _analytics_worker(_session(bad), fast_settings)
    assert w._resolved_geometry is not None
    assert w._resolved_geometry.source == "fallback_default_on_error"
    assert w._resolved_geometry.error_reason is not None


# --- status payload lineage --------------------------------------------


def test_status_payload_carries_applied_geometry(fast_settings) -> None:  # type: ignore[no-untyped-def]
    w = _analytics_worker(_session(_operator_envelope(revision=11)), fast_settings)
    status = w._build("attached")
    assert status.payload["applied_geometry"] == {
        "source": "operator_saved",
        "revision": 11,
        "line_count": 1,
        "roi_count": 1,
    }
    # No error_code on a healthy operator-saved geometry.
    assert status.error_code is None


def test_status_payload_carries_default_geometry(fast_settings) -> None:  # type: ignore[no-untyped-def]
    w = _analytics_worker(_session(None), fast_settings)
    status = w._build("attached")
    assert status.payload["applied_geometry"]["source"] == "default"
    assert status.payload["applied_geometry"]["revision"] is None
    assert status.error_code is None


def test_invalid_envelope_status_sets_error_code(fast_settings) -> None:  # type: ignore[no-untyped-def]
    """The product call for invalid geometry is 'fall back + degrade':
    keep the camera running but surface ``error_code="geometry_invalid"``
    on status so dashboards can see the degradation."""
    bad = _operator_envelope(revision=1)
    bad["config"]["frame"]["coordinate_space"] = "absolute"
    w = _analytics_worker(_session(bad), fast_settings)
    status = w._build("attached")
    assert status.error_code == "geometry_invalid"
    assert status.error_message  # error_reason text propagates
    assert status.payload["applied_geometry"]["source"] == "fallback_default_on_error"


def test_caller_error_code_wins_over_geometry_invalid(fast_settings) -> None:  # type: ignore[no-untyped-def]
    """Runtime errors (container crash, model load failure) keep their own
    taxonomy — geometry degradation must not overwrite a more-specific
    error from the caller."""
    bad = _operator_envelope(revision=1)
    bad["config"]["frame"]["coordinate_space"] = "absolute"
    w = _analytics_worker(_session(bad), fast_settings)
    status = w._build("failed", error_code="container_crash", error_message="exit 1")
    assert status.error_code == "container_crash"
    # applied_geometry lineage still flows so the bad config is visible.
    assert status.payload["applied_geometry"]["source"] == "fallback_default_on_error"


def test_caller_payload_merges_with_applied_geometry(fast_settings) -> None:  # type: ignore[no-untyped-def]
    """A caller passing extra payload fields (e.g. analytics counters) must
    not clobber the geometry lineage block — and vice versa."""
    w = _analytics_worker(_session(_operator_envelope()), fast_settings)
    status = w._build("processing", payload={"fps_processed": 30, "frames_seen": 1234})
    assert status.payload["fps_processed"] == 30
    assert status.payload["frames_seen"] == 1234
    assert status.payload["applied_geometry"]["source"] == "operator_saved"


# --- factory forwarding ------------------------------------------------


def test_default_pm_factory_forwards_resolved_geometry(fast_settings) -> None:  # type: ignore[no-untyped-def]
    """``_default_pm_factory()`` must hand the analytics PM the resolved
    geometry so the rendered nvdsanalytics INI matches saved geometry, not
    the probe default."""
    w = _analytics_worker(_session(_operator_envelope(revision=2)), fast_settings)
    pm = w._default_pm_factory()
    assert isinstance(pm, DeepStreamMetadataProcessManager)
    forwarded = pm._analytics_geometry
    assert forwarded is not None
    assert {r.id for r in forwarded.rois} == {"main-room"}
    assert {line.id for line in forwarded.lines} == {"doorway"}


def test_default_pm_factory_null_geometry_for_non_analytics_modes(fast_settings) -> None:  # type: ignore[no-untyped-def]
    """The metadata / tracker / inference probes don't run nvdsanalytics
    and must keep ``analytics_geometry=None`` so the legacy code path
    isn't accidentally enabled."""
    w = DeepStreamWorker(
        session=_session(_operator_envelope()),
        settings=fast_settings,
        api=FakeApi(),
        package=resolve("deepstream.tracker_probe.v0"),
    )
    pm = w._default_pm_factory()
    assert isinstance(pm, DeepStreamMetadataProcessManager)
    assert pm._analytics_geometry is None


def test_non_analytics_worker_does_not_resolve_geometry(fast_settings) -> None:  # type: ignore[no-untyped-def]
    """Tracker/metadata/decode/inference modes don't consume geometry, so
    we skip the resolver and don't leak ``applied_geometry`` lineage onto
    their status payloads."""
    w = DeepStreamWorker(
        session=_session(_operator_envelope()),
        settings=fast_settings,
        api=FakeApi(),
        package=resolve("deepstream.metadata_probe.v0"),
    )
    assert w._resolved_geometry is None
    status = w._build("attached")
    assert "applied_geometry" not in status.payload
