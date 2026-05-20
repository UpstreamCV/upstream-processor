"""P4-B — DeepStream line-crossing observation emit path.

End-to-end (in-process) tests of the supervisor-managed analytics
worker actually writing ``processor.line_crossing`` observations via
the existing ``ObservationWriter``. No Docker / GPU / pyds — the
``FakeMetadataPM`` from the P3.5 test file feeds scripted analytics
frames; ``FakeApi`` records ``post_observation`` calls; the worker
runs end-to-end through ``_drive_metadata_probe``.

What this file is the contract for:

- ``DeepStreamWorker`` accepts an ``ObservationWriter`` and uses it
  only in analytics mode.
- Each integer increment in ``frame.source.analytics.line_crossing_cumulative``
  produces exactly one ``processor.line_crossing`` observation.
- Repeated / rewound cumulative counts produce zero observations.
- The emitted payload is the deterministic shape locked in by
  P4-A (runtime-neutral top-level, DeepStream details under
  ``source`` / ``debug``, ``idempotency_key`` inside ``payload``).
- ``ObservationWriter.emit`` is called with the right
  ``stream_session_id`` / ``obs_type`` / ``package_id`` / ``observed_at``.
- Transient API failures don't crash the worker — counters surface
  the error count.
- Tracker- / metadata-only modes do NOT emit observations.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from app.errors import ApiServerError
from app.observations.observation_adapter import CANDIDATE_LINE_CROSSING_TYPE
from app.observations.schemas import OBSERVATION_TYPES, EligibleSession, ObservationPayload
from app.observations.writer import ObservationWriter
from app.packages.resolver import resolve
from app.runtimes.base import WorkerState
from app.runtimes.deepstream_runtime.worker import DeepStreamWorker

from tests._fakes import FakeApi
from tests.conftest import eligible_session_dict
from tests.test_deepstream_metadata_probe import FakeMetadataPM

_URL = "rtsp://127.0.0.1:18554/sess_abcdef0123456789"
_KEY = "abcdef0123456789"
_PROCESSOR_ID = "proc_test_001"


# --- schema + taxonomy lock-step (regression on P4-B widening) ---------


def test_line_crossing_now_in_observation_types_tuple() -> None:
    """The whole point of P4-B is that this type is now accepted by
    the writer's strict guard. Explicit assertion so a future demotion
    breaks loudly."""
    assert CANDIDATE_LINE_CROSSING_TYPE in OBSERVATION_TYPES


def test_roi_occupancy_now_supported_by_writer() -> None:
    """P4-C widened the whitelist with ``processor.roi_occupancy``
    (the ROI emit path lives in tests/test_deepstream_roi_occupancy_emit.py).
    This is the line-crossing-test-file regression that the writer
    accepts both types — line crossing keeps working, ROI is now also
    legal."""
    assert "processor.roi_occupancy" in OBSERVATION_TYPES


# --- shared fixtures ---------------------------------------------------


def _session(**ov: object) -> EligibleSession:
    return EligibleSession.model_validate(eligible_session_dict(**ov))


def _frame_line(
    *,
    frame_num: int,
    objects: list[dict[str, Any]] | None = None,
    line_crossing_cumulative: dict[str, int] | None = None,
    roi_counts: dict[str, int] | None = None,
) -> str:
    payload: dict[str, Any] = {
        "runtime": "deepstream",
        "metadata_source": "pyds_pad_probe",
        "stream_session_id": None,
        "frame": {"frame_num": frame_num, "width": 1280, "height": 720},
        "objects": objects or [],
    }
    analytics: dict[str, Any] = {}
    if line_crossing_cumulative is not None:
        analytics["line_crossing_cumulative"] = line_crossing_cumulative
    if roi_counts is not None:
        analytics["roi_counts"] = roi_counts
    if analytics:
        payload["source"] = {"analytics": analytics}
    return "UPSTREAM_METADATA=" + json.dumps(payload)


def _person(*, tracker_id: str | None = None, conf: float = 0.51) -> dict[str, Any]:
    return {
        "class_id": 2,
        "label": "person",
        "confidence": conf,
        "bbox_pixels": {"left": 100.0, "top": 100.0, "width": 200.0, "height": 400.0},
        "bbox_normalized": {"x": 0.078, "y": 0.139, "w": 0.156, "h": 0.556},
        "tracker_id": tracker_id,
    }


def _analytics_worker(api: FakeApi, settings, pm: FakeMetadataPM) -> DeepStreamWorker:  # type: ignore[no-untyped-def]
    writer = ObservationWriter(api, _PROCESSOR_ID)
    return DeepStreamWorker(
        session=_session(internal_rtsp_url=_URL),
        settings=settings,
        api=api,
        package=resolve("deepstream.analytics_probe.v0"),
        writer=writer,
        pm_factory=lambda: pm,
    )


# --- WorkerManager wiring ---------------------------------------------


def test_worker_manager_passes_writer_to_deepstream_worker(fast_settings) -> None:  # type: ignore[no-untyped-def]
    """``WorkerManager.build_worker`` must thread the existing writer
    into the deepstream worker — P4-B's core wiring change."""
    from app.supervisor.worker_manager import build_worker

    api = FakeApi()
    writer = ObservationWriter(api, _PROCESSOR_ID)
    spec = resolve("deepstream.analytics_probe.v0")
    worker = build_worker(
        session=_session(internal_rtsp_url=_URL),
        spec=spec,
        settings=fast_settings,
        api=api,
        writer=writer,
    )
    assert isinstance(worker, DeepStreamWorker)
    assert worker._writer is writer  # type: ignore[attr-defined]


# --- emit semantics ----------------------------------------------------


async def test_emits_one_observation_on_zero_to_one(fast_settings) -> None:  # type: ignore[no-untyped-def]
    api = FakeApi()
    pm = FakeMetadataPM(
        [
            _frame_line(frame_num=0, line_crossing_cumulative={"entry-line": 0}),
            _frame_line(
                frame_num=1,
                objects=[_person(tracker_id="0")],
                line_crossing_cumulative={"entry-line": 1},
            ),
        ],
        exit_code=124,
    )
    w = _analytics_worker(api, fast_settings, pm)
    await asyncio.wait_for(w.run(), timeout=3)

    assert w.state is WorkerState.STOPPED
    line_obs = [o for o in api.observations if o.type == "processor.line_crossing"]
    assert len(line_obs) == 1
    obs = line_obs[0]
    assert obs.stream_session_id == w.session_id
    assert obs.processor_id == _PROCESSOR_ID
    assert obs.package_id == "deepstream.analytics_probe.v0"
    body = obs.payload
    assert body["line_id"] == "entry-line"
    assert body["cumulative_count"] == 1
    assert body["count_delta"] == 1
    expected_key = f"sess:{w.session_id}:processor.line_crossing:entry-line:1"
    assert body["idempotency_key"] == expected_key


async def test_emits_two_observations_on_zero_to_one_to_two(fast_settings) -> None:  # type: ignore[no-untyped-def]
    """The exact 0 → 1 → 2 pattern P3.5 captured live. Two distinct
    events with distinct idempotency keys."""
    api = FakeApi()
    pm = FakeMetadataPM(
        [
            _frame_line(frame_num=0, line_crossing_cumulative={"entry-line": 0}),
            _frame_line(frame_num=1, line_crossing_cumulative={"entry-line": 1}),
            _frame_line(frame_num=2, line_crossing_cumulative={"entry-line": 2}),
        ],
        exit_code=124,
    )
    w = _analytics_worker(api, fast_settings, pm)
    await asyncio.wait_for(w.run(), timeout=3)

    line_obs = [o for o in api.observations if o.type == "processor.line_crossing"]
    assert len(line_obs) == 2
    assert [o.payload["cumulative_count"] for o in line_obs] == [1, 2]
    keys = [o.payload["idempotency_key"] for o in line_obs]
    assert keys[0].endswith(":entry-line:1")
    assert keys[1].endswith(":entry-line:2")
    assert keys[0] != keys[1]


async def test_no_emit_on_repeated_cumulative(fast_settings) -> None:  # type: ignore[no-untyped-def]
    api = FakeApi()
    pm = FakeMetadataPM(
        [
            _frame_line(frame_num=0, line_crossing_cumulative={"entry-line": 1}),
            _frame_line(frame_num=1, line_crossing_cumulative={"entry-line": 1}),
            _frame_line(frame_num=2, line_crossing_cumulative={"entry-line": 1}),
        ],
        exit_code=124,
    )
    w = _analytics_worker(api, fast_settings, pm)
    await asyncio.wait_for(w.run(), timeout=3)
    # The first frame sees 0 -> 1 and emits one event; the next two
    # frames repeat cumulative=1 and must emit nothing.
    line_obs = [o for o in api.observations if o.type == "processor.line_crossing"]
    assert len(line_obs) == 1
    assert line_obs[0].payload["cumulative_count"] == 1


async def test_no_emit_on_counter_rewind(fast_settings) -> None:  # type: ignore[no-untyped-def]
    """Producer rewind shouldn't fool the adapter. Watermark held at
    5; a later jump to 6 emits one event for cum=6."""
    api = FakeApi()
    pm = FakeMetadataPM(
        [
            _frame_line(frame_num=0, line_crossing_cumulative={"entry-line": 5}),
            _frame_line(frame_num=1, line_crossing_cumulative={"entry-line": 2}),
            _frame_line(frame_num=2, line_crossing_cumulative={"entry-line": 6}),
        ],
        exit_code=124,
    )
    w = _analytics_worker(api, fast_settings, pm)
    await asyncio.wait_for(w.run(), timeout=3)
    # Frame 0: 0 -> 5 emits five events (1..5). Frame 1: rewind to 2 emits nothing.
    # Frame 2: 5 -> 6 emits one event (cum=6). Total = 6 events.
    line_obs = [o for o in api.observations if o.type == "processor.line_crossing"]
    cum_values = sorted(int(o.payload["cumulative_count"]) for o in line_obs)
    assert cum_values == [1, 2, 3, 4, 5, 6]


# --- payload shape -----------------------------------------------------


async def test_payload_is_runtime_neutral_at_top_level(fast_settings) -> None:  # type: ignore[no-untyped-def]
    """DeepStream / pyds / nvdsanalytics vocabulary stays under
    ``source`` / ``debug``. Test guards against a future careless edit
    that promotes a deepstream field to the top of the body."""
    api = FakeApi()
    pm = FakeMetadataPM(
        [
            _frame_line(frame_num=0, line_crossing_cumulative={"entry-line": 0}),
            _frame_line(frame_num=42, line_crossing_cumulative={"entry-line": 1}),
        ],
        exit_code=124,
    )
    w = _analytics_worker(api, fast_settings, pm)
    await asyncio.wait_for(w.run(), timeout=3)
    line_obs = [o for o in api.observations if o.type == "processor.line_crossing"]
    body = line_obs[0].payload
    forbidden_top = {
        "runtime",
        "package_id",
        "metadata_source",
        "analytics_plugin",
        "tracker_algorithm",
        "frame_num",
        "raw_line_crossing_cumulative",
        "geometry_source",
    }
    leaked = forbidden_top & set(body.keys())
    assert not leaked, f"DeepStream vocabulary leaked to top level: {leaked}"
    # And: provenance lives under source.
    assert body["source"]["runtime"] == "deepstream"
    assert body["source"]["package_id"] == "deepstream.analytics_probe.v0"
    assert body["source"]["metadata_source"] == "pyds_pad_probe"
    assert body["source"]["analytics_plugin"] == "nvdsanalytics"
    assert body["source"]["tracker_algorithm"] == "iou"
    # Raw counter + frame number under debug.
    assert body["debug"]["frame_num"] == 42
    assert body["debug"]["raw_line_crossing_cumulative"] == {"entry-line": 1}
    assert body["debug"]["geometry_source"] == "probe_default"


async def test_line_crossing_unchanged_by_p4c(fast_settings) -> None:  # type: ignore[no-untyped-def]
    """P4-C added a parallel ROI write path. The line-crossing path
    must still emit independently — this test asserts that a frame
    carrying BOTH a line-crossing increment AND a non-zero ROI count
    still produces exactly one line-crossing observation (plus,
    incidentally, ROI observations, but that's the other test
    file's territory). Defends against an accidental rewrite of
    ``_emit_line_crossings`` during P4-C work."""
    api = FakeApi()
    pm = FakeMetadataPM(
        [
            _frame_line(
                frame_num=0,
                line_crossing_cumulative={"entry-line": 0},
                roi_counts={"entry-roi": 0},
            ),
            _frame_line(
                frame_num=1,
                line_crossing_cumulative={"entry-line": 1},
                roi_counts={"entry-roi": 1},
            ),
        ],
        exit_code=124,
    )
    w = _analytics_worker(api, fast_settings, pm)
    await asyncio.wait_for(w.run(), timeout=3)
    obs_types = [o.type for o in api.observations]
    # Exactly one line-crossing event (0 -> 1).
    assert obs_types.count("processor.line_crossing") == 1
    # ROI also emits in P4-C (two: first-seen at 0, then change to 1).
    # Number is not the point of this test (covered in the ROI test
    # file); just assert the line-crossing count is unchanged.


# --- mode gating -------------------------------------------------------


async def test_tracker_mode_does_not_emit_line_crossing(fast_settings) -> None:  # type: ignore[no-untyped-def]
    """tracker_probe mode never runs nvdsanalytics; even with a writer
    threaded in, the adapter state is not built and emits never happen."""
    api = FakeApi()
    pm = FakeMetadataPM(
        [
            _frame_line(frame_num=0, line_crossing_cumulative={"entry-line": 1}),
        ],
        exit_code=124,
    )
    writer = ObservationWriter(api, _PROCESSOR_ID)
    w = DeepStreamWorker(
        session=_session(internal_rtsp_url=_URL),
        settings=fast_settings,
        api=api,
        package=resolve("deepstream.tracker_probe.v0"),
        writer=writer,
        pm_factory=lambda: pm,
    )
    await asyncio.wait_for(w.run(), timeout=3)
    assert all(o.type != "processor.line_crossing" for o in api.observations)


async def test_metadata_mode_does_not_emit_line_crossing(fast_settings) -> None:  # type: ignore[no-untyped-def]
    api = FakeApi()
    pm = FakeMetadataPM(
        [
            _frame_line(frame_num=0, line_crossing_cumulative={"entry-line": 1}),
        ],
        exit_code=124,
    )
    writer = ObservationWriter(api, _PROCESSOR_ID)
    w = DeepStreamWorker(
        session=_session(internal_rtsp_url=_URL),
        settings=fast_settings,
        api=api,
        package=resolve("deepstream.metadata_probe.v0"),
        writer=writer,
        pm_factory=lambda: pm,
    )
    await asyncio.wait_for(w.run(), timeout=3)
    assert all(o.type != "processor.line_crossing" for o in api.observations)


async def test_analytics_mode_without_writer_does_not_emit(fast_settings) -> None:  # type: ignore[no-untyped-def]
    """If no writer is threaded in (unit-test path, or a misconfigured
    supervisor), the worker stays silent — never tries to emit and never
    crashes. The status payload signals ``observations_enabled=False``."""
    api = FakeApi()
    pm = FakeMetadataPM(
        [
            _frame_line(frame_num=0, line_crossing_cumulative={"entry-line": 1}),
        ],
        exit_code=124,
    )
    w = DeepStreamWorker(
        session=_session(internal_rtsp_url=_URL),
        settings=fast_settings,
        api=api,
        package=resolve("deepstream.analytics_probe.v0"),
        writer=None,
        pm_factory=lambda: pm,
    )
    await asyncio.wait_for(w.run(), timeout=3)
    assert api.observations == []
    stopped = next(s for s in api.statuses if s.status == "stopped")
    assert stopped.payload.get("observations_enabled") is False


# --- error handling ----------------------------------------------------


async def test_transient_write_failure_does_not_crash_worker(fast_settings) -> None:  # type: ignore[no-untyped-def]
    """A transient API error during observation emit is absorbed: the
    worker still ends in ``stopped``, the error count surfaces in the
    status payload, and the RTSP / session key are not in the log."""
    api = FakeApi()
    # Cause every post_observation to raise once before recording.
    api.observation_error = ApiServerError("simulated upstream 500", status=500)
    pm = FakeMetadataPM(
        [
            _frame_line(frame_num=0, line_crossing_cumulative={"entry-line": 0}),
            _frame_line(frame_num=1, line_crossing_cumulative={"entry-line": 1}),
        ],
        exit_code=124,
    )
    w = _analytics_worker(api, fast_settings, pm)
    await asyncio.wait_for(w.run(), timeout=3)
    assert w.state is WorkerState.STOPPED
    stopped = next(s for s in api.statuses if s.status == "stopped")
    p = stopped.payload
    assert p["observations_enabled"] is True
    # The emit failed, so emitted_count stays 0 and write_errors is non-zero.
    assert p["observations_emitted_count"] == 0
    assert p["observation_write_errors_count"] >= 1
    # No raw stream key in any status payload (redaction sanity).
    for s in api.statuses:
        text = json.dumps(s.payload)
        assert _KEY not in text


# --- counters ----------------------------------------------------------


async def test_emit_counters_surfaced_in_status_payload(fast_settings) -> None:  # type: ignore[no-untyped-def]
    api = FakeApi()
    pm = FakeMetadataPM(
        [
            _frame_line(frame_num=0, line_crossing_cumulative={"entry-line": 0}),
            _frame_line(frame_num=1, line_crossing_cumulative={"entry-line": 1}),
            _frame_line(frame_num=2, line_crossing_cumulative={"entry-line": 2}),
        ],
        exit_code=124,
    )
    w = _analytics_worker(api, fast_settings, pm)
    await asyncio.wait_for(w.run(), timeout=3)
    stopped = next(s for s in api.statuses if s.status == "stopped")
    p = stopped.payload
    assert p["observations_enabled"] is True
    assert p["observations_emitted_count"] == 2
    assert p["line_crossing_observations_emitted"] == 2
    assert p["observation_write_errors_count"] == 0
    assert p["last_observation_at"] is not None


# --- writer .emit() call surface --------------------------------------


async def test_writer_emit_call_surface(fast_settings) -> None:  # type: ignore[no-untyped-def]
    """``ObservationPayload`` records exactly what reached the API.
    Verifies obs_type / package_id / processor_id / stream_session_id /
    observed_at are the values the rest of the codebase expects."""
    api = FakeApi()
    pm = FakeMetadataPM(
        [
            _frame_line(frame_num=0, line_crossing_cumulative={"entry-line": 0}),
            _frame_line(frame_num=1, line_crossing_cumulative={"entry-line": 1}),
        ],
        exit_code=124,
    )
    w = _analytics_worker(api, fast_settings, pm)
    await asyncio.wait_for(w.run(), timeout=3)
    line_obs = [o for o in api.observations if o.type == "processor.line_crossing"]
    assert len(line_obs) == 1
    o = line_obs[0]
    assert isinstance(o, ObservationPayload)
    assert o.type == "processor.line_crossing"
    assert str(o.stream_session_id) == str(w.session_id)
    assert o.processor_id == _PROCESSOR_ID
    assert o.package_id == "deepstream.analytics_probe.v0"
    # observed_at: the in-test frames don't carry a timestamp, so the
    # adapter fell back to now_utc() — value is whatever the writer
    # received, but it must be timezone-aware.
    assert o.observed_at.tzinfo is not None
