"""P4-C — DeepStream ROI occupancy observation emit path.

In-process tests of the supervisor-managed analytics worker actually
writing ``processor.roi_occupancy`` observations via the existing
``ObservationWriter``. No Docker / GPU / pyds — ``FakeMetadataPM``
feeds scripted analytics frames; ``FakeApi`` records ``post_observation``
calls; the worker runs end-to-end through ``_drive_metadata_probe``.

What this file is the contract for:

- ``DeepStreamWorker`` accepts ROI emissions only in analytics mode and
  only when a writer is wired.
- The emit policy is **first-seen + on-change + interval heartbeat**;
  unchanged frames inside the interval window emit nothing.
- ``payload["debug"]["emit_reason"]`` is one of ``"first" | "change" |
  "interval"`` (a P4-C addition).
- ``ObservationWriter.emit`` is called with the right
  ``stream_session_id`` / ``obs_type`` / ``package_id`` / ``observed_at``.
- Transient API failures don't crash the worker — counters surface
  the error count; the line-crossing path is unaffected.
- Tracker- / metadata-only modes do NOT emit ROI observations.
- ``writer=None`` is safe; status payload reports
  ``observations_enabled=False``.
"""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime, timedelta
from typing import Any

from app.errors import ApiServerError
from app.observations.observation_adapter import CANDIDATE_ROI_OCCUPANCY_TYPE
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


# --- schema + taxonomy lock-step (regression on P4-C widening) ----------


def test_roi_occupancy_now_in_observation_types_tuple() -> None:
    """The whole point of P4-C: this type is now accepted by the
    writer's strict guard."""
    assert CANDIDATE_ROI_OCCUPANCY_TYPE in OBSERVATION_TYPES


def test_future_event_types_still_unsupported() -> None:
    """P4-D widened the whitelist with zone_entered + zone_exited (the
    test file's name predates P4-D; the remaining future types are
    dwell variants, direction, track types, telemetry). The two
    zone types are intentionally NOT in this list anymore."""
    for t in (
        "processor.dwell_started",
        "processor.dwell_updated",
        "processor.dwell_exceeded",
        "processor.direction_detected",
        "processor.track_state",
        "processor.track_update",
    ):
        assert t not in OBSERVATION_TYPES, t


# --- shared fixtures ---------------------------------------------------


def _session(**ov: object) -> EligibleSession:
    return EligibleSession.model_validate(eligible_session_dict(**ov))


_BASE_TIME = datetime(2026, 5, 20, 16, 0, 0, tzinfo=UTC)


def _now_iso(offset_seconds: float = 0.0) -> str:
    return (_BASE_TIME + timedelta(seconds=offset_seconds)).isoformat()


def _frame_line(
    *,
    frame_num: int,
    objects: list[dict[str, Any]] | None = None,
    line_crossing_cumulative: dict[str, int] | None = None,
    roi_counts: dict[str, int] | None = None,
    timestamp: str | None = None,
) -> str:
    """Same shape as the P4-B line-crossing test helper but lets the
    caller stamp a per-frame timestamp (needed to test the ROI
    interval heartbeat path; without timestamps every frame uses
    ``now_utc()`` and the interval logic can't be exercised
    deterministically)."""
    payload: dict[str, Any] = {
        "runtime": "deepstream",
        "metadata_source": "pyds_pad_probe",
        "stream_session_id": None,
        "frame": {"frame_num": frame_num, "width": 1280, "height": 720},
        "objects": objects or [],
    }
    if timestamp is not None:
        payload["frame"]["timestamp"] = timestamp
    analytics: dict[str, Any] = {}
    if line_crossing_cumulative is not None:
        analytics["line_crossing_cumulative"] = line_crossing_cumulative
    if roi_counts is not None:
        analytics["roi_counts"] = roi_counts
    if analytics:
        payload["source"] = {"analytics": analytics}
    return "UPSTREAM_METADATA=" + json.dumps(payload)


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


def _roi_obs(api: FakeApi) -> list[ObservationPayload]:
    return [o for o in api.observations if o.type == "processor.roi_occupancy"]


# --- emit policy: first / change / interval ----------------------------


async def test_emits_first_observation_for_each_roi(fast_settings) -> None:  # type: ignore[no-untyped-def]
    """First frame for any ROI emits even when the count is zero, so
    the consumer knows the ROI exists."""
    api = FakeApi()
    pm = FakeMetadataPM(
        [_frame_line(frame_num=0, roi_counts={"entry-roi": 0})],
        exit_code=124,
    )
    w = _analytics_worker(api, fast_settings, pm)
    await asyncio.wait_for(w.run(), timeout=3)

    obs = _roi_obs(api)
    assert len(obs) == 1
    body = obs[0].payload
    assert body["roi_id"] == "entry-roi"
    assert body["total_count"] == 0
    assert body["sequence"] == 1
    assert body["debug"]["emit_reason"] == "first"
    assert body["idempotency_key"].endswith(":entry-roi:1")


async def test_emits_on_count_change(fast_settings) -> None:  # type: ignore[no-untyped-def]
    """Total count went 0 → 1 → 1 → 2 → 2:
    first (count=0) + change (count=1) + change (count=2) = 3 emits."""
    api = FakeApi()
    pm = FakeMetadataPM(
        [
            _frame_line(frame_num=0, roi_counts={"entry-roi": 0}),
            _frame_line(frame_num=1, roi_counts={"entry-roi": 1}),
            _frame_line(frame_num=2, roi_counts={"entry-roi": 1}),
            _frame_line(frame_num=3, roi_counts={"entry-roi": 2}),
            _frame_line(frame_num=4, roi_counts={"entry-roi": 2}),
        ],
        exit_code=124,
    )
    w = _analytics_worker(api, fast_settings, pm)
    await asyncio.wait_for(w.run(), timeout=3)

    obs = _roi_obs(api)
    counts = [o.payload["total_count"] for o in obs]
    reasons = [o.payload["debug"]["emit_reason"] for o in obs]
    assert counts == [0, 1, 2]
    assert reasons == ["first", "change", "change"]


async def test_no_emit_on_unchanged_within_interval(fast_settings) -> None:  # type: ignore[no-untyped-def]
    """50 frames at the same count, no per-frame timestamps (so
    ``now_utc()`` flows naturally) — with the default 30s interval,
    only the first observation lands."""
    api = FakeApi()
    frames = [_frame_line(frame_num=i, roi_counts={"entry-roi": 1}) for i in range(50)]
    pm = FakeMetadataPM(frames, exit_code=124)
    w = _analytics_worker(api, fast_settings, pm)
    await asyncio.wait_for(w.run(), timeout=3)
    obs = _roi_obs(api)
    assert len(obs) == 1
    assert obs[0].payload["debug"]["emit_reason"] == "first"


async def test_emits_on_interval_heartbeat(fast_settings) -> None:  # type: ignore[no-untyped-def]
    """Same count but the producer's frame timestamps cross the
    ``min_interval_seconds`` boundary — heartbeat emits."""
    settings = fast_settings.model_copy(update={"deepstream_roi_min_interval_seconds": 30.0})
    api = FakeApi()
    pm = FakeMetadataPM(
        [
            _frame_line(frame_num=0, roi_counts={"entry-roi": 1}, timestamp=_now_iso(0)),
            _frame_line(frame_num=1, roi_counts={"entry-roi": 1}, timestamp=_now_iso(10)),
            _frame_line(frame_num=2, roi_counts={"entry-roi": 1}, timestamp=_now_iso(31)),
            _frame_line(frame_num=3, roi_counts={"entry-roi": 1}, timestamp=_now_iso(62)),
        ],
        exit_code=124,
    )
    w = _analytics_worker(api, settings, pm)
    await asyncio.wait_for(w.run(), timeout=3)
    obs = _roi_obs(api)
    # First at 0s, no emit at 10s (too soon), heartbeat at 31s, heartbeat at 62s.
    reasons = [o.payload["debug"]["emit_reason"] for o in obs]
    assert reasons == ["first", "interval", "interval"]
    assert obs[0].payload["sequence"] == 1
    assert obs[1].payload["sequence"] == 2
    assert obs[2].payload["sequence"] == 3


async def test_short_interval_emits_every_frame(fast_settings) -> None:  # type: ignore[no-untyped-def]
    """A test-only knob lets the interval be tiny. Useful to assert
    the interval path is wired up without needing 30s of synthetic
    time travel in other tests."""
    settings = fast_settings.model_copy(update={"deepstream_roi_min_interval_seconds": 0.0})
    api = FakeApi()
    pm = FakeMetadataPM(
        [
            _frame_line(frame_num=0, roi_counts={"entry-roi": 1}, timestamp=_now_iso(0)),
            _frame_line(frame_num=1, roi_counts={"entry-roi": 1}, timestamp=_now_iso(0.001)),
            _frame_line(frame_num=2, roi_counts={"entry-roi": 1}, timestamp=_now_iso(0.002)),
        ],
        exit_code=124,
    )
    w = _analytics_worker(api, settings, pm)
    await asyncio.wait_for(w.run(), timeout=3)
    obs = _roi_obs(api)
    assert len(obs) == 3
    # First emit reason "first", then "interval" twice.
    assert [o.payload["debug"]["emit_reason"] for o in obs] == ["first", "interval", "interval"]


# --- payload shape -----------------------------------------------------


async def test_payload_idempotency_key_uses_sequence_not_frame_num(fast_settings) -> None:  # type: ignore[no-untyped-def]
    api = FakeApi()
    pm = FakeMetadataPM(
        [
            _frame_line(frame_num=42, roi_counts={"entry-roi": 0}),
            _frame_line(frame_num=99, roi_counts={"entry-roi": 1}),
        ],
        exit_code=124,
    )
    w = _analytics_worker(api, fast_settings, pm)
    await asyncio.wait_for(w.run(), timeout=3)
    obs = _roi_obs(api)
    assert len(obs) == 2
    # Keys end in :<sequence>, not :<frame_num>. The session UUID
    # might happen to contain frame-number digits by chance, so the
    # ``endswith`` form is the load-bearing assertion; the raw frame
    # numbers (42, 99) only land in payload["debug"]["frame_num"].
    assert obs[0].payload["idempotency_key"].endswith(":entry-roi:1")
    assert obs[1].payload["idempotency_key"].endswith(":entry-roi:2")
    assert obs[0].payload["sequence"] == 1
    assert obs[1].payload["sequence"] == 2
    assert obs[0].payload["debug"]["frame_num"] == 42
    assert obs[1].payload["debug"]["frame_num"] == 99


async def test_payload_runtime_neutral_at_top_level(fast_settings) -> None:  # type: ignore[no-untyped-def]
    """DeepStream / nvdsanalytics vocabulary stays under ``source`` /
    ``debug``. emit_reason ALSO lives under ``debug``, not at the top."""
    api = FakeApi()
    pm = FakeMetadataPM(
        [
            _frame_line(frame_num=7, roi_counts={"entry-roi": 1}),
        ],
        exit_code=124,
    )
    w = _analytics_worker(api, fast_settings, pm)
    await asyncio.wait_for(w.run(), timeout=3)
    body = _roi_obs(api)[0].payload
    forbidden_top = {
        "runtime",
        "package_id",
        "metadata_source",
        "analytics_plugin",
        "tracker_algorithm",
        "frame_num",
        "raw_roi_counts",
        "geometry_source",
        "emit_reason",
    }
    leaked = forbidden_top & set(body.keys())
    assert not leaked, f"DeepStream vocabulary leaked to top level: {leaked}"
    # Provenance under source.
    assert body["source"]["runtime"] == "deepstream"
    assert body["source"]["package_id"] == "deepstream.analytics_probe.v0"
    assert body["source"]["metadata_source"] == "pyds_pad_probe"
    assert body["source"]["analytics_plugin"] == "nvdsanalytics"
    assert body["source"]["tracker_algorithm"] == "iou"
    # Raw counter + frame number + emit reason under debug.
    assert body["debug"]["frame_num"] == 7
    assert body["debug"]["geometry_source"] == "probe_default"
    assert body["debug"]["raw_roi_counts"] == {"entry-roi": 1}
    assert body["debug"]["emit_reason"] == "first"


# --- defensive (missing / malformed) -----------------------------------


async def test_no_emit_when_roi_counts_missing(fast_settings) -> None:  # type: ignore[no-untyped-def]
    """A frame with no ``source.analytics`` block (or no roi_counts
    inside it) must not trigger any ROI write."""
    api = FakeApi()
    pm = FakeMetadataPM(
        [
            _frame_line(frame_num=0),  # no source.analytics at all
            _frame_line(frame_num=1, line_crossing_cumulative={"entry-line": 0}),
        ],
        exit_code=124,
    )
    w = _analytics_worker(api, fast_settings, pm)
    await asyncio.wait_for(w.run(), timeout=3)
    assert _roi_obs(api) == []


async def test_no_emit_when_roi_counts_malformed(fast_settings) -> None:  # type: ignore[no-untyped-def]
    """A roi_counts value that's not a dict-of-numbers must NOT
    crash the consumer and must NOT emit. (The NeutralFrameMetadata
    schema is permissive about ``source.analytics``'s inner shape so
    we defensively coerce + filter in the worker.)"""
    api = FakeApi()
    # Build a frame whose analytics.roi_counts is the wrong shape.
    bad = "UPSTREAM_METADATA=" + json.dumps(
        {
            "runtime": "deepstream",
            "metadata_source": "pyds_pad_probe",
            "stream_session_id": None,
            "frame": {"frame_num": 0, "width": 1280, "height": 720},
            "objects": [],
            "source": {"analytics": {"roi_counts": "not-a-dict"}},
        }
    )
    bad_inner = "UPSTREAM_METADATA=" + json.dumps(
        {
            "runtime": "deepstream",
            "metadata_source": "pyds_pad_probe",
            "stream_session_id": None,
            "frame": {"frame_num": 1, "width": 1280, "height": 720},
            "objects": [],
            "source": {"analytics": {"roi_counts": {"entry-roi": "not-an-int"}}},
        }
    )
    pm = FakeMetadataPM([bad, bad_inner], exit_code=124)
    w = _analytics_worker(api, fast_settings, pm)
    await asyncio.wait_for(w.run(), timeout=3)
    assert _roi_obs(api) == []


# --- mode gating -------------------------------------------------------


async def test_tracker_mode_does_not_emit_roi(fast_settings) -> None:  # type: ignore[no-untyped-def]
    """tracker_probe mode never runs nvdsanalytics; even with a writer
    + ROI counts in the frame, the adapter is not built and emits
    never happen."""
    api = FakeApi()
    pm = FakeMetadataPM(
        [_frame_line(frame_num=0, roi_counts={"entry-roi": 1})],
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
    assert _roi_obs(api) == []


async def test_metadata_mode_does_not_emit_roi(fast_settings) -> None:  # type: ignore[no-untyped-def]
    api = FakeApi()
    pm = FakeMetadataPM(
        [_frame_line(frame_num=0, roi_counts={"entry-roi": 1})],
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
    assert _roi_obs(api) == []


async def test_analytics_mode_without_writer_does_not_emit(fast_settings) -> None:  # type: ignore[no-untyped-def]
    """``writer=None`` keeps the worker silent on the ROI path too;
    status payload signals ``observations_enabled=False``."""
    api = FakeApi()
    pm = FakeMetadataPM(
        [_frame_line(frame_num=0, roi_counts={"entry-roi": 1})],
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


async def test_transient_roi_write_failure_does_not_crash_worker(fast_settings) -> None:  # type: ignore[no-untyped-def]
    """A transient API error during ROI emit is absorbed: the worker
    still ends in ``stopped``, the error count surfaces in the status
    payload, and the RTSP / session key are not in any logged
    payload."""
    api = FakeApi()
    api.observation_error = ApiServerError("simulated upstream 500", status=500)
    pm = FakeMetadataPM(
        [
            _frame_line(frame_num=0, roi_counts={"entry-roi": 0}),
            _frame_line(frame_num=1, roi_counts={"entry-roi": 1}),
        ],
        exit_code=124,
    )
    w = _analytics_worker(api, fast_settings, pm)
    await asyncio.wait_for(w.run(), timeout=3)
    assert w.state is WorkerState.STOPPED
    stopped = next(s for s in api.statuses if s.status == "stopped")
    p = stopped.payload
    assert p["observations_enabled"] is True
    # Every ROI emit attempt failed → roi_occupancy_observations_emitted == 0,
    # observation_write_errors_count >= 2.
    assert p["roi_occupancy_observations_emitted"] == 0
    assert p["observation_write_errors_count"] >= 2
    for s in api.statuses:
        text = json.dumps(s.payload)
        assert _KEY not in text


# --- combined line + ROI in same cycle ---------------------------------


async def test_line_and_roi_emit_independently_in_same_cycle(fast_settings) -> None:  # type: ignore[no-untyped-def]
    """A frame that ALSO carries a line-crossing increment must
    produce BOTH an ``processor.line_crossing`` row AND an
    ``processor.roi_occupancy`` row in the same consumer iteration.
    Verifies the two emit helpers operate independently."""
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
    line = [o for o in api.observations if o.type == "processor.line_crossing"]
    roi = [o for o in api.observations if o.type == "processor.roi_occupancy"]
    assert len(line) == 1  # 0 -> 1 increment
    assert len(roi) == 2  # first (count=0) + change (count=1)


async def test_emit_counters_distinguish_line_and_roi(fast_settings) -> None:  # type: ignore[no-untyped-def]
    """The status payload exposes the two counters separately so an
    operator can tell line- vs ROI-throughput apart."""
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
            _frame_line(
                frame_num=2,
                line_crossing_cumulative={"entry-line": 2},
                roi_counts={"entry-roi": 1},
            ),
        ],
        exit_code=124,
    )
    w = _analytics_worker(api, fast_settings, pm)
    await asyncio.wait_for(w.run(), timeout=3)
    stopped = next(s for s in api.statuses if s.status == "stopped")
    p = stopped.payload
    assert p["observations_enabled"] is True
    assert p["line_crossing_observations_emitted"] == 2  # 0->1, 1->2
    assert p["roi_occupancy_observations_emitted"] == 2  # first, change to 1
    assert p["observations_emitted_count"] == 4
    assert p["observation_write_errors_count"] == 0


# --- writer .emit() call surface --------------------------------------


async def test_writer_emit_call_surface_for_roi(fast_settings) -> None:  # type: ignore[no-untyped-def]
    """``ObservationPayload`` records exactly what reached the API.
    Verifies obs_type / package_id / processor_id / stream_session_id /
    observed_at on the ROI path."""
    api = FakeApi()
    pm = FakeMetadataPM(
        [_frame_line(frame_num=0, roi_counts={"entry-roi": 0})],
        exit_code=124,
    )
    w = _analytics_worker(api, fast_settings, pm)
    await asyncio.wait_for(w.run(), timeout=3)
    obs = _roi_obs(api)
    assert len(obs) == 1
    o = obs[0]
    assert isinstance(o, ObservationPayload)
    assert o.type == "processor.roi_occupancy"
    assert str(o.stream_session_id) == str(w.session_id)
    assert o.processor_id == _PROCESSOR_ID
    assert o.package_id == "deepstream.analytics_probe.v0"
    assert o.observed_at.tzinfo is not None


# --- P4-C cleanup: debounce / hysteresis -------------------------------


async def test_short_flap_below_threshold_does_not_emit(fast_settings) -> None:  # type: ignore[no-untyped-def]
    """Reproduces the P4-C live-finding pattern: count flips 0 -> 1 ->
    0 quickly. With debounce=5, neither the up-flap nor the
    down-flap should commit a change emit. Only the first observation
    fires."""
    settings = fast_settings.model_copy(update={"deepstream_roi_change_stability_frames": 5})
    api = FakeApi()
    pm = FakeMetadataPM(
        [
            _frame_line(frame_num=0, roi_counts={"entry-roi": 0}),  # first emit
            _frame_line(frame_num=1, roi_counts={"entry-roi": 1}),  # pending 1
            _frame_line(frame_num=2, roi_counts={"entry-roi": 1}),  # pending 2
            _frame_line(frame_num=3, roi_counts={"entry-roi": 0}),  # flap clears
            _frame_line(frame_num=4, roi_counts={"entry-roi": 1}),  # pending starts over
            _frame_line(frame_num=5, roi_counts={"entry-roi": 0}),  # flap clears
            _frame_line(frame_num=6, roi_counts={"entry-roi": 0}),  # steady
        ],
        exit_code=124,
    )
    w = _analytics_worker(api, settings, pm)
    await asyncio.wait_for(w.run(), timeout=3)
    obs = _roi_obs(api)
    # Only the first observation lands; all the 0<->1 flaps are
    # suppressed because the candidate never accumulates 5 consecutive
    # same-value frames.
    assert len(obs) == 1
    assert obs[0].payload["debug"]["emit_reason"] == "first"


async def test_stable_count_commits_after_threshold(fast_settings) -> None:  # type: ignore[no-untyped-def]
    """If the count holds at the new value for >= stability_frames
    frames in a row, the change emits."""
    settings = fast_settings.model_copy(update={"deepstream_roi_change_stability_frames": 3})
    api = FakeApi()
    pm = FakeMetadataPM(
        [
            _frame_line(frame_num=0, roi_counts={"entry-roi": 0}),  # first
            _frame_line(frame_num=1, roi_counts={"entry-roi": 1}),  # pending 1
            _frame_line(frame_num=2, roi_counts={"entry-roi": 1}),  # pending 2
            _frame_line(frame_num=3, roi_counts={"entry-roi": 1}),  # commit
            _frame_line(frame_num=4, roi_counts={"entry-roi": 1}),  # same as last, no emit
        ],
        exit_code=124,
    )
    w = _analytics_worker(api, settings, pm)
    await asyncio.wait_for(w.run(), timeout=3)
    obs = _roi_obs(api)
    assert len(obs) == 2
    assert obs[0].payload["debug"]["emit_reason"] == "first"
    assert obs[1].payload["debug"]["emit_reason"] == "change"
    assert obs[1].payload["total_count"] == 1


async def test_interval_heartbeat_emits_stable_count_under_debounce(  # type: ignore[no-untyped-def]
    fast_settings,
) -> None:
    """A pending change that hasn't yet hit the stability threshold
    must NOT replace the heartbeat's reported count. Heartbeat fires
    with the last STABLE count even while a candidate is pending."""
    settings = fast_settings.model_copy(
        update={
            "deepstream_roi_change_stability_frames": 10,
            "deepstream_roi_min_interval_seconds": 30.0,
        }
    )
    api = FakeApi()
    pm = FakeMetadataPM(
        [
            _frame_line(frame_num=0, roi_counts={"entry-roi": 0}, timestamp=_now_iso(0)),
            # Three pending frames at 1 — well below threshold of 10.
            _frame_line(frame_num=1, roi_counts={"entry-roi": 1}, timestamp=_now_iso(10)),
            _frame_line(frame_num=2, roi_counts={"entry-roi": 1}, timestamp=_now_iso(20)),
            _frame_line(frame_num=3, roi_counts={"entry-roi": 1}, timestamp=_now_iso(31)),
        ],
        exit_code=124,
    )
    w = _analytics_worker(api, settings, pm)
    await asyncio.wait_for(w.run(), timeout=3)
    obs = _roi_obs(api)
    assert len(obs) == 2
    assert obs[0].payload["debug"]["emit_reason"] == "first"
    assert obs[0].payload["total_count"] == 0
    assert obs[1].payload["debug"]["emit_reason"] == "interval"
    # The interval heartbeat reports the LAST STABLE count (0), NOT
    # the pending candidate (1). This is the P4-C cleanup contract.
    assert obs[1].payload["total_count"] == 0


def test_settings_default_for_stability_frames_is_5() -> None:
    """Production default is 5 (per the brief). Tests can override via
    ``fast_settings`` (which sets 1 for backward compat with existing
    immediate-change tests) but the underlying Settings default is the
    production value."""
    from app.config import Settings

    # Don't use ``get_settings()`` (cached fixture state) — instantiate
    # directly with only the required secret env. ``conftest.py``
    # already exports ``UPSTREAM_PROCESSOR_WEBHOOK_SECRET``.
    s = Settings()
    assert s.deepstream_roi_change_stability_frames == 5


def test_invalid_stability_frame_count_rejected_by_adapter() -> None:
    """``Settings`` accepts any int (pydantic permits it) but the
    adapter constructor rejects values < 1 so a misconfigured env
    var fails fast inside the worker, not silently."""
    from uuid import uuid4 as _uuid4

    import pytest
    from app.observations.observation_adapter import RoiOccupancyState

    with pytest.raises(ValueError, match="change_stability_frames must be >= 1"):
        RoiOccupancyState(stream_session_id=_uuid4(), change_stability_frames=0)


async def test_status_payload_exposes_stability_frames(fast_settings) -> None:  # type: ignore[no-untyped-def]
    """The configured debounce threshold appears in the analytics
    status payload so an operator can tell `no observations because
    steady scene` apart from `no observations because debounce too
    aggressive`."""
    settings = fast_settings.model_copy(update={"deepstream_roi_change_stability_frames": 7})
    api = FakeApi()
    pm = FakeMetadataPM(
        [_frame_line(frame_num=0, roi_counts={"entry-roi": 0})],
        exit_code=124,
    )
    w = _analytics_worker(api, settings, pm)
    await asyncio.wait_for(w.run(), timeout=3)
    stopped = next(s for s in api.statuses if s.status == "stopped")
    assert stopped.payload.get("roi_change_stability_frames") == 7
