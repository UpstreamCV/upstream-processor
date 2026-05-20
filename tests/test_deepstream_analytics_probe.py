"""P3.5 — supervisor-managed analytics probe worker unit tests.

No Docker / GPU / pyds / gi. Reuses the scripted ``FakeMetadataPM`` shape
from ``test_deepstream_metadata_probe.py`` and asserts the additions for
``deepstream.analytics_probe.v0``:

- the package resolves and ``DeepStreamWorker._mode`` picks ``analytics_probe``,
- ``build_analytics_config(...)`` produces an INI body with both a ROI
  filter and a line-crossing section,
- ``build_metadata_docker_command(analytics_enabled=True, ...)`` injects
  the analytics CLI arg and rejects analytics-without-tracker,
- ``DeepStreamMetadataProcessManager(analytics_enabled=True)`` writes
  ``analytics_config.txt`` into the per-run work dir + uses the
  ``upstream-dsa-`` container prefix + cleans up on stop,
- the worker's status payload carries ``analytics_enabled=True``,
  ``has_analytics_metadata``, ROI / line counters, and bounded
  ``analytics_labels_sample``,
- tracker_probe / metadata_probe modes still work and do NOT advertise
  analytics fields,
- malformed analytics payloads do not crash the worker / consumer,
- redaction still works in analytics mode,
- no observations are emitted.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest
from app.observations.schemas import EligibleSession
from app.packages.resolver import resolve
from app.runtimes.base import WorkerState
from app.runtimes.deepstream_runtime import metadata_probe as meta_mod
from app.runtimes.deepstream_runtime.inference import build_analytics_config
from app.runtimes.deepstream_runtime.metadata_probe import (
    DeepStreamMetadataProcessManager,
    build_metadata_docker_command,
)
from app.runtimes.deepstream_runtime.worker import DeepStreamWorker

from tests._fakes import FakeApi
from tests.conftest import eligible_session_dict
from tests.test_deepstream_metadata_probe import FakeMetadataPM

_URL = "rtsp://127.0.0.1:18554/sess_abcdef0123456789"
_KEY = "abcdef0123456789"


def _session(**ov: object) -> EligibleSession:
    return EligibleSession.model_validate(eligible_session_dict(**ov))


def _frame_line(
    *,
    frame_num: int,
    objects: list[dict[str, Any]] | None = None,
    source_analytics: dict[str, Any] | None = None,
) -> str:
    payload: dict[str, Any] = {
        "runtime": "deepstream",
        "metadata_source": "pyds_pad_probe",
        "stream_session_id": None,
        "frame": {"frame_num": frame_num, "width": 1280, "height": 720},
        "objects": objects or [],
    }
    if source_analytics is not None:
        payload["source"] = {"analytics": source_analytics}
    return "UPSTREAM_METADATA=" + json.dumps(payload)


def _person(
    *,
    tracker_id: str | None = None,
    conf: float = 0.51,
    analytics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    obj: dict[str, Any] = {
        "class_id": 2,
        "label": "person",
        "confidence": conf,
        "bbox_pixels": {"left": 420.87, "top": 91.65, "width": 495.94, "height": 627.02},
        "bbox_normalized": {"x": 0.3288, "y": 0.1273, "w": 0.3875, "h": 0.8709},
        "tracker_id": tracker_id,
    }
    if analytics is not None:
        obj["debug"] = {"analytics": analytics}
    return obj


def _worker(api: FakeApi, settings, pm: FakeMetadataPM) -> DeepStreamWorker:  # type: ignore[no-untyped-def]
    return DeepStreamWorker(
        session=_session(internal_rtsp_url=_URL),
        settings=settings,
        api=api,
        package=resolve("deepstream.analytics_probe.v0"),
        pm_factory=lambda: pm,
    )


# --- mode dispatch ------------------------------------------------------


def test_analytics_package_selects_analytics_mode(fast_settings) -> None:  # type: ignore[no-untyped-def]
    w = _worker(FakeApi(), fast_settings, FakeMetadataPM([]))
    assert w._mode == "analytics_probe"


def test_tracker_and_metadata_modes_unchanged(fast_settings) -> None:  # type: ignore[no-untyped-def]
    """Regression: most-specific-wins ordering must keep
    metadata_probe / tracker_probe pointing at their own modes even now
    that analytics_probe exists above them in the ordering."""
    tracker = DeepStreamWorker(
        session=_session(internal_rtsp_url=_URL),
        settings=fast_settings,
        api=FakeApi(),
        package=resolve("deepstream.tracker_probe.v0"),
        pm_factory=lambda: FakeMetadataPM([]),
    )
    metadata = DeepStreamWorker(
        session=_session(internal_rtsp_url=_URL),
        settings=fast_settings,
        api=FakeApi(),
        package=resolve("deepstream.metadata_probe.v0"),
        pm_factory=lambda: FakeMetadataPM([]),
    )
    assert tracker._mode == "tracker_probe"
    assert metadata._mode == "metadata_probe"


# --- pure analytics config builder -------------------------------------


def test_build_analytics_config_has_roi_and_line_sections() -> None:
    body = build_analytics_config(frame_width=1280, frame_height=720)
    assert "[property]" in body
    assert "config-width=1280" in body
    assert "config-height=720" in body
    # ROI section uses the SDK's "roi-<LABEL>=<polygon-points>" key
    # convention (label in the KEY, points in the value). The SDK
    # parser strips the "roi-" prefix and uses the suffix as the ROI
    # label that shows up in objInROIcnt.
    assert "[roi-filtering-stream-0]" in body
    assert "roi-entry-roi=" in body
    assert "roi-RF=" not in body  # regression: old (broken) format
    # Line section uses "line-crossing-<LABEL>=<line-coords>;<dir-coords>".
    assert "[line-crossing-stream-0]" in body
    assert "line-crossing-entry-line=" in body
    # Inverse-roi=0 + class-id=-1 are required to count all classes.
    assert "inverse-roi=0" in body
    assert "class-id=-1" in body
    # Sanity: no tabs in an INI file (the SDK parser is whitespace-sensitive).
    assert "\t" not in body


def test_build_analytics_config_rejects_bad_dimensions() -> None:
    with pytest.raises(ValueError, match="must be > 0"):
        build_analytics_config(frame_width=0, frame_height=720)
    with pytest.raises(ValueError, match="must be > 0"):
        build_analytics_config(frame_width=1280, frame_height=-10)


def test_build_analytics_config_rejects_empty_labels() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        build_analytics_config(roi_label="")
    with pytest.raises(ValueError, match="non-empty"):
        build_analytics_config(line_label="")


# --- P4-D follow-up: geometry + class filter knobs --------------------


def test_default_roi_uses_loose_5pct_margin() -> None:
    """P4-D live verification showed the previous 20%/10% ROI margins
    were too tight (operator foot point at the bottom edge fell
    outside the ROI). Default is now 5% on each side so the ROI
    covers the central 90% x 90% of the frame."""
    body = build_analytics_config(frame_width=1280, frame_height=720)
    # 5% inset → rx0=64, ry0=36, rx1=1216, ry1=684.
    assert "roi-entry-roi=64;36;1216;36;1216;684;64;684" in body
    # The previous (broken) 20%/10% bounds must NOT show up — that's a
    # regression test for the foot-point-below-ROI live finding.
    assert "256;72;1024;72" not in body
    assert "256;648" not in body


def test_explicit_roi_margin_overrides_default() -> None:
    """A scene with a top-mounted camera and small feet might want
    a 0.0 margin (full frame). Probe knob honors it."""
    body = build_analytics_config(frame_width=1280, frame_height=720, roi_margin_pct=0.0)
    # 0% margin → polygon at (0, 0)..(1280, 720) — full frame.
    assert "roi-entry-roi=0;0;1280;0;1280;720;0;720" in body


def test_build_analytics_config_rejects_invalid_margin() -> None:
    """Out-of-range margins fail fast at config generation, not
    silently producing a malformed polygon."""
    with pytest.raises(ValueError, match=r"roi_margin_pct must be in"):
        build_analytics_config(roi_margin_pct=-0.01)
    with pytest.raises(ValueError, match=r"roi_margin_pct must be in"):
        build_analytics_config(roi_margin_pct=0.5)
    with pytest.raises(ValueError, match=r"roi_margin_pct must be in"):
        build_analytics_config(roi_margin_pct=1.0)


def test_default_class_ids_all_maps_to_minus_one() -> None:
    """P4-D follow-up: ``class_ids="all"`` (the probe default) emits
    ``class-id=-1`` so every detected class counts (TrafficCamNet
    sometimes misclassifies people as cars; we want every class to
    register in the ROI for the probe)."""
    body = build_analytics_config(class_ids="all")
    # Both sections use the same class filter.
    assert body.count("class-id=-1\n") == 2


def test_explicit_class_id_filters_to_one_class() -> None:
    body = build_analytics_config(class_ids="2")
    assert body.count("class-id=2\n") == 2
    assert "class-id=-1" not in body


def test_class_ids_minus_one_and_empty_map_to_all() -> None:
    """``"-1"`` is the literal nvdsanalytics value; ``""`` is a friendly
    no-op alias the operator might paste in via ``UPSTREAM_*`` env."""
    assert "class-id=-1\n" in build_analytics_config(class_ids="-1")
    assert "class-id=-1\n" in build_analytics_config(class_ids="")


def test_class_ids_rejects_comma_list() -> None:
    """nvdsanalytics' single-section ``class-id=`` takes a SINGLE int,
    so a comma-list would silently apply only one class if we passed
    it through. Reject with a clear error pointing at the operator
    knob so a typo fails fast."""
    with pytest.raises(ValueError, match="comma-lists are not supported"):
        build_analytics_config(class_ids="0,2")


def test_class_ids_rejects_non_integer() -> None:
    with pytest.raises(ValueError, match="single integer"):
        build_analytics_config(class_ids="person")
    with pytest.raises(ValueError, match="single integer"):
        build_analytics_config(class_ids="abc")


def test_settings_defaults_for_p4d_followup() -> None:
    """Pin the production defaults: ``roi_margin_pct=0.05`` (matching
    the probe-default geometry above) and ``class_ids="all"`` (matching
    the all-class probe default)."""
    from app.config import Settings

    s = Settings()
    assert s.deepstream_analytics_roi_margin_pct == 0.05
    assert s.deepstream_analytics_class_ids == "all"


# --- pure docker command builder ---------------------------------------


def test_docker_command_with_analytics_requires_tracker() -> None:
    with pytest.raises(ValueError, match="analytics_enabled requires tracker_enabled"):
        build_metadata_docker_command(
            work_dir="/tmp/wd",
            image="upstream-deepstream:8.0-pyds-v1.2.2",
            network="host",
            probe_timeout_seconds=60,
            container_name="upstream-dsa-test",
            sample_limit=3,
            stream_session_id="abc-123",
            rtsp_url=_URL,
            tracker_enabled=False,
            analytics_enabled=True,
        )


def test_docker_command_with_analytics_includes_analytics_arg() -> None:
    cmd = build_metadata_docker_command(
        work_dir="/tmp/wd",
        image="upstream-deepstream:8.0-pyds-v1.2.2",
        network="host",
        probe_timeout_seconds=60,
        container_name="upstream-dsa-test",
        sample_limit=3,
        stream_session_id="abc-123",
        rtsp_url=_URL,
        tracker_enabled=True,
        tracker_lib_path="/opt/.../libnvds_nvmultiobjecttracker.so",
        analytics_enabled=True,
    )
    inner = cmd[-1]
    assert "--analytics-config-path /tmp/wd/analytics_config.txt" in inner
    # Tracker args still present too (analytics builds on tracker).
    assert "--tracker-config-path" in inner


# --- process manager: analytics config write + cleanup -----------------


class _FakeProc:
    def __init__(self) -> None:
        self.returncode: int | None = 0
        self.stdout = None

    async def wait(self) -> int:
        return 0

    def terminate(self) -> None:  # pragma: no cover
        pass

    def kill(self) -> None:  # pragma: no cover
        pass


def test_analytics_pm_rejects_analytics_without_tracker(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Pre-flight: constructing a PM with analytics_enabled but no
    tracker raises so a typo in worker wiring fails fast."""
    with pytest.raises(ValueError, match="analytics_enabled requires tracker_enabled"):
        DeepStreamMetadataProcessManager(
            rtsp_url=_URL,
            image="upstream-deepstream:8.0-pyds-v1.2.2",
            network="host",
            rtsp_transport="tcp",
            probe_timeout_seconds=5,
            stop_timeout_seconds=0.01,
            model="trafficcamnet",
            output_base_dir=str(tmp_path),
            keep_temp_files=False,
            sample_limit=3,
            stream_session_id="abc-123",
            width=1280,
            height=720,
            tracker_enabled=False,
            analytics_enabled=True,
        )


async def test_analytics_pm_writes_analytics_config(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    async def _fake_exec(*_a: object, **_k: object):  # type: ignore[no-untyped-def]
        return _FakeProc()

    monkeypatch.setattr(meta_mod.asyncio, "create_subprocess_exec", _fake_exec)
    pm = DeepStreamMetadataProcessManager(
        rtsp_url=_URL,
        image="upstream-deepstream:8.0-pyds-v1.2.2",
        network="host",
        rtsp_transport="tcp",
        probe_timeout_seconds=5,
        stop_timeout_seconds=0.01,
        model="trafficcamnet",
        output_base_dir=str(tmp_path),
        keep_temp_files=False,
        sample_limit=3,
        stream_session_id="abc-123",
        width=1280,
        height=720,
        tracker_enabled=True,
        tracker_algorithm="iou",
        tracker_lib_path="/opt/.../libnvds_nvmultiobjecttracker.so",
        analytics_enabled=True,
    )
    await pm.start()
    wd = pm._work_dir
    assert wd is not None and wd.is_dir()
    # Analytics config + tracker config + pgie config + pyds app all
    # land in the work dir.
    assert (wd / "analytics_config.txt").exists()
    assert (wd / "tracker_config.yml").exists()
    assert (wd / "pgie.txt").exists()
    text = (wd / "analytics_config.txt").read_text(encoding="utf-8")
    assert "[roi-filtering-stream-0]" in text
    assert "[line-crossing-stream-0]" in text
    # Container name reflects analytics mode (-dsa-, not -dst- / -dsm-).
    assert pm._container.startswith("upstream-dsa-")
    cmd = pm.command()
    inner = cmd[-1]
    assert "--analytics-config-path" in inner
    await pm.stop()
    assert not wd.exists()  # analytics_config + tracker_config shredded


# --- lifecycle + status payload ----------------------------------------


async def test_analytics_lifecycle_emits_analytics_payload(fast_settings) -> None:  # type: ignore[no-untyped-def]
    api = FakeApi()
    # Three frames: first has no analytics, second has ROI hit, third
    # has line crossing + frame-level cumulative counters.
    pm = FakeMetadataPM(
        [
            _frame_line(frame_num=0, objects=[_person(tracker_id="100")]),
            _frame_line(
                frame_num=1,
                objects=[
                    _person(
                        tracker_id="100",
                        analytics={"roi_labels": ["entry-roi"]},
                    )
                ],
            ),
            _frame_line(
                frame_num=2,
                objects=[
                    _person(
                        tracker_id="100",
                        analytics={
                            "roi_labels": ["entry-roi"],
                            "line_labels": ["entry-line"],
                            "direction": "dirSouth",
                        },
                    )
                ],
                source_analytics={
                    "roi_counts": {"entry-roi": 1},
                    "line_crossing_cumulative": {"entry-line": 1},
                    "line_crossing_current": {"entry-line": 1},
                },
            ),
        ],
        exit_code=124,
    )
    w = _worker(api, fast_settings, pm)
    await asyncio.wait_for(w.run(), timeout=3)

    seq = api.status_sequence()
    assert seq[0] == "starting"
    assert "attached" in seq
    assert "processing" in seq
    assert seq[-1] == "stopped"
    assert w.state is WorkerState.STOPPED

    assert {s.package_id for s in api.statuses} == {"deepstream.analytics_probe.v0"}
    stopped = next(s for s in api.statuses if s.status == "stopped")
    p = stopped.payload

    assert p["mode"] == "analytics_probe"
    assert p["analytics_enabled"] is True
    # tracker fields still flow (analytics builds on tracker).
    assert p["tracker_enabled"] is True
    assert p["unique_tracker_ids_count"] == 1
    assert p["tracker_ids_sample"] == ["100"]
    # analytics evidence: object-level ROI on f1, ROI+LC+dir on f2;
    # frame-level ROI counts on f2, LC cumulative on f2.
    assert p["has_analytics_metadata"] is True
    assert p["roi_metadata_count"] == 1  # one frame had frame-level ROI counts
    assert p["line_metadata_count"] == 1  # one frame had LC cumulative
    assert p["direction_metadata_count"] == 1
    assert p["objects_with_roi_metadata"] == 2  # f1 + f2
    assert p["objects_with_line_metadata"] == 1  # f2 only
    # analytics_labels_sample is bounded; here we expect entry-roi, entry-line,
    # dirSouth.
    assert set(p["analytics_labels_sample"]) == {"entry-roi", "entry-line", "dirSouth"}
    assert p["analytics_labels_count"] == 3

    # P3.5 emits NO observations.
    assert api.observations == []


async def test_analytics_labels_sample_is_bounded(fast_settings) -> None:  # type: ignore[no-untyped-def]
    """Many distinct analytics labels must NOT all flow into the status
    payload; analytics_labels_sample is FIFO-capped at
    deepstream_analytics_labels_sample_limit."""
    api = FakeApi()
    settings = fast_settings.model_copy(update={"deepstream_analytics_labels_sample_limit": 3})
    # 5 distinct ROI labels across 5 frames.
    frames = [
        _frame_line(
            frame_num=i,
            objects=[
                _person(
                    tracker_id="t",
                    analytics={"roi_labels": [f"roi-{i}"]},
                )
            ],
        )
        for i in range(5)
    ]
    pm = FakeMetadataPM(frames, exit_code=124)
    w = _worker(api, settings, pm)
    await asyncio.wait_for(w.run(), timeout=3)
    stopped = next(s for s in api.statuses if s.status == "stopped")
    p = stopped.payload
    assert p["analytics_labels_count"] == 5
    assert len(p["analytics_labels_sample"]) == 3
    # First three observed labels are kept (FIFO bounded).
    assert p["analytics_labels_sample"] == ["roi-0", "roi-1", "roi-2"]


async def test_analytics_mode_when_no_analytics_metadata_flows(fast_settings) -> None:  # type: ignore[no-untyped-def]
    """Pipeline ran but no nvdsanalytics user-meta arrived (e.g. scene
    didn't cross the line / enter the ROI). Payload should still
    advertise analytics_enabled=true and have_analytics_metadata=false."""
    api = FakeApi()
    pm = FakeMetadataPM(
        [_frame_line(frame_num=0, objects=[_person(tracker_id="42")])],
        exit_code=124,
    )
    w = _worker(api, fast_settings, pm)
    await asyncio.wait_for(w.run(), timeout=3)
    stopped = next(s for s in api.statuses if s.status == "stopped")
    p = stopped.payload
    assert p["analytics_enabled"] is True
    assert p["has_analytics_metadata"] is False
    assert p["roi_metadata_count"] == 0
    assert p["objects_with_roi_metadata"] == 0
    assert p["analytics_labels_sample"] == []
    assert api.observations == []


async def test_tracker_mode_payload_has_no_analytics_fields(fast_settings) -> None:  # type: ignore[no-untyped-def]
    """tracker_probe mode (no nvdsanalytics) must NOT advertise
    analytics_enabled / has_analytics_metadata / *_metadata_count /
    analytics_labels_sample so operators can tell the two modes apart
    from the status row alone."""
    api = FakeApi()
    pm = FakeMetadataPM(
        [_frame_line(frame_num=0, objects=[_person(tracker_id="7")])],
        exit_code=124,
    )
    w = DeepStreamWorker(
        session=_session(internal_rtsp_url=_URL),
        settings=fast_settings,
        api=api,
        package=resolve("deepstream.tracker_probe.v0"),
        pm_factory=lambda: pm,
    )
    await asyncio.wait_for(w.run(), timeout=3)
    stopped = next(s for s in api.statuses if s.status == "stopped")
    p = stopped.payload
    assert p["mode"] == "tracker_probe"
    assert "analytics_enabled" not in p
    assert "has_analytics_metadata" not in p
    assert "roi_metadata_count" not in p
    assert "line_metadata_count" not in p
    assert "analytics_labels_sample" not in p
    # Tracker fields still present (regression).
    assert p["tracker_enabled"] is True
    assert api.observations == []


async def test_analytics_malformed_lines_dont_crash(fast_settings) -> None:  # type: ignore[no-untyped-def]
    """A malformed analytics blob (wrong type / missing keys) on either
    side of the frame must be skipped without crashing the worker."""
    api = FakeApi()
    # Object with analytics dict that has the wrong types AND a frame with
    # source.analytics that isn't a dict.
    bad_object_line = "UPSTREAM_METADATA=" + json.dumps(
        {
            "runtime": "deepstream",
            "metadata_source": "pyds_pad_probe",
            "stream_session_id": None,
            "frame": {"frame_num": 0, "width": 1280, "height": 720},
            "objects": [
                {
                    "class_id": 2,
                    "label": "person",
                    "confidence": 0.6,
                    "bbox_pixels": {"left": 0, "top": 0, "width": 100, "height": 100},
                    "bbox_normalized": {"x": 0, "y": 0, "w": 0.1, "h": 0.1},
                    "tracker_id": "1",
                    "debug": {"analytics": "this-should-be-a-dict-but-isnt"},
                }
            ],
        }
    )
    bad_frame_line = "UPSTREAM_METADATA=" + json.dumps(
        {
            "runtime": "deepstream",
            "metadata_source": "pyds_pad_probe",
            "stream_session_id": None,
            "frame": {"frame_num": 1, "width": 1280, "height": 720},
            "objects": [],
            "source": {"analytics": "this-should-be-a-dict-but-isnt"},
        }
    )
    pm = FakeMetadataPM([bad_object_line, bad_frame_line], exit_code=124)
    w = _worker(api, fast_settings, pm)
    await asyncio.wait_for(w.run(), timeout=3)
    # The worker shouldn't crash; both malformed-but-pydantic-valid frames
    # still count toward frames_seen and we never set has_analytics_metadata.
    stopped = next(s for s in api.statuses if s.status == "stopped")
    p = stopped.payload
    assert p["has_analytics_metadata"] is False
    assert p["roi_metadata_count"] == 0
    assert p["objects_with_roi_metadata"] == 0
    assert api.observations == []


async def test_analytics_failed_on_unsupported_sm_is_redacted(fast_settings) -> None:  # type: ignore[no-untyped-def]
    """Regression: the hard-fail / redaction path the metadata probe
    already exercises still works in analytics mode."""
    api = FakeApi()
    unsupported = (
        f"ERROR: [TRT]: IBuilder::buildSerializedNetwork for {_URL}: "
        "Error Code 1: Internal Error (Unsupported SM: 0xc00)"
    )
    pm = FakeMetadataPM([unsupported], exit_code=139)
    w = _worker(api, fast_settings, pm)
    await asyncio.wait_for(w.run(), timeout=3)
    failed = next(s for s in api.statuses if s.status == "failed")
    assert failed.error_code == "deepstream_unsupported_sm"
    assert _KEY not in (failed.error_message or "")
    assert "sess_<redacted>" in (failed.error_message or "")
    assert failed.payload["analytics_enabled"] is True
    assert failed.payload["has_analytics_metadata"] is False
    assert api.observations == []
