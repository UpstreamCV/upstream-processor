"""P3 — supervisor-managed tracker probe worker unit tests.

No Docker / GPU / pyds / gi. Reuses the scripted ``FakeMetadataPM`` shape
from ``test_deepstream_metadata_probe.py`` and asserts the additions for
``deepstream.tracker_probe.v0``:

- the package resolves and ``DeepStreamWorker._mode`` picks ``tracker_probe``,
- ``build_metadata_docker_command(tracker_enabled=True, ...)`` emits the
  expected tracker CLI args,
- ``DeepStreamMetadataProcessManager(tracker_enabled=True)`` writes
  ``tracker_config.yml`` into the work dir alongside ``pgie.txt`` /
  ``pyds_metadata_app.py`` and cleans it on stop,
- the worker's status payload carries ``tracker_enabled=True``,
  ``unique_tracker_ids_count``, and a bounded ``tracker_ids_sample`` when
  the producer emits real tracker ids,
- metadata_probe mode still works and does NOT advertise tracker fields,
- the unit ``build_tracker_config`` helper produces a YAML body the
  ``nvtracker`` element can consume,
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
from app.runtimes.deepstream_runtime.inference import build_tracker_config
from app.runtimes.deepstream_runtime.metadata_probe import (
    DeepStreamMetadataProcessManager,
    build_metadata_docker_command,
)
from app.runtimes.deepstream_runtime.worker import DeepStreamWorker

from tests._fakes import FakeApi
from tests.conftest import eligible_session_dict
from tests.test_deepstream_metadata_probe import FakeMetadataPM

_URL = "rtsp://127.0.0.1:18554/sess_abcdef0123456789"


def _session(**ov: object) -> EligibleSession:
    return EligibleSession.model_validate(eligible_session_dict(**ov))


def _frame_line(*, frame_num: int, objects: list[dict[str, Any]] | None = None) -> str:
    return "UPSTREAM_METADATA=" + json.dumps(
        {
            "runtime": "deepstream",
            "metadata_source": "pyds_pad_probe",
            "stream_session_id": None,
            "frame": {"frame_num": frame_num, "width": 1280, "height": 720},
            "objects": objects or [],
        }
    )


def _person(tracker_id: str | None, conf: float = 0.51) -> dict[str, Any]:
    return {
        "class_id": 2,
        "label": "person",
        "confidence": conf,
        "bbox_pixels": {"left": 420.87, "top": 91.65, "width": 495.94, "height": 627.02},
        "bbox_normalized": {"x": 0.3288, "y": 0.1273, "w": 0.3875, "h": 0.8709},
        "tracker_id": tracker_id,
    }


def _worker(api: FakeApi, settings, pm: FakeMetadataPM) -> DeepStreamWorker:  # type: ignore[no-untyped-def]
    return DeepStreamWorker(
        session=_session(internal_rtsp_url=_URL),
        settings=settings,
        api=api,
        package=resolve("deepstream.tracker_probe.v0"),
        pm_factory=lambda: pm,
    )


# --- mode dispatch ------------------------------------------------------


def test_tracker_package_selects_tracker_mode(fast_settings) -> None:  # type: ignore[no-untyped-def]
    w = _worker(FakeApi(), fast_settings, FakeMetadataPM([]))
    assert w._mode == "tracker_probe"


def test_metadata_package_still_selects_metadata_mode(fast_settings) -> None:  # type: ignore[no-untyped-def]
    w = DeepStreamWorker(
        session=_session(internal_rtsp_url=_URL),
        settings=fast_settings,
        api=FakeApi(),
        package=resolve("deepstream.metadata_probe.v0"),
        pm_factory=lambda: FakeMetadataPM([]),
    )
    assert w._mode == "metadata_probe"


# --- pure tracker config builder ---------------------------------------


def test_build_tracker_config_iou_contains_required_sections() -> None:
    body = build_tracker_config("iou")
    # nvtracker requires the %YAML:1.0 marker, the BaseConfig +
    # TargetManagement sections, and a DataAssociator (matching the DS 8.0
    # sample). Spot-check that all are present in our generated config.
    assert body.startswith("%YAML:1.0\n")
    assert "BaseConfig:" in body
    assert "TargetManagement:" in body
    assert "DataAssociator:" in body
    assert "minIouDiff4NewTarget" in body
    # No tabs (YAML is whitespace-sensitive).
    assert "\t" not in body


def test_build_tracker_config_rejects_unknown_algorithm() -> None:
    with pytest.raises(ValueError, match="unknown deepstream tracker algorithm"):
        build_tracker_config("definitely-not-real")


# --- pure docker command builder ---------------------------------------


def test_docker_command_without_tracker_omits_tracker_args() -> None:
    cmd = build_metadata_docker_command(
        work_dir="/tmp/wd",
        image="upstream-deepstream:8.0-pyds-v1.2.2",
        network="host",
        probe_timeout_seconds=60,
        container_name="upstream-dsm-test",
        sample_limit=3,
        stream_session_id="abc-123",
        rtsp_url=_URL,
    )
    inner = cmd[-1]
    assert "--tracker-config-path" not in inner
    assert "--tracker-lib-path" not in inner
    assert "--tracker-width" not in inner
    assert "tracker_config.yml" not in inner


def test_docker_command_with_tracker_includes_tracker_args() -> None:
    cmd = build_metadata_docker_command(
        work_dir="/tmp/wd",
        image="upstream-deepstream:8.0-pyds-v1.2.2",
        network="host",
        probe_timeout_seconds=60,
        container_name="upstream-dst-test",
        sample_limit=3,
        stream_session_id="abc-123",
        rtsp_url=_URL,
        tracker_enabled=True,
        tracker_lib_path="/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so",
        tracker_width=960,
        tracker_height=544,
    )
    inner = cmd[-1]
    assert "--tracker-config-path /tmp/wd/tracker_config.yml" in inner
    lib = "/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so"
    assert f"--tracker-lib-path {lib}" in inner
    assert "--tracker-width 960" in inner
    assert "--tracker-height 544" in inner


# --- process manager: tracker config write + cleanup -------------------


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


async def test_tracker_pm_writes_tracker_config(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
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
        tracker_lib_path="/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so",
        tracker_width=960,
        tracker_height=544,
    )
    await pm.start()
    wd = pm._work_dir
    assert wd is not None and wd.is_dir()
    # Tracker config lands alongside pgie.txt + the pyds app.
    tracker_yml = wd / "tracker_config.yml"
    assert tracker_yml.exists()
    text = tracker_yml.read_text(encoding="utf-8")
    assert text.startswith("%YAML:1.0\n")
    assert "BaseConfig:" in text
    # Container name reflects tracker mode.
    assert pm._container.startswith("upstream-dst-")
    # Docker command includes the tracker args.
    cmd = pm.command()
    inner = cmd[-1]
    assert "--tracker-config-path" in inner
    assert "--tracker-width 960" in inner
    await pm.stop()
    assert not wd.exists()  # tracker yaml shredded with the rest of the dir


def test_tracker_pm_unknown_algorithm_raises(tmp_path) -> None:  # type: ignore[no-untyped-def]
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
        tracker_algorithm="not-real",
        tracker_lib_path="/opt/.../libnvds_nvmultiobjecttracker.so",
    )
    with pytest.raises(ValueError, match="unknown deepstream tracker algorithm"):
        pm._prepare_work_dir()


async def test_metadata_pm_without_tracker_does_not_write_tracker_yml(  # type: ignore[no-untyped-def]
    tmp_path, monkeypatch
) -> None:
    """Regression: metadata_probe (tracker_enabled=False) must NOT create a
    tracker_config.yml in the work dir — ensures the tracker scaffolding
    stays inert in the legacy mode."""

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
    )
    await pm.start()
    try:
        wd = pm._work_dir
        assert wd is not None
        assert (wd / "pgie.txt").exists()
        assert not (wd / "tracker_config.yml").exists()
        assert pm._container.startswith("upstream-dsm-")
    finally:
        await pm.stop()


# --- lifecycle + status payload ----------------------------------------


async def test_tracker_lifecycle_emits_tracker_payload(fast_settings) -> None:  # type: ignore[no-untyped-def]
    api = FakeApi()
    # Three frames with tracker ids "101", "101", "102" — two unique tracker
    # ids overall, "101" stable across the first two frames.
    pm = FakeMetadataPM(
        [
            _frame_line(frame_num=0, objects=[_person("101")]),
            _frame_line(frame_num=1, objects=[_person("101")]),
            _frame_line(frame_num=2, objects=[_person("102")]),
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

    # tracker_probe row identifies the right package
    assert {s.package_id for s in api.statuses} == {"deepstream.tracker_probe.v0"}

    stopped = next(s for s in api.statuses if s.status == "stopped")
    p = stopped.payload
    assert p["mode"] == "tracker_probe"
    assert p["tracker_enabled"] is True
    assert p["tracker_algorithm"] == "iou"
    assert p["has_tracker_id"] is True
    assert p["unique_tracker_ids_count"] == 2
    assert sorted(p["tracker_ids_sample"]) == ["101", "102"]
    # frame counters still flow.
    assert p["frames_seen"] == 3
    assert p["frames_with_objects"] == 3
    assert p["objects_seen"] == 3

    # P3 emits NO observations.
    assert api.observations == []


async def test_tracker_ids_sample_is_bounded(fast_settings) -> None:  # type: ignore[no-untyped-def]
    """unique_tracker_ids_count is unbounded; tracker_ids_sample must cap
    at deepstream_tracker_ids_sample_limit so the status payload stays
    tiny under busy scenes."""
    api = FakeApi()
    # Force a small sample cap.
    settings = fast_settings.model_copy(update={"deepstream_tracker_ids_sample_limit": 3})
    # 6 unique tracker ids — sample should hold only the first 3.
    frames = [_frame_line(frame_num=i, objects=[_person(str(100 + i))]) for i in range(6)]
    pm = FakeMetadataPM(frames, exit_code=124)
    w = _worker(api, settings, pm)
    await asyncio.wait_for(w.run(), timeout=3)

    stopped = next(s for s in api.statuses if s.status == "stopped")
    p = stopped.payload
    assert p["unique_tracker_ids_count"] == 6
    assert len(p["tracker_ids_sample"]) == 3
    # First three observed ids are the ones captured (FIFO bounded).
    assert p["tracker_ids_sample"] == ["100", "101", "102"]


async def test_metadata_mode_payload_has_no_tracker_fields(fast_settings) -> None:  # type: ignore[no-untyped-def]
    """metadata_probe mode (no nvtracker) must NOT advertise
    tracker_enabled / unique_tracker_ids_count / tracker_ids_sample so
    operators can tell the two modes apart from the status row alone."""
    api = FakeApi()
    pm = FakeMetadataPM(
        [_frame_line(frame_num=0, objects=[_person(None)])],
        exit_code=124,
    )
    w = DeepStreamWorker(
        session=_session(internal_rtsp_url=_URL),
        settings=fast_settings,
        api=api,
        package=resolve("deepstream.metadata_probe.v0"),
        pm_factory=lambda: pm,
    )
    await asyncio.wait_for(w.run(), timeout=3)
    stopped = next(s for s in api.statuses if s.status == "stopped")
    p = stopped.payload
    assert p["mode"] == "metadata_probe"
    assert "tracker_enabled" not in p
    assert "tracker_algorithm" not in p
    assert "unique_tracker_ids_count" not in p
    assert "tracker_ids_sample" not in p
    # has_tracker_id stays — it's metadata_probe's "did the producer set any
    # tracker_id?" signal, and the answer here is False.
    assert p["has_tracker_id"] is False
    assert api.observations == []
