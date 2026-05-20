"""P2 — supervisor-managed metadata probe worker unit tests.

No Docker / GPU / pyds / gi. A scripted ``ProcessManager`` double feeds the
worker the exact structured-stdout shape the in-container pyds metadata
app emits (``UPSTREAM_METADATA=<json>`` / ``UPSTREAM_METADATA_SUMMARY=<json>``).
Asserts the worker:

- selects ``metadata_probe`` mode from the resolved package,
- builds the BoundedBridgeQueue / BridgeQueueMetadataSource seam,
- validates each metadata line into ``NeutralFrameMetadata``,
- posts ``starting → attached → processing → stopped`` with the right
  payload counters and ``queue_stats``,
- treats ``Unsupported SM`` log lines as ``failed`` (with redacted
  ``error_message`` + ``deepstream_unsupported_sm``),
- skips malformed JSON without crashing,
- emits **zero** observations.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Iterable

import pytest
from app.observations.schemas import EligibleSession
from app.packages.resolver import resolve
from app.runtimes.base import WorkerState
from app.runtimes.deepstream_runtime import metadata_probe as meta_mod
from app.runtimes.deepstream_runtime.metadata_probe import (
    DeepStreamMetadataProcessManager,
    build_metadata_docker_command,
)
from app.runtimes.deepstream_runtime.worker import (
    DeepStreamWorker,
    parse_metadata_line,
)

from tests._fakes import FakeApi
from tests.conftest import eligible_session_dict

_URL = "rtsp://127.0.0.1:18554/sess_abcdef0123456789"
_KEY = "abcdef0123456789"


def _frame_line(
    *,
    frame_num: int,
    objects: list[dict[str, object]] | None = None,
) -> str:
    payload = {
        "runtime": "deepstream",
        "metadata_source": "pyds_pad_probe",
        "stream_session_id": None,
        "frame": {"frame_num": frame_num, "width": 1280, "height": 720},
        "objects": objects or [],
    }
    return "UPSTREAM_METADATA=" + json.dumps(payload)


def _person_object(conf: float = 0.51) -> dict[str, object]:
    return {
        "class_id": 2,
        "label": "person",
        "confidence": conf,
        "bbox_pixels": {"left": 420.87, "top": 91.65, "width": 495.94, "height": 627.02},
        "bbox_normalized": {"x": 0.3288, "y": 0.1273, "w": 0.3875, "h": 0.8709},
        "tracker_id": None,
    }


_SUMMARY_LINE = "UPSTREAM_METADATA_SUMMARY=" + json.dumps(
    {
        "runtime": "deepstream",
        "metadata_source": "pyds_pad_probe",
        "duration_seconds": 5,
        "frames_seen": 3,
        "frames_with_objects": 2,
        "objects_seen": 2,
        "has_bbox": True,
        "has_confidence": True,
        "has_tracker_id": False,
        "class_counts": {"2": 2},
        "label_counts": {"person": 2},
        "metadata_emitted": 3,
        "samples_emitted": 2,
    }
)

_UNSUPPORTED_SM = (
    f"ERROR: [TRT]: IBuilder::buildSerializedNetwork for {_URL}: "
    "Error Code 1: Internal Error (Unsupported SM: 0xc00)"
)


def _session(**ov: object) -> EligibleSession:
    return EligibleSession.model_validate(eligible_session_dict(**ov))


class FakeMetadataPM:
    """ProcessManager double for the metadata probe — scripted stdout
    lines, no Docker / GPU / pyds."""

    def __init__(self, lines: Iterable[str], *, exit_code: int = 124) -> None:
        self._lines = list(lines)
        self.exit_code = exit_code
        self.started = False
        self.stopped = False
        self._rc: int | None = None
        self.block_until_stop = False

    async def start(self) -> None:
        self.started = True

    async def readline(self) -> str | None:
        await asyncio.sleep(0)
        if self._lines:
            return self._lines.pop(0)
        if self.block_until_stop and not self.stopped:
            await asyncio.sleep(0.01)
            return ""
        self._rc = self.exit_code
        return None

    async def stop(self) -> None:
        self.stopped = True
        if self._rc is None:
            self._rc = self.exit_code

    @property
    def returncode(self) -> int | None:
        return self._rc


def _worker(
    api: FakeApi,
    settings,  # type: ignore[no-untyped-def]
    pm: FakeMetadataPM,
    **sess_ov: object,
) -> DeepStreamWorker:
    return DeepStreamWorker(
        session=_session(internal_rtsp_url=_URL, **sess_ov),
        settings=settings,
        api=api,
        package=resolve("deepstream.metadata_probe.v0"),
        pm_factory=lambda: pm,
    )


# --- mode selection -----------------------------------------------------


def test_metadata_package_selects_metadata_mode(fast_settings) -> None:  # type: ignore[no-untyped-def]
    w = _worker(FakeApi(), fast_settings, FakeMetadataPM([]))
    assert w._mode == "metadata_probe"


def test_decode_inference_modes_unchanged(fast_settings) -> None:  # type: ignore[no-untyped-def]
    """A package with neither metadata_probe nor inference_probe support
    still maps to decode_probe — guards against regression in
    mode-dispatch ordering."""
    decode_pkg = resolve("deepstream.decode_probe.v0")
    inference_pkg = resolve("deepstream.inference_probe.v0")
    decode = DeepStreamWorker(
        session=_session(internal_rtsp_url=_URL),
        settings=fast_settings,
        api=FakeApi(),
        package=decode_pkg,
        pm_factory=lambda: FakeMetadataPM([]),
    )
    inference = DeepStreamWorker(
        session=_session(internal_rtsp_url=_URL),
        settings=fast_settings,
        api=FakeApi(),
        package=inference_pkg,
        pm_factory=lambda: FakeMetadataPM([]),
    )
    assert decode._mode == "decode_probe"
    assert inference._mode == "inference_probe"


# --- pure parser --------------------------------------------------------


def test_parse_metadata_line_routes_correctly() -> None:
    f = _frame_line(frame_num=0)
    assert parse_metadata_line(f) == ("frame", f.split("=", 1)[1])
    assert parse_metadata_line(_SUMMARY_LINE) == (
        "summary",
        _SUMMARY_LINE.split("=", 1)[1],
    )
    assert parse_metadata_line("some unrelated stdout") is None
    assert parse_metadata_line("") is None
    # A line that *contains* the marker mid-string must NOT match.
    assert parse_metadata_line("noise UPSTREAM_METADATA={}") is None


# --- pure command builder ----------------------------------------------


def test_build_metadata_docker_command_has_gpus_network_and_mount() -> None:
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
    assert cmd[:3] == ["docker", "run", "--rm"]
    assert "--gpus" in cmd and cmd[cmd.index("--gpus") + 1] == "all"
    assert "--network" in cmd and cmd[cmd.index("--network") + 1] == "host"
    assert "-v" in cmd and cmd[cmd.index("-v") + 1] == "/tmp/wd:/tmp/wd"
    assert "upstream-deepstream:8.0-pyds-v1.2.2" in cmd
    # Pythondontwritebytecode env present (prevents root-owned __pycache__
    # in the bind-mounted host work dir).
    assert any(arg == "PYTHONDONTWRITEBYTECODE=1" for arg in cmd)
    # The raw RTSP URL is passed via env var, not echoed elsewhere.
    rtsp_env = next(arg for arg in cmd if arg.startswith("RTSP_URL="))
    assert _URL in rtsp_env
    inner = cmd[-1]
    assert "python3 -B /tmp/wd/pyds_metadata_app.py" in inner
    assert "--pgie-config /tmp/wd/pgie.txt" in inner
    assert "--seconds 60" in inner
    assert "--sample-limit 3" in inner
    assert "--stream-session-id abc-123" in inner


# --- lifecycle ----------------------------------------------------------


async def test_metadata_lifecycle_starting_attached_processing_stopped(
    fast_settings,  # type: ignore[no-untyped-def]
) -> None:
    api = FakeApi()
    # Mix: empty frame, two with a person, one empty, summary, then EOF.
    pm = FakeMetadataPM(
        [
            _frame_line(frame_num=0),
            _frame_line(frame_num=1, objects=[_person_object(0.51)]),
            _frame_line(frame_num=2, objects=[_person_object(0.52)]),
            _frame_line(frame_num=3),
            _SUMMARY_LINE,
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
    assert pm.started and pm.stopped

    # status rows report the DeepStream metadata-probe runtime/package
    assert {s.runtime for s in api.statuses} == {"deepstream"}
    assert {s.package_id for s in api.statuses} == {"deepstream.metadata_probe.v0"}

    stopped = next(s for s in api.statuses if s.status == "stopped")
    p = stopped.payload
    assert p["mode"] == "metadata_probe"
    assert p["metadata_enabled"] is True
    assert p["metadata_source"] == "pyds_pad_probe"
    assert p["frames_seen"] == 4
    assert p["frames_with_objects"] == 2
    assert p["objects_seen"] == 2
    assert p["class_counts"] == {2: 2}
    assert p["label_counts"] == {"person": 2}
    assert p["has_bbox"] is True and p["has_confidence"] is True
    assert p["has_tracker_id"] is False
    assert p["sample_count"] == 2
    # queue_stats present and consistent with accepted/consumed.
    qs = p["queue_stats"]
    assert qs["accepted"] == 4 == qs["consumed"]
    assert qs["dropped"] == 0 and qs["closed"] is True
    # summary line was captured and surfaced.
    assert p["app_summary"]["label_counts"] == {"person": 2}

    # P2 emits NO observations.
    assert api.observations == []


async def test_metadata_failed_on_unsupported_sm_is_redacted(
    fast_settings,  # type: ignore[no-untyped-def]
) -> None:
    api = FakeApi()
    pm = FakeMetadataPM([_UNSUPPORTED_SM], exit_code=139)
    w = _worker(api, fast_settings, pm)
    await asyncio.wait_for(w.run(), timeout=3)

    seq = api.status_sequence()
    assert seq[0] == "starting"
    assert "failed" in seq
    assert "stopped" not in seq
    assert w.state is WorkerState.FAILED
    failed = next(s for s in api.statuses if s.status == "failed")
    assert failed.error_code == "deepstream_unsupported_sm"
    assert _KEY not in (failed.error_message or "")
    assert "sess_<redacted>" in (failed.error_message or "")
    # status payload still carries the empty counters + queue_stats.
    assert failed.payload["metadata_enabled"] is True
    assert failed.payload["frames_seen"] == 0
    assert "queue_stats" in failed.payload
    assert api.observations == []


async def test_metadata_failed_on_nonzero_exit_no_metadata(
    fast_settings,  # type: ignore[no-untyped-def]
) -> None:
    api = FakeApi()
    pm = FakeMetadataPM(["some random log line", "ERROR: oh no"], exit_code=1)
    w = _worker(api, fast_settings, pm)
    await asyncio.wait_for(w.run(), timeout=3)
    seq = api.status_sequence()
    assert seq[0] == "starting"
    assert "failed" in seq
    failed = next(s for s in api.statuses if s.status == "failed")
    assert failed.error_code == "deepstream_exit"
    assert "metadata process rc=1" in (failed.error_message or "")
    assert api.observations == []


async def test_metadata_skips_malformed_lines(fast_settings) -> None:  # type: ignore[no-untyped-def]
    api = FakeApi()
    # Malformed JSON for one frame line, schema-invalid for another, then
    # a valid one. The worker must reach "processing" thanks to the valid
    # line and skip the bad ones without crashing.
    pm = FakeMetadataPM(
        [
            "UPSTREAM_METADATA=not-json",  # malformed JSON
            'UPSTREAM_METADATA={"runtime":""}',  # invalid (blank runtime)
            _frame_line(frame_num=99, objects=[_person_object(0.6)]),
        ],
        exit_code=124,
    )
    w = _worker(api, fast_settings, pm)
    await asyncio.wait_for(w.run(), timeout=3)

    seq = api.status_sequence()
    assert "attached" in seq and "processing" in seq
    assert seq[-1] == "stopped"
    stopped = next(s for s in api.statuses if s.status == "stopped")
    assert stopped.payload["frames_seen"] == 1
    assert stopped.payload["objects_seen"] == 1
    assert api.observations == []


async def test_metadata_missing_rtsp_url_fails_fast(fast_settings) -> None:  # type: ignore[no-untyped-def]
    api = FakeApi()
    pm = FakeMetadataPM([])
    w = DeepStreamWorker(
        session=_session(internal_rtsp_url=""),
        settings=fast_settings,
        api=api,
        package=resolve("deepstream.metadata_probe.v0"),
        pm_factory=lambda: pm,
    )
    await asyncio.wait_for(w.run(), timeout=3)
    assert api.status_sequence() == ["failed"]
    assert api.statuses[0].error_code == "no_rtsp_url"
    assert not pm.started
    assert api.observations == []


async def test_metadata_graceful_request_stop(fast_settings) -> None:  # type: ignore[no-untyped-def]
    """When the supervisor calls request_stop() mid-stream, the worker
    must close the queue, drain the consumer, and post stopped."""
    api = FakeApi()
    pm = FakeMetadataPM(
        [
            _frame_line(frame_num=0, objects=[_person_object()]),
        ]
    )
    pm.block_until_stop = True
    w = _worker(api, fast_settings, pm)
    task = asyncio.create_task(w.run())

    for _ in range(300):
        if "processing" in api.status_sequence():
            break
        await asyncio.sleep(0.01)
    w.request_stop()
    await asyncio.wait_for(task, timeout=3)
    seq = api.status_sequence()
    assert "attached" in seq and "processing" in seq
    assert seq[-1] == "stopped"
    assert w.state is WorkerState.STOPPED
    assert api.observations == []


# --- real PM: temp work dir copy of pyds_metadata_app.py ----------------


class _FakeProc:
    def __init__(self) -> None:
        self.returncode: int | None = 0
        self.stdout = None

    async def wait(self) -> int:
        return 0

    def terminate(self) -> None:  # pragma: no cover - returncode already set
        pass

    def kill(self) -> None:  # pragma: no cover
        pass


async def test_metadata_pm_writes_app_and_pgie_then_cleans(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
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
    wd = pm._work_dir
    assert wd is not None and wd.is_dir()
    # The pyds app + pgie config land in the work dir.
    assert (wd / "pyds_metadata_app.py").exists()
    assert (wd / "pgie.txt").exists()
    # The pgie config references the bundled TrafficCamNet INT8 path.
    pgie_text = (wd / "pgie.txt").read_text(encoding="utf-8")
    assert "resnet18_trafficcamnet_pruned.onnx" in pgie_text
    assert "cal_trt.bin" in pgie_text
    cmd = pm.command()
    assert "--gpus" in cmd and "-v" in cmd
    await pm.stop()
    assert not wd.exists()  # cleaned up by default


def test_metadata_pm_unknown_model_raises(tmp_path) -> None:  # type: ignore[no-untyped-def]
    pm = DeepStreamMetadataProcessManager(
        rtsp_url=_URL,
        image="upstream-deepstream:8.0-pyds-v1.2.2",
        network="host",
        rtsp_transport="tcp",
        probe_timeout_seconds=5,
        stop_timeout_seconds=0.01,
        model="bogus-model",
        output_base_dir=str(tmp_path),
        keep_temp_files=False,
        sample_limit=3,
        stream_session_id="abc-123",
        width=1280,
        height=720,
    )
    with pytest.raises(ValueError, match="unknown deepstream inference model"):
        pm._prepare_work_dir()
