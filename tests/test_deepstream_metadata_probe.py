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


def test_build_metadata_docker_command_pins_gpu_device() -> None:
    # UPSTREAM_DEEPSTREAM_GPUS="device=1" pins the metadata/analytics worker
    # to physical GPU 1 (only that card visible inside the container).
    cmd = build_metadata_docker_command(
        work_dir="/tmp/wd",
        image="img",
        network="host",
        probe_timeout_seconds=60,
        container_name="upstream-dsm-test",
        sample_limit=3,
        stream_session_id="abc-123",
        rtsp_url=_URL,
        gpus="device=1",
    )
    assert cmd[cmd.index("--gpus") + 1] == "device=1"


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


# --- P2 cleanup: shutdown hardening -------------------------------------


class _SlowReadlinePM(FakeMetadataPM):
    """A PM whose ``readline`` parks forever until ``stop`` is called.

    Lets us prove that ``request_stop`` wakes the consumer immediately
    (by closing the queue from the worker side) rather than waiting for
    the reader's next poll iteration. Also tracks ``stop_call_count`` for
    idempotency."""

    def __init__(
        self,
        seed_lines: list[str] | None = None,
        *,
        exit_code: int = 124,
    ) -> None:
        super().__init__(seed_lines or [], exit_code=exit_code)
        self.stop_call_count = 0
        self.queue_closed_at_stop: bool | None = None
        # When the test wants to inspect queue state when stop() is called,
        # it sets ``observe_worker`` to the worker instance.
        self.observe_worker: object | None = None
        self._stop_event = asyncio.Event()

    async def readline(self) -> str | None:
        await asyncio.sleep(0)
        if self._lines:
            return self._lines.pop(0)
        if self.stopped:
            return None
        # Park until stop() signals via the event (no busy-loop).
        await self._stop_event.wait()
        self._rc = self.exit_code
        return None

    async def stop(self) -> None:
        self.stop_call_count += 1
        self._stop_event.set()
        # Snapshot: was the worker's metadata bridge queue already closed
        # by the time PM.stop was invoked?  (request_stop should have
        # closed it BEFORE the worker reached _safe_stop_pm.)
        worker = self.observe_worker
        if worker is not None and self.queue_closed_at_stop is None:
            q = getattr(worker, "_metadata_queue", None)
            if q is not None:
                self.queue_closed_at_stop = q.stats().closed
        self.stopped = True
        if self._rc is None:
            self._rc = self.exit_code


async def test_request_stop_closes_queue_synchronously(fast_settings) -> None:  # type: ignore[no-untyped-def]
    """``request_stop`` must close the metadata queue immediately (no
    await), so a consumer parked in ``asyncio.to_thread(queue.Queue.get)``
    wakes within milliseconds — not after the reader's next poll."""
    api = FakeApi()
    pm = _SlowReadlinePM(seed_lines=[_frame_line(frame_num=0, objects=[_person_object()])])
    w = _worker(api, fast_settings, pm)
    pm.observe_worker = w
    task = asyncio.create_task(w.run())

    # Wait until _drive_metadata_probe has built the queue and started
    # gather(). The queue attribute is set synchronously inside the
    # driver before reader/consumer kick off.
    for _ in range(200):
        if w._metadata_queue is not None:
            break
        await asyncio.sleep(0.01)
    assert w._metadata_queue is not None
    queue = w._metadata_queue
    assert queue.stats().closed is False  # not yet closed

    # request_stop must close the queue immediately (sync, no await).
    w.request_stop()
    assert queue.stats().closed is True

    # Worker shuts down promptly (reader/consumer woken, PM stop runs).
    await asyncio.wait_for(task, timeout=2.0)
    assert w.state is WorkerState.STOPPED
    assert pm.stopped is True
    # PM.stop was called AFTER the queue was already closed.
    assert pm.queue_closed_at_stop is True
    # Queue handle was cleared after the driver finished.
    assert w._metadata_queue is None
    assert api.observations == []


async def test_request_stop_wakes_consumer_with_no_more_metadata(  # type: ignore[no-untyped-def]
    fast_settings,
) -> None:
    """If no metadata is in flight when request_stop fires, the consumer
    is parked on the queue; close() must wake it so the worker can finish
    and post stopped (instead of hanging until the reader's poll cycle)."""
    api = FakeApi()
    pm = _SlowReadlinePM(seed_lines=[])  # no metadata at all
    w = _worker(api, fast_settings, pm)
    pm.observe_worker = w
    task = asyncio.create_task(w.run())

    for _ in range(200):
        if w._metadata_queue is not None:
            break
        await asyncio.sleep(0.01)
    w.request_stop()
    await asyncio.wait_for(task, timeout=2.0)

    seq = api.status_sequence()
    assert seq[0] == "starting"
    assert seq[-1] == "stopped"  # graceful even with no metadata arrived
    assert "attached" not in seq  # we never got a valid frame
    assert pm.stopped is True
    assert api.observations == []


async def test_pm_stop_is_idempotent(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """Calling ``DeepStreamMetadataProcessManager.stop()`` twice does
    not raise, does not re-issue ``docker stop``, and leaves the work
    dir gone after the first call."""
    docker_stop_invocations: list[tuple[object, ...]] = []

    class _StopperProc:
        async def wait(self) -> int:
            return 0

    class _RunningProc:
        def __init__(self) -> None:
            self.returncode: int | None = 0
            self.stdout = None

        async def wait(self) -> int:
            return 0

        def terminate(self) -> None:  # pragma: no cover - already done
            pass

        def kill(self) -> None:  # pragma: no cover
            pass

    async def _fake_exec(*args: object, **_k: object):  # type: ignore[no-untyped-def]
        if args and args[0] == "docker" and len(args) > 1 and args[1] == "stop":
            docker_stop_invocations.append(args)
            return _StopperProc()
        return _RunningProc()

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

    await pm.stop()
    assert not wd.exists()
    assert pm._work_dir is None  # cleanup nulls it for idempotency
    assert len(docker_stop_invocations) == 1
    assert pm._stopped is True

    # Second call: no exception, no extra docker stop, still no work dir.
    await pm.stop()
    assert len(docker_stop_invocations) == 1  # not re-invoked
    assert pm._work_dir is None


async def test_pm_keep_temp_files_preserves_workdir(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    class _SP:
        async def wait(self) -> int:
            return 0

    class _RP:
        def __init__(self) -> None:
            self.returncode: int | None = 0
            self.stdout = None

        async def wait(self) -> int:
            return 0

        def terminate(self) -> None:
            pass

        def kill(self) -> None:
            pass

    async def _fake_exec(*args: object, **_k: object):  # type: ignore[no-untyped-def]
        if args and args[0] == "docker" and len(args) > 1 and args[1] == "stop":
            return _SP()
        return _RP()

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
        keep_temp_files=True,
        sample_limit=3,
        stream_session_id="abc-123",
        width=1280,
        height=720,
    )
    await pm.start()
    wd = pm._work_dir
    assert wd is not None and wd.is_dir()
    await pm.stop()
    # keep_temp_files=True must preserve the dir AND not null out the path
    # (operator may want to inspect contents post-run).
    assert wd.exists()
    assert pm._work_dir is wd


async def test_pm_stop_bounded_when_killed_proc_does_not_reap(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """Regression: a long-running --gpus container killed mid-stream can leave
    the docker-run client stalled on teardown, so ``self._proc.wait()`` after
    kill never returns. stop() must NOT hang on that wait forever -- worker
    teardown is awaited by the supervisor, so an unbounded wait wedges the
    whole reconcile loop (observed live after the continuous-mode change).

    Model a proc whose wait() never returns and whose terminate() doesn't make
    it exit (so the kill path runs), then assert stop() still completes within
    a bounded time."""

    class _HangingProc:
        def __init__(self) -> None:
            self.returncode: int | None = None  # stays alive -> kill path runs
            self.stdout = None

        async def wait(self) -> int:
            await asyncio.sleep(3600)  # never returns within the test window
            return -9

        def terminate(self) -> None:
            pass  # does not cause exit -> the first wait_for times out

        def kill(self) -> None:
            pass  # SIGKILL "sent" but wait() still hangs (stalled reap)

    class _QuickStopper:
        async def wait(self) -> int:
            return 0

    async def _fake_exec(*args: object, **_k: object):  # type: ignore[no-untyped-def]
        if args and args[0] == "docker" and len(args) > 1 and args[1] == "stop":
            return _QuickStopper()
        return _HangingProc()

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
    # The whole point: stop() must return promptly despite the hanging reap.
    # The outer wait_for guards the suite so a regression fails loudly instead
    # of stalling forever. stop_timeout=0.01 -> two ~10ms timeouts inside.
    await asyncio.wait_for(pm.stop(), timeout=2.0)
