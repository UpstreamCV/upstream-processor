from __future__ import annotations

import asyncio
from collections.abc import Iterable

import pytest
from app.observations.schemas import EligibleSession
from app.packages.resolver import resolve
from app.redact import redact_rtsp, scrub_rtsp_in_text
from app.runtimes.base import WorkerState
from app.runtimes.deepstream_runtime import process_manager as pm_mod
from app.runtimes.deepstream_runtime.process_manager import (
    DeepStreamProcessManager,
    build_docker_command,
)
from app.runtimes.deepstream_runtime.worker import DeepStreamWorker

from tests._fakes import FakeApi
from tests.conftest import eligible_session_dict

_URL = "rtsp://127.0.0.1:18554/sess_abcdef0123456789"

_FPS = "/GstFPSDisplaySink: last-message = rendered: {n}, dropped: 0, current: 30.1, average: 30.2"


def _session(**ov: object) -> EligibleSession:
    return EligibleSession.model_validate(eligible_session_dict(**ov))


class FakePM:
    """ProcessManager double — scripted lines, no Docker/GPU."""

    def __init__(self, lines: Iterable[str], exit_code: int = 0) -> None:
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
            return ""  # keep the loop alive until request_stop
        self._rc = self.exit_code
        return None

    async def stop(self) -> None:
        self.stopped = True
        if self._rc is None:
            self._rc = self.exit_code

    @property
    def returncode(self) -> int | None:
        return self._rc


def _worker(api: FakeApi, settings, pm: FakePM, **sess_ov: object) -> DeepStreamWorker:  # type: ignore[no-untyped-def]
    return DeepStreamWorker(
        session=_session(internal_rtsp_url=_URL, **sess_ov),
        settings=settings,
        api=api,
        package=resolve("deepstream.decode_probe.v0"),
        pm_factory=lambda: pm,
    )


# --- redaction ------------------------------------------------------------


def test_redact_rtsp_masks_stream_key() -> None:
    assert redact_rtsp(_URL) == "rtsp://127.0.0.1:18554/sess_<redacted>"
    assert "abcdef0123456789" not in redact_rtsp(_URL)


def test_scrub_rtsp_in_text() -> None:
    line = f'... rtspsrc location="{_URL}" protocols=tcp ...'
    out = scrub_rtsp_in_text(line)
    assert "abcdef0123456789" not in out
    assert "sess_<redacted>" in out


# --- command builder ------------------------------------------------------


def test_command_builder_has_gpus_and_network_host() -> None:
    cmd = build_docker_command(
        rtsp_url=_URL,
        image="nvcr.io/nvidia/deepstream:7.1-samples-multiarch",
        network="host",
        rtsp_transport="tcp",
        probe_timeout_seconds=30,
        container_name="upstream-ds-test",
    )
    assert cmd[:3] == ["docker", "run", "--rm"]
    assert "--gpus" in cmd and cmd[cmd.index("--gpus") + 1] == "all"
    assert "--network" in cmd and cmd[cmd.index("--network") + 1] == "host"
    assert "nvcr.io/nvidia/deepstream:7.1-samples-multiarch" in cmd
    inner = cmd[-1]
    assert "nvv4l2decoder" in inner and "fpsdisplaysink" in inner
    # The real URL must be in the argv (it is passed to the subprocess) ...
    assert _URL in inner
    # ... but the manager logs only the redacted form (asserted elsewhere).
    dpm = DeepStreamProcessManager(
        rtsp_url=_URL,
        image="img",
        network="host",
        rtsp_transport="tcp",
        probe_timeout_seconds=5,
        stop_timeout_seconds=1,
    )
    assert "--gpus" in dpm.command()


# --- worker lifecycle -----------------------------------------------------


async def test_lifecycle_starting_attached_processing_stopped(fast_settings) -> None:  # type: ignore[no-untyped-def]
    api = FakeApi()
    pm = FakePM([_FPS.format(n=10), _FPS.format(n=40), _FPS.format(n=70)])
    w = _worker(api, fast_settings, pm)
    await asyncio.wait_for(w.run(), timeout=3)

    seq = api.status_sequence()
    assert seq[0] == "starting"
    assert "attached" in seq
    assert "processing" in seq
    assert seq[-1] == "stopped"
    assert w.state is WorkerState.STOPPED
    assert pm.started and pm.stopped
    # status rows report the DeepStream runtime/package, not the session's mock
    assert {s.runtime for s in api.statuses} == {"deepstream"}
    assert {s.package_id for s in api.statuses} == {"deepstream.decode_probe.v0"}
    # fps_processed populated on attached/processing/stopped
    proc = next(s for s in api.statuses if s.status == "processing")
    assert proc.fps_processed == pytest.approx(30.2)
    assert proc.payload["runtime"] == "deepstream"
    # M8.2B emits NO observations
    assert api.observations == []


async def test_failed_when_process_exits_nonzero(fast_settings) -> None:  # type: ignore[no-untyped-def]
    api = FakeApi()
    pm = FakePM(["some line", "ERROR: could not link rtph264depay"], exit_code=1)
    w = _worker(api, fast_settings, pm)
    await asyncio.wait_for(w.run(), timeout=3)

    seq = api.status_sequence()
    assert seq[0] == "starting"
    assert "failed" in seq
    assert "stopped" not in seq
    assert w.state is WorkerState.FAILED
    failed = next(s for s in api.statuses if s.status == "failed")
    assert failed.error_code == "deepstream_exit"
    assert api.observations == []


async def test_graceful_stop_posts_stopped(fast_settings) -> None:  # type: ignore[no-untyped-def]
    api = FakeApi()
    pm = FakePM([_FPS.format(n=15)])
    pm.block_until_stop = True
    w = _worker(api, fast_settings, pm)
    task = asyncio.create_task(w.run())

    async def _attached() -> bool:
        return "attached" in api.status_sequence()

    for _ in range(200):
        if await _attached():
            break
        await asyncio.sleep(0.01)
    w.request_stop()
    await asyncio.wait_for(task, timeout=3)

    seq = api.status_sequence()
    assert "attached" in seq
    assert seq[-1] == "stopped"
    assert w.state is WorkerState.STOPPED
    assert pm.stopped
    assert api.observations == []


async def test_missing_rtsp_url_fails_fast(fast_settings) -> None:  # type: ignore[no-untyped-def]
    api = FakeApi()
    pm = FakePM([])
    w = DeepStreamWorker(
        session=_session(internal_rtsp_url=""),
        settings=fast_settings,
        api=api,
        package=resolve("deepstream.decode_probe.v0"),
        pm_factory=lambda: pm,
    )
    await asyncio.wait_for(w.run(), timeout=3)
    assert api.status_sequence() == ["failed"]
    assert api.statuses[0].error_code == "no_rtsp_url"
    assert not pm.started


# --- process manager: hung process is killed after timeout ----------------


class _HungProc:
    def __init__(self) -> None:
        self.returncode: int | None = None
        self.killed = False
        self._dead = asyncio.Event()

    def terminate(self) -> None:
        pass  # ignores SIGTERM

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9
        self._dead.set()

    async def wait(self) -> int:
        await self._dead.wait()
        assert self.returncode is not None
        return self.returncode


async def test_process_manager_kills_hung_process(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    async def _fake_exec(*_a: object, **_k: object):  # type: ignore[no-untyped-def]
        class _Stopper:
            async def wait(self) -> int:
                return 0

        return _Stopper()

    monkeypatch.setattr(pm_mod.asyncio, "create_subprocess_exec", _fake_exec)
    dpm = DeepStreamProcessManager(
        rtsp_url=_URL,
        image="img",
        network="host",
        rtsp_transport="tcp",
        probe_timeout_seconds=5,
        stop_timeout_seconds=0.05,
    )
    hung = _HungProc()
    dpm._proc = hung  # type: ignore[assignment]
    await asyncio.wait_for(dpm.stop(), timeout=3)
    assert hung.killed is True
    assert dpm.returncode == -9
