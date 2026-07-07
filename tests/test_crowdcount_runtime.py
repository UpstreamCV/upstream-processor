"""CrowdCount runtime: package resolution, dispatch, process-manager argv
hygiene, and the worker lifecycle against a fake process manager (no Docker,
no GPU, no network)."""

from __future__ import annotations

import asyncio
import os
from collections import deque

import pytest
from app.observations.schemas import EligibleSession
from app.observations.writer import ObservationWriter
from app.packages.resolver import resolve
from app.runtimes.base import WorkerState
from app.runtimes.crowdcount_runtime.process_manager import (
    CrowdCountProcessManager,
)
from app.runtimes.crowdcount_runtime.worker import CrowdCountWorker
from app.supervisor.worker_manager import build_worker

from tests._fakes import FakeApi
from tests.conftest import eligible_session_dict

_RTSP = "rtsp://TRIPWIRE.invalid:8554/sess_0123456789abcdef"


def _session(**ov: object) -> EligibleSession:
    ov.setdefault("package_id", "overhead_people_counter.v1")
    return EligibleSession.model_validate(eligible_session_dict(**ov))


def _worker(api: FakeApi, settings, session: EligibleSession, pm) -> CrowdCountWorker:  # type: ignore[no-untyped-def]
    return CrowdCountWorker(
        session=session,
        settings=settings,
        api=api,
        package=resolve("overhead_people_counter.v1"),
        writer=ObservationWriter(api, settings.processor_id),
        pm_factory=lambda: pm,
    )


class FakePM:
    """Scripted ProcessManager: yields lines, then None with a return code."""

    def __init__(self, lines: list[str], returncode: int = 0) -> None:
        self._lines = deque(lines)
        self._rc = returncode
        self._returncode: int | None = None
        self.started = False
        self.stopped = False

    async def start(self) -> None:
        self.started = True

    async def readline(self) -> str | None:
        if self._lines:
            return self._lines.popleft()
        self._returncode = self._rc
        return None

    async def stop(self) -> None:
        self.stopped = True
        self._returncode = self._rc

    @property
    def returncode(self) -> int | None:
        return self._returncode


class HangingPM(FakePM):
    """Emits its lines, then hangs on readline until stop() is called
    (a healthy long-running counter)."""

    def __init__(self, lines: list[str], returncode: int = 143) -> None:
        super().__init__(lines, returncode)
        self._stop_evt = asyncio.Event()

    async def readline(self) -> str | None:
        if self._lines:
            return self._lines.popleft()
        await self._stop_evt.wait()
        self._returncode = self._rc
        return None

    async def stop(self) -> None:
        self.stopped = True
        self._returncode = self._rc
        self._stop_evt.set()


# --- package resolution + dispatch -----------------------------------------


def test_resolves_overhead_people_counter_package() -> None:
    spec = resolve("overhead_people_counter.v1")
    assert spec.runtime == "crowdcount"
    assert spec.container_image == "crowdcount-counter:dev"
    # Model-less: inference lives inside the CrowdCount image, not the
    # DeepStream model/tracker/analytics registries.
    assert spec.model_id is None
    assert spec.tracker_profile_id is None
    assert spec.analytics_profile_id is None
    # Counts-of-record are NOT an Upstream output; only the alongside
    # observation is declared.
    assert spec.outputs == ("processor.line_crossing",)
    assert spec.supports.get("people_counter") is True


def test_build_worker_dispatches_crowdcount_runtime(fast_settings) -> None:  # type: ignore[no-untyped-def]
    api = FakeApi()
    w = build_worker(
        session=_session(),
        spec=resolve("overhead_people_counter.v1"),
        settings=fast_settings,
        api=api,
        writer=ObservationWriter(api, fast_settings.processor_id),
    )
    assert isinstance(w, CrowdCountWorker)  # constructed, not run -> no docker
    assert w.state is WorkerState.INIT


# --- process manager: argv + env-file hygiene -------------------------------


def _pm(**ov: object) -> CrowdCountProcessManager:
    kwargs: dict[str, object] = {
        "rtsp_url": _RTSP,
        "image": "crowdcount-counter:dev",
        "network": "host",
        "stop_timeout_seconds": 8.0,
        "api_base_url": "https://api.crowdcount.tech",
        "device_id": "C4804",
        "width": 1280,
        "height": 720,
        "fps": 30,
        "gpus": "device=0",
    }
    kwargs.update(ov)
    return CrowdCountProcessManager(**kwargs)  # type: ignore[arg-type]


def test_pm_command_argv_hygiene() -> None:
    pm = _pm()
    try:
        argv = pm.command()
        joined = " ".join(argv)
        assert argv[:3] == ["docker", "run", "--rm"]
        assert "--gpus" in argv and argv[argv.index("--gpus") + 1] == "device=0"
        assert "--network" in argv and argv[argv.index("--network") + 1] == "host"
        assert "--env-file" in argv
        assert "crowdcount-counter:dev" in argv
        # Long-running worker, not a bounded probe.
        assert "timeout" not in joined
        # The RTSP URL travels ONLY in the env-file, never argv.
        assert _RTSP not in joined
        name = argv[argv.index("--name") + 1]
        assert name.startswith("upstream-cc-")
    finally:
        pm._cleanup_env_file()


def test_pm_env_file_contents_and_cleanup() -> None:
    pm = _pm()
    argv = pm.command()
    path = argv[argv.index("--env-file") + 1]
    assert os.path.exists(path)
    # mkstemp => owner-only perms; the file carries the internal RTSP URL.
    assert (os.stat(path).st_mode & 0o777) == 0o600
    with open(path, encoding="utf-8") as f:
        content = f.read()
    assert f"CC_RTSP_URL={_RTSP}" in content
    assert "CC_DEVICE_ID=C4804" in content
    assert "CC_API_BASE_URL=https://api.crowdcount.tech" in content
    assert "CC_FRAME_WIDTH=1280" in content
    assert "CC_FRAME_HEIGHT=720" in content
    assert "CC_FRAME_FPS=30" in content
    # Idempotent: a second command() reuses the same file.
    assert pm.command()[argv.index("--env-file") + 1] == path
    pm._cleanup_env_file()
    assert not os.path.exists(path)


def test_pm_stop_without_start_cleans_env_file() -> None:
    pm = _pm()
    path = pm.command()[pm.command().index("--env-file") + 1]
    assert os.path.exists(path)
    asyncio.run(pm.stop())  # no subprocess started — must still clean up
    assert not os.path.exists(path)


# --- worker lifecycle (fake PM, no Docker) ----------------------------------


def _statuses(api: FakeApi) -> list[str]:
    return [s.status for s in api.statuses]


async def test_worker_walks_lifecycle_to_stopped(fast_settings) -> None:  # type: ignore[no-untyped-def]
    api = FakeApi()
    pm = FakePM(["booting counting pipeline", "CROWDCOUNT_READY", "counting"], returncode=0)
    w = _worker(api, fast_settings, _session(), pm)
    await asyncio.wait_for(w.run(), timeout=5)

    assert w.state is WorkerState.STOPPED
    assert pm.started and pm.stopped
    seq = _statuses(api)
    assert seq[0] == "starting"
    assert "attached" in seq
    assert seq[-1] == "stopped"
    # Status rows reflect what actually ran.
    assert api.statuses[0].runtime == "crowdcount"
    assert api.statuses[0].package_id == "overhead_people_counter.v1"


async def test_worker_attaches_on_existing_prod_ready_line(fast_settings) -> None:  # type: ignore[no-untyped-def]
    # The unmodified CrowdCount prod processor prints
    # "[<stream>] Worker ready (BoT-SORT + crossing logic on CPU)".
    api = FakeApi()
    pm = FakePM(["[C4804] Worker ready (BoT-SORT + crossing logic on CPU)"], returncode=0)
    w = _worker(api, fast_settings, _session(), pm)
    await asyncio.wait_for(w.run(), timeout=5)
    assert "attached" in _statuses(api)
    assert w.state is WorkerState.STOPPED


async def test_worker_failed_on_bad_exit_without_ready(fast_settings) -> None:  # type: ignore[no-untyped-def]
    api = FakeApi()
    pm = FakePM(["ERROR: could not open stream"], returncode=1)
    w = _worker(api, fast_settings, _session(), pm)
    await asyncio.wait_for(w.run(), timeout=5)

    assert w.state is WorkerState.FAILED
    seq = _statuses(api)
    assert "attached" not in seq
    assert seq[-1] == "failed"
    last = api.statuses[-1]
    assert last.error_code == "crowdcount_exit"
    assert "rc=1" in (last.error_message or "")
    assert "could not open stream" in (last.error_message or "")


async def test_worker_request_stop_is_graceful(fast_settings) -> None:  # type: ignore[no-untyped-def]
    api = FakeApi()
    pm = HangingPM(["CROWDCOUNT_READY"], returncode=143)  # SIGTERM exit
    w = _worker(api, fast_settings, _session(), pm)
    task = asyncio.create_task(w.run())

    deadline = asyncio.get_event_loop().time() + 2.0
    while asyncio.get_event_loop().time() < deadline:
        if w.state is WorkerState.PROCESSING:
            break
        await asyncio.sleep(0.01)
    assert w.state is WorkerState.PROCESSING
    assert "processing" in _statuses(api)

    w.request_stop()
    await asyncio.wait_for(task, timeout=5)
    assert w.state is WorkerState.STOPPED
    assert pm.stopped
    assert _statuses(api)[-1] == "stopped"


async def test_worker_missing_rtsp_url_fails_fast(fast_settings) -> None:  # type: ignore[no-untyped-def]
    api = FakeApi()
    pm = FakePM([])
    w = _worker(api, fast_settings, _session(internal_rtsp_url=""), pm)
    await asyncio.wait_for(w.run(), timeout=5)

    assert w.state is WorkerState.FAILED
    assert not pm.started  # never launched a container
    assert _statuses(api) == ["failed"]
    assert api.statuses[0].error_code == "no_rtsp_url"


async def test_worker_sigkill_exit_after_stop_is_graceful(fast_settings) -> None:  # type: ignore[no-untyped-def]
    api = FakeApi()
    pm = FakePM(["CROWDCOUNT_READY"], returncode=137)  # SIGKILL after grace
    w = _worker(api, fast_settings, _session(), pm)
    await asyncio.wait_for(w.run(), timeout=5)
    assert w.state is WorkerState.STOPPED
    assert _statuses(api)[-1] == "stopped"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
