"""M8.2C1-B supervisor-managed inference worker — unit tests.

No Docker/GPU: a scripted ``ProcessManager`` double feeds the worker the
exact log lines the M8.2C1-A live run produced. Asserts the inference
status lifecycle, redaction, no-observations, temp-dir cleanup, and that
the M8.2B decode path is untouched (covered by ``test_deepstream_runtime``).
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterable

import pytest
from app.observations.schemas import EligibleSession
from app.packages.resolver import resolve
from app.runtimes.base import WorkerState
from app.runtimes.deepstream_runtime import inference as inf_mod
from app.runtimes.deepstream_runtime.inference import (
    DeepStreamInferenceProcessManager,
    build_app_config,
    build_inference_docker_command,
    build_pgie_config,
    resolve_model_dir,
)
from app.runtimes.deepstream_runtime.worker import DeepStreamWorker

from tests._fakes import FakeApi
from tests.conftest import eligible_session_dict

_URL = "rtsp://127.0.0.1:18554/sess_abcdef0123456789"
_KEY = "abcdef0123456789"

# Log lines as emitted by the proven M8.2C1-A DeepStream 8.0 run.
_ATTACH = (
    "0:00:00.42 WARN nvinfer NvDsInferContextImpl::deserializeEngineAndBackend(): "
    "deserialize engine from file :/tmp/x/trafficcamnet_b1_gpu0_int8.engine failed"
)
_ENGINE_OK = (
    "0:00:30.70 INFO nvinfer NvDsInferContextImpl::buildModel(): serialize cuda "
    "engine to file: /opt/nvidia/deepstream/deepstream-8.0/samples/models/"
    "Primary_Detector/resnet18_trafficcamnet_pruned.onnx_b1_gpu0_int8.engine successfully"
)
_PERF = "**PERF:  30.00 (29.98)\t"
_APP_OK = "App run successful"
_UNSUPPORTED_SM = (
    f"ERROR: [TRT]: IBuilder::buildSerializedNetwork for {_URL}: Error Code 1: "
    "Internal Error (Unsupported SM: 0xc00)"
)


def _session(**ov: object) -> EligibleSession:
    return EligibleSession.model_validate(eligible_session_dict(**ov))


class FakeInferencePM:
    """ProcessManager double with KITTI evidence — no Docker/GPU."""

    def __init__(
        self,
        lines: Iterable[str],
        *,
        exit_code: int = 124,
        kitti: tuple[int, int] = (4, 9),
    ) -> None:
        self._lines = list(lines)
        self.exit_code = exit_code
        self._kitti = kitti
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

    def kitti_stats(self) -> tuple[int, int]:
        return self._kitti


def _worker(
    api: FakeApi,
    settings,
    pm: FakeInferencePM,
    **sess_ov: object,  # type: ignore[no-untyped-def]
) -> DeepStreamWorker:
    return DeepStreamWorker(
        session=_session(internal_rtsp_url=_URL, **sess_ov),
        settings=settings,
        api=api,
        package=resolve("deepstream.inference_probe.v0"),
        pm_factory=lambda: pm,
    )


# --- mode selection -------------------------------------------------------


def test_inference_package_selects_inference_mode(fast_settings) -> None:  # type: ignore[no-untyped-def]
    w = _worker(FakeApi(), fast_settings, FakeInferencePM([]))
    assert w._mode == "inference_probe"


def test_decode_package_still_selects_decode_mode(fast_settings) -> None:  # type: ignore[no-untyped-def]
    w = DeepStreamWorker(
        session=_session(internal_rtsp_url=_URL),
        settings=fast_settings,
        api=FakeApi(),
        package=resolve("deepstream.decode_probe.v0"),
        pm_factory=lambda: FakeInferencePM([]),
    )
    assert w._mode == "decode_probe"


# --- pure builders --------------------------------------------------------


def test_inference_docker_command_gpus_network_mount() -> None:
    cmd = build_inference_docker_command(
        work_dir="/tmp/wd",
        image="nvcr.io/nvidia/deepstream:8.0-samples-multiarch",
        network="host",
        probe_timeout_seconds=60,
        container_name="upstream-dsi-test",
    )
    assert cmd[:3] == ["docker", "run", "--rm"]
    assert "--gpus" in cmd and cmd[cmd.index("--gpus") + 1] == "all"
    assert "--network" in cmd and cmd[cmd.index("--network") + 1] == "host"
    assert "-v" in cmd and cmd[cmd.index("-v") + 1] == "/tmp/wd:/tmp/wd"
    assert "nvcr.io/nvidia/deepstream:8.0-samples-multiarch" in cmd
    inner = cmd[-1]
    assert "deepstream-app -c /tmp/wd/ds_app.txt" in inner
    assert "timeout -s INT 60" in inner


def test_pgie_and_app_config_contents() -> None:
    pgie = build_pgie_config(model_dir="/m/Primary_Detector", engine_path="/tmp/e.engine")
    assert "onnx-file=/m/Primary_Detector/resnet18_trafficcamnet_pruned.onnx" in pgie
    assert "int8-calib-file=/m/Primary_Detector/cal_trt.bin" in pgie
    assert "network-mode=1" in pgie  # INT8
    app = build_app_config(
        rtsp_url=_URL,
        pgie_path="/tmp/pgie.txt",
        kitti_dir="/tmp/kitti",
        width=1280,
        height=720,
        rtsp_transport="tcp",
    )
    assert f"uri={_URL}" in app  # raw URL only in this temp file
    assert "config-file=/tmp/pgie.txt" in app
    assert "gie-kitti-output-dir=/tmp/kitti" in app
    assert "select-rtp-protocol=4" in app  # tcp
    assert "type=1" in app and "[sink0]" in app  # fakesink, no display
    assert "[osd]\nenable=0" in app and "[tracker]\nenable=0" in app
    # UDP transport flips the rtp protocol selector
    assert "select-rtp-protocol=0" in build_app_config(
        rtsp_url=_URL,
        pgie_path="/p",
        kitti_dir="/k",
        width=1,
        height=2,
        rtsp_transport="udp",
    )


def test_resolve_model_dir_rejects_unknown() -> None:
    assert resolve_model_dir("trafficcamnet").endswith("Primary_Detector")
    assert resolve_model_dir("sample_trafficcamnet").endswith("Primary_Detector")
    with pytest.raises(ValueError, match="unknown deepstream inference model"):
        resolve_model_dir("yolo-custom")


# --- inference lifecycle --------------------------------------------------


async def test_inference_lifecycle_starting_attached_processing_stopped(
    fast_settings,  # type: ignore[no-untyped-def]
) -> None:
    api = FakeApi()
    pm = FakeInferencePM([_ATTACH, _ENGINE_OK, _PERF, _PERF, _APP_OK], exit_code=124, kitti=(7, 15))
    w = _worker(api, fast_settings, pm)
    await asyncio.wait_for(w.run(), timeout=3)

    seq = api.status_sequence()
    assert seq[0] == "starting"
    assert "attached" in seq
    assert "processing" in seq
    assert seq[-1] == "stopped"
    assert w.state is WorkerState.STOPPED
    assert pm.started and pm.stopped

    # status rows report the DeepStream inference runtime/package
    assert {s.runtime for s in api.statuses} == {"deepstream"}
    assert {s.package_id for s in api.statuses} == {"deepstream.inference_probe.v0"}

    proc = next(s for s in api.statuses if s.status == "processing")
    assert proc.payload["mode"] == "inference_probe"
    assert proc.payload["inference_enabled"] is True
    assert proc.payload["inference_engine_built"] is True
    assert proc.payload["model"] == "trafficcamnet"
    assert proc.payload["image"] == "nvcr.io/nvidia/deepstream:8.0-samples-multiarch"

    stopped = next(s for s in api.statuses if s.status == "stopped")
    assert stopped.fps_processed == pytest.approx(29.98)
    assert stopped.payload["detection_frames_count"] == 7
    assert stopped.payload["detection_count"] == 15

    # M8.2C1-B emits NO observations
    assert api.observations == []


async def test_inference_failed_on_unsupported_sm_is_redacted(
    fast_settings,  # type: ignore[no-untyped-def]
) -> None:
    api = FakeApi()
    pm = FakeInferencePM([_ATTACH, _UNSUPPORTED_SM], exit_code=139)
    w = _worker(api, fast_settings, pm)
    await asyncio.wait_for(w.run(), timeout=3)

    seq = api.status_sequence()
    assert seq[0] == "starting"
    assert "failed" in seq
    assert "stopped" not in seq
    assert w.state is WorkerState.FAILED
    failed = next(s for s in api.statuses if s.status == "failed")
    assert failed.error_code == "deepstream_unsupported_sm"
    # the raw stream key must never appear; redacted form must
    assert _KEY not in (failed.error_message or "")
    assert "sess_<redacted>" in (failed.error_message or "")
    assert failed.payload["inference_enabled"] is True
    assert api.observations == []


async def test_inference_graceful_stop_posts_stopped(fast_settings) -> None:  # type: ignore[no-untyped-def]
    api = FakeApi()
    pm = FakeInferencePM([_ATTACH, _ENGINE_OK, _PERF])
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
    assert pm.stopped
    assert api.observations == []


async def test_inference_missing_rtsp_url_fails_fast(fast_settings) -> None:  # type: ignore[no-untyped-def]
    api = FakeApi()
    pm = FakeInferencePM([])
    w = DeepStreamWorker(
        session=_session(internal_rtsp_url=""),
        settings=fast_settings,
        api=api,
        package=resolve("deepstream.inference_probe.v0"),
        pm_factory=lambda: pm,
    )
    await asyncio.wait_for(w.run(), timeout=3)
    assert api.status_sequence() == ["failed"]
    assert api.statuses[0].error_code == "no_rtsp_url"
    assert not pm.started
    assert api.observations == []


# --- real process manager: temp dir handling ------------------------------


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


def _make_pm(tmp_path, *, keep: bool):  # type: ignore[no-untyped-def]
    return DeepStreamInferenceProcessManager(
        rtsp_url=_URL,
        image="img",
        network="host",
        rtsp_transport="tcp",
        probe_timeout_seconds=5,
        stop_timeout_seconds=0.01,
        model="trafficcamnet",
        output_base_dir=str(tmp_path),
        keep_temp_files=keep,
        width=1280,
        height=720,
    )


async def test_inference_pm_writes_then_cleans_temp_dir(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    async def _fake_exec(*_a: object, **_k: object):  # type: ignore[no-untyped-def]
        return _FakeProc()

    monkeypatch.setattr(inf_mod.asyncio, "create_subprocess_exec", _fake_exec)
    pm = _make_pm(tmp_path, keep=False)
    await pm.start()
    wd = pm._work_dir
    assert wd is not None and wd.is_dir()
    assert (wd / "ds_app.txt").exists() and (wd / "pgie.txt").exists()
    # raw RTSP URL lives ONLY in the temp ds_app.txt (never logged/committed)
    assert _URL in (wd / "ds_app.txt").read_text(encoding="utf-8")
    cmd = pm.command()
    assert "--gpus" in cmd and "-v" in cmd
    await pm.stop()
    assert not wd.exists()  # cleaned up by default


async def test_inference_pm_keeps_temp_dir_when_configured(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    async def _fake_exec(*_a: object, **_k: object):  # type: ignore[no-untyped-def]
        return _FakeProc()

    monkeypatch.setattr(inf_mod.asyncio, "create_subprocess_exec", _fake_exec)
    pm = _make_pm(tmp_path, keep=True)
    await pm.start()
    wd = pm._work_dir
    assert wd is not None and wd.is_dir()
    await pm.stop()
    assert wd.is_dir()  # preserved for inspection


def test_inference_pm_unknown_model_raises(tmp_path) -> None:  # type: ignore[no-untyped-def]
    pm = DeepStreamInferenceProcessManager(
        rtsp_url=_URL,
        image="img",
        network="host",
        rtsp_transport="tcp",
        probe_timeout_seconds=5,
        stop_timeout_seconds=0.01,
        model="bogus-model",
        output_base_dir=str(tmp_path),
        keep_temp_files=False,
        width=1280,
        height=720,
    )
    with pytest.raises(ValueError, match="unknown deepstream inference model"):
        pm._prepare_work_dir()
