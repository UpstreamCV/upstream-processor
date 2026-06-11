"""Processing P5-B — YOLOX DeepStream probe tests.

No Docker / GPU / pyds. Covers:

- the ``deepstream.yolox_probe.v0`` package resolving through the registry
  compatibility layer (and the existing packages staying intact);
- ``build_model_runtime_plan``: TrafficCamNet byte-equivalence (the plan is
  a pure refactor for the builtin model) and the YOLOX plan's PGIE /
  labels / mounts / required-files contract;
- the metadata process manager's host-artifact behavior (missing-artifact
  failure, work-dir contents, extra ``-v`` mount, RTSP only in the env);
- the worker selecting the package's registry model
  (``_effective_model``) and the full metadata-probe lifecycle with the
  YOLOX package emitting ZERO observations;
- shell-script hygiene (syntax, redaction helper, no ``set -x``).
"""

from __future__ import annotations

import asyncio
import json
import subprocess
from collections.abc import Iterable
from pathlib import Path

import pytest
from app.observations.schemas import EligibleSession
from app.packages.resolver import resolve
from app.runtimes.base import WorkerState
from app.runtimes.deepstream_runtime.inference import (
    build_model_runtime_plan,
    build_pgie_config,
)
from app.runtimes.deepstream_runtime.metadata_probe import (
    DeepStreamMetadataProcessManager,
    build_metadata_docker_command,
)
from app.runtimes.deepstream_runtime.worker import DeepStreamWorker

from tests._fakes import FakeApi
from tests.conftest import eligible_session_dict

_URL = "rtsp://127.0.0.1:18554/sess_abcdef0123456789"
_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
_TCN_DIR = "/opt/nvidia/deepstream/deepstream/samples/models/Primary_Detector"


def _session(**ov: object) -> EligibleSession:
    return EligibleSession.model_validate(eligible_session_dict(**ov))


# --- package resolution -------------------------------------------------------


def test_yolox_probe_package_resolves() -> None:
    spec = resolve("deepstream.yolox_probe.v0")
    assert spec.runtime == "deepstream"
    assert spec.status == "probe"
    assert spec.model_id == "yolox_s_v1"
    assert spec.tracker_profile_id is None
    assert spec.analytics_profile_id is None
    assert spec.supports.get("metadata_probe") is True
    assert spec.outputs == ()  # status/debug evidence only — no observations
    assert spec.pricing_tier == "free_probe"


def test_existing_packages_unaffected_by_yolox_addition() -> None:
    for package_id in (
        "mock.detect.v0",
        "deepstream.decode_probe.v0",
        "deepstream.inference_probe.v0",
        "deepstream.metadata_probe.v0",
        "deepstream.tracker_probe.v0",
        "deepstream.analytics_probe.v0",
    ):
        assert resolve(package_id).package_id == package_id


# --- model runtime plan -------------------------------------------------------


def test_trafficcamnet_plan_is_byte_identical_to_direct_builder() -> None:
    """The plan path is a pure refactor for the builtin model: same PGIE
    bytes, engine in the work dir, no mounts, no host requirements."""
    for threshold in (None, 0.55):
        plan = build_model_runtime_plan(
            model="trafficcamnet",
            work_dir="/tmp/work",
            model_store_dir="/opt/upstream/models",
            confidence_threshold=threshold,
        )
        assert plan.pgie_config == build_pgie_config(
            model_dir=_TCN_DIR,
            engine_path="/tmp/work/trafficcamnet_b1_gpu0_int8.engine",
            confidence_threshold=threshold,
        )
        assert plan.model_id == "trafficcamnet_v1"
        assert plan.image_builtin is True
        assert plan.labels_body is None
        assert plan.extra_mount_dirs == ()
        assert plan.required_files == ()


def test_yolox_plan_contract() -> None:
    plan = build_model_runtime_plan(
        model="yolox_s_v1",
        work_dir="/tmp/work",
        model_store_dir="/data/models/",  # trailing slash normalized
        confidence_threshold=None,
    )
    assert plan.model_id == "yolox_s_v1"
    assert plan.model_family == "yolox"
    assert plan.model_dir == "/data/models/yolox_s_v1"
    assert plan.image_builtin is False
    assert plan.inference_backend == "tensorrt_fp16"
    assert plan.extra_mount_dirs == ("/data/models/yolox_s_v1",)
    assert plan.required_files == (
        "/data/models/yolox_s_v1/yolox_s.onnx",
        "/data/models/yolox_s_v1/libnvdsparsebbox_yolox.so",
    )
    pgie = plan.pgie_config
    assert "onnx-file=/data/models/yolox_s_v1/yolox_s.onnx" in pgie
    assert "model-engine-file=/data/models/yolox_s_v1/yolox_s_b1_gpu0_fp16.engine" in pgie
    assert "custom-lib-path=/data/models/yolox_s_v1/libnvdsparsebbox_yolox.so" in pgie
    assert "parse-bbox-func-name=NvDsInferParseYolox" in pgie
    assert "labelfile-path=/tmp/work/labels.txt" in pgie
    assert "num-detected-classes=80" in pgie
    assert "network-mode=2" in pgie  # FP16 — no INT8 calibration cache
    assert "net-scale-factor=1.0" in pgie  # YOLOX: no input normalization
    assert "model-color-format=1" in pgie  # BGR, like the YOLOX demo
    assert "maintain-aspect-ratio=1" in pgie
    assert "pre-cluster-threshold=0.25\n" in pgie  # YOLOX demo default
    # Labels are generated from the registry's COCO-ordered classes.
    assert plan.labels_body is not None
    labels = plan.labels_body.strip().split("\n")
    assert len(labels) == 80
    assert labels[0] == "person"
    assert labels[-1] == "toothbrush"


def test_yolox_plan_alias_and_confidence_override() -> None:
    plan = build_model_runtime_plan(
        model="yolox_s",  # registry alias
        work_dir="/w",
        model_store_dir="/m",
        confidence_threshold=0.6,
    )
    assert plan.model_id == "yolox_s_v1"
    assert plan.pgie_config.count("pre-cluster-threshold=0.6\n") == 1


def test_yolox_plan_contains_no_rtsp_or_secrets() -> None:
    plan = build_model_runtime_plan(model="yolox_s_v1", work_dir="/w", model_store_dir="/m")
    blob = plan.pgie_config + (plan.labels_body or "")
    for needle in ("rtsp://", "secret", "Authorization", "Bearer", "srt://"):
        assert needle not in blob


def test_plan_rejects_planned_and_unknown_models() -> None:
    with pytest.raises(ValueError, match="not runnable"):
        build_model_runtime_plan(model="yolov8n_v1", work_dir="/w", model_store_dir="/m")
    with pytest.raises(ValueError, match="unknown deepstream inference model"):
        build_model_runtime_plan(model="nope", work_dir="/w", model_store_dir="/m")


# --- metadata process manager -------------------------------------------------


def _pm(model: str, store: Path, work_base: Path) -> DeepStreamMetadataProcessManager:
    return DeepStreamMetadataProcessManager(
        rtsp_url=_URL,
        image="img:test",
        network="host",
        rtsp_transport="tcp",
        probe_timeout_seconds=5,
        stop_timeout_seconds=1,
        model=model,
        output_base_dir=str(work_base),
        keep_temp_files=True,  # inspect contents; cleaned by tmp_path
        sample_limit=3,
        stream_session_id="s",
        width=1280,
        height=720,
        model_store_dir=str(store),
    )


def test_pm_fails_clearly_when_yolox_artifacts_missing(tmp_path: Path) -> None:
    pm = _pm("yolox_s_v1", tmp_path / "store", tmp_path / "work")
    with pytest.raises(ValueError, match="model artifacts missing"):
        pm._prepare_work_dir()
    with pytest.raises(ValueError, match=r"prepare_yolox_model\.sh"):
        pm._prepare_work_dir()


def test_pm_yolox_workdir_and_mounts(tmp_path: Path) -> None:
    store = tmp_path / "store"
    model_dir = store / "yolox_s_v1"
    model_dir.mkdir(parents=True)
    (model_dir / "yolox_s.onnx").write_bytes(b"onnx")
    (model_dir / "libnvdsparsebbox_yolox.so").write_bytes(b"so")
    pm = _pm("yolox_s_v1", store, tmp_path / "work")
    work = pm._prepare_work_dir()
    pm._work_dir = work
    try:
        pgie = (work / "pgie.txt").read_text()
        assert f"onnx-file={model_dir}/yolox_s.onnx" in pgie
        assert "NvDsInferParseYolox" in pgie
        labels = (work / "labels.txt").read_text().strip().split("\n")
        assert labels[0] == "person" and len(labels) == 80
        assert "rtsp://" not in pgie  # raw URL never lands in configs
        argv = pm.command()
        assert f"{model_dir}:{model_dir}" in argv  # host model dir mounted
        # RTSP URL reaches the container ONLY via the env assignment.
        url_args = [a for a in argv if _URL in a]
        assert url_args == [f"RTSP_URL={_URL}"]
    finally:
        pm._keep_temp_files = False
        pm._cleanup_work_dir()


def test_pm_trafficcamnet_workdir_unchanged(tmp_path: Path) -> None:
    """TrafficCamNet equivalence at the PM level: same pgie bytes as the
    direct builder, no labels file, no extra mounts."""
    pm = _pm("trafficcamnet", tmp_path / "store", tmp_path / "work")
    work = pm._prepare_work_dir()
    pm._work_dir = work
    try:
        assert (work / "pgie.txt").read_text() == build_pgie_config(
            model_dir=_TCN_DIR,
            engine_path=str(work / "trafficcamnet_b1_gpu0_int8.engine"),
            confidence_threshold=None,
        )
        assert not (work / "labels.txt").exists()
        argv = pm.command()
        assert argv.count("-v") == 1  # only the work dir is mounted
    finally:
        pm._keep_temp_files = False
        pm._cleanup_work_dir()


def test_build_metadata_docker_command_extra_mounts() -> None:
    argv = build_metadata_docker_command(
        work_dir="/w",
        image="img",
        network="host",
        probe_timeout_seconds=5,
        container_name="c",
        sample_limit=3,
        stream_session_id="s",
        rtsp_url=_URL,
        extra_mount_dirs=("/models/yolox_s_v1",),
    )
    joined = " ".join(argv)
    assert "-v /w:/w" in joined
    assert "-v /models/yolox_s_v1:/models/yolox_s_v1" in joined


# --- worker behavior ----------------------------------------------------------


class FakeMetadataPM:
    """Scripted ProcessManager double (no Docker/GPU)."""

    def __init__(self, lines: Iterable[str], *, exit_code: int = 124) -> None:
        self._lines = list(lines)
        self.exit_code = exit_code
        self._rc: int | None = None

    async def start(self) -> None:
        pass

    async def readline(self) -> str | None:
        await asyncio.sleep(0)
        if self._lines:
            return self._lines.pop(0)
        self._rc = self.exit_code
        return None

    async def stop(self) -> None:
        if self._rc is None:
            self._rc = self.exit_code

    @property
    def returncode(self) -> int | None:
        return self._rc


def _yolox_frame_line(frame_num: int) -> str:
    return "UPSTREAM_METADATA=" + json.dumps(
        {
            "runtime": "deepstream",
            "metadata_source": "pyds_pad_probe",
            "stream_session_id": None,
            "frame": {"frame_num": frame_num, "width": 1280, "height": 720},
            "objects": [
                {
                    "class_id": 0,
                    "label": "person",
                    "confidence": 0.87,
                    "bbox_pixels": {"left": 100.0, "top": 50.0, "width": 200.0, "height": 400.0},
                    "bbox_normalized": {"x": 0.078, "y": 0.069, "w": 0.156, "h": 0.555},
                    "tracker_id": None,
                }
            ],
        }
    )


def test_worker_effective_model_prefers_package_model_id(fast_settings) -> None:  # type: ignore[no-untyped-def]
    yolox = DeepStreamWorker(
        session=_session(internal_rtsp_url=_URL),
        settings=fast_settings,
        api=FakeApi(),
        package=resolve("deepstream.yolox_probe.v0"),
        pm_factory=lambda: FakeMetadataPM([]),
    )
    assert yolox._mode == "metadata_probe"
    assert yolox._effective_model == "yolox_s_v1"
    decode = DeepStreamWorker(
        session=_session(internal_rtsp_url=_URL),
        settings=fast_settings,
        api=FakeApi(),
        package=resolve("deepstream.decode_probe.v0"),
        pm_factory=lambda: FakeMetadataPM([]),
    )
    # No model_id on the decode package — the global setting still applies.
    assert decode._effective_model == fast_settings.deepstream_inference_model


async def test_yolox_probe_lifecycle_metadata_only(fast_settings) -> None:  # type: ignore[no-untyped-def]
    """Full metadata-probe run with the YOLOX package: starting → attached →
    processing → stopped, person metadata counted, model_id in the status
    payload, and ZERO observations."""
    api = FakeApi()
    pm = FakeMetadataPM([_yolox_frame_line(1), _yolox_frame_line(2)])
    worker = DeepStreamWorker(
        session=_session(internal_rtsp_url=_URL),
        settings=fast_settings,
        api=api,
        package=resolve("deepstream.yolox_probe.v0"),
        pm_factory=lambda: pm,
    )
    await asyncio.wait_for(worker.run(), timeout=5)
    assert worker.state is WorkerState.STOPPED
    statuses = [s.status for s in api.statuses]
    assert statuses[0] == "starting"
    assert "attached" in statuses and "processing" in statuses
    assert statuses[-1] == "stopped"
    final = api.statuses[-1].payload
    assert final["model"] == "yolox_s_v1"
    assert final["mode"] == "metadata_probe"
    assert final["label_counts"] == {"person": 2}
    assert final["objects_seen"] == 2
    assert api.observations == []  # probe milestone: no observation writes
    assert api.perception_snapshots == []
    # Secret/RTSP hygiene: nothing in any posted payload carries the URL.
    blob = json.dumps([s.model_dump(mode="json") for s in api.statuses])
    assert _URL not in blob
    assert "test-secret-value-xyz" not in blob


# --- scripts hygiene -----------------------------------------------------------


@pytest.mark.parametrize("script", ["prepare_yolox_model.sh", "probe_deepstream_yolox_metadata.sh"])
def test_yolox_scripts_syntax_ok(script: str) -> None:
    proc = subprocess.run(["bash", "-n", str(_SCRIPTS / script)], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr


def test_probe_script_redacts_and_never_traces() -> None:
    body = (_SCRIPTS / "probe_deepstream_yolox_metadata.sh").read_text()
    assert "redact()" in body  # RTSP redaction helper defined...
    assert '"$URL" | redact' in body  # ...and applied to the echoed URL
    assert "set -x" not in body  # tracing would leak the URL via argv
    prepare = (_SCRIPTS / "prepare_yolox_model.sh").read_text()
    assert "set -x" not in prepare
    for needle in ("rtsp://", "WEBHOOK_SECRET", "Authorization"):
        assert needle not in prepare  # prepare never touches streams/secrets


def test_probe_script_fails_clearly_without_artifacts(tmp_path: Path) -> None:
    """With an empty MODEL_DIR the probe script must exit 2 and point the
    operator at the prepare script — without needing docker to run."""
    proc = subprocess.run(
        ["bash", str(_SCRIPTS / "probe_deepstream_yolox_metadata.sh"), _URL],
        capture_output=True,
        text=True,
        env={"PATH": "/usr/bin:/bin", "MODEL_DIR": str(tmp_path)},
    )
    assert proc.returncode == 2
    combined = proc.stdout + proc.stderr
    assert "missing" in combined
    # The failure path never echoes the raw stream URL.
    assert _URL not in combined
