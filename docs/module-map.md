# Module map

What each file owns.

## Top level
- `pyproject.toml` — hatchling build, flat `app` package, console script
  `upstream-processor = app.main:main`; ruff (line-length 100, py311,
  E/F/I/N/B/ASYNC/C4/UP/RUF/SIM), mypy strict + pydantic plugin, pytest
  asyncio auto.
- `Makefile` — install/run/test/lint/format/typecheck/check. `typecheck`
  runs `mypy app` (app code only — same convention as upstream-api-server's
  `mypy src`).
- `.env.example` — all `UPSTREAM_*` vars, placeholder secret.

## `app/`
- `config.py` — `Settings(BaseSettings)`, `env_prefix="UPSTREAM_"`,
  `SecretStr` secret, `@lru_cache get_settings`.
- `errors.py` — `ProcessorError` base; `ApiError` + subclasses
  (`ApiAuthError`/`ApiSessionGoneError`/`ApiValidationError`/`ApiContractError`/
  `ApiServerError`/`ApiConnectionError`/`ApiRateLimitError`),
  `PackageResolutionError`, `ConfigError`; `TRANSIENT_API_ERRORS` tuple.
- `logging_setup.py` — `setup_logging` + `SecretRedactionFilter` (scrubs the
  literal secret, the auth header, `secret=/token=/password=`, JWTs).
- `backoff.py` — `Backoff` (bounded jittered) + `wait_or_event`
  (shutdown-interruptible sleep). Shared by poller and workers.
- `clock.py` — `now_utc()` / `ensure_aware()`; the only tz source.
- `main.py` — composition root, signal handling, exit-code contract.

### `app/api_client/`
- `base.py` — `ProcessorApi` Protocol (decouples supervisor from the concrete
  client; lets tests inject a fake).
- `upstream_api.py` — `UpstreamApiClient`: async, isolated `_auth_header`
  (only `X-Upstream-Webhook-Secret`), `exclude_none` request bodies, maps
  every failure to a typed error, never logs body/secret.

### `app/observations/`
- `schemas.py` — pydantic v2 models; the single mirror of the M8.0 wire
  contract (eligible/status/observation request+response + the
  `{"sessions":[...]}` envelope).
- `writer.py` — `ObservationWriter.emit`: the one chokepoint for observations;
  enforces the three mock types; stamps tz-aware `observed_at`.
- `lifecycle_policy.py` (P5-Gap) — `LifecyclePolicy`: conservative,
  env-configurable controls for disappearance-based inferred `zone_exited` /
  dwell closure. Built per worker from `deepstream_lifecycle_*` settings; fed to
  `ZoneMembershipState.detect_disappearances`.
- `observation_adapter.py` — the five per-stream adapters (line / object-line /
  ROI / zone / dwell) + `deepstream_source_block`. P5-Gap added
  `ZoneMembershipState.detect_disappearances` (inferred exits) + inferred
  fields on `ZoneTransitionEvent`; P5-G added `requested_package_id` to the
  source block.

### `app/packages/`
- `package_registry.json` — the catalog of packages (`mock.detect.v0`, the five
  TrafficCamNet DeepStream probes, P5-B's `deepstream.yolox_probe.v0`, P5-D's
  `deepstream.yolox_analytics_probe.v0` (the generic/balanced default), P5-E's
  experimental `deepstream.yolox_analytics_nvdcf_probe.v0`, and P5-K's curated
  expansion — 5 YOLOX-S+IOU people variants
  (`deepstream.yolox_people_{fast,high_recall,line_crossing,occupancy,dwell}_probe.v0`),
  a TrafficCamNet vehicle package, and 8 per-new-model people packages
  (yolox_nano/tiny/m, yolo11n/s/m, yolov8n, peoplenet)). Each references
  model/tracker/analytics registry ids plus compute/pricing placeholders and
  P5-D package-readiness metadata (category/triggers/complexity/production_ready
  honesty gate). `outputs`/`supported_triggers`/`supported_shapes` are
  catalog/UX declarations, NOT a runtime emission filter — the worker `_mode`
  (from the package `supports` block) picks the pipeline.
- `model_registry.json` / `tracker_registry.json` / `analytics_registry.json`
  — curated lower-level components (Processing P5..P5-K): 10 detection models
  (TrafficCamNet + YOLOX-S/Nano/Tiny/M + YOLO11-N/S/M + YOLOv8-N + PeopleNet),
  each carrying a `license_class` (permissive/commercial_agreement/restrictive)
  + `first_party` provenance axis and an optional `deepstream.onnx_file`; only
  `yolox_s_v1`/`trafficcamnet_v1` are live-verified `probe`, the eight P5-K
  models are `experimental` (in-repo prep, no GPU verification yet). P5-L added a
  `metrics` block to all 10 models — PUBLISHED reference benchmarks (mAP / params /
  input or, for the TAO models, a labelled vendor KPI), surfaced for an operator
  model-comparison UI; `upstream_measured` is always false (NOT measured by
  Upstream). Plus the
  live-verified IOU tracker profile + P5-E's authored-but-unverified
  `experimental` NvDCF profile (ReID-free perf config) + still-planned NvSORT,
  and the nvdsanalytics probe profile.
- `registries.py` — strict typed loaders/validators for the three component
  registries (`RegistryValidationError` on duplicates / missing fields / bad
  enums) + runnable-entry lookup helpers used by the DeepStream config hooks
  (`find_runnable_deepstream_tracker_by_algorithm` and P5-E's by-profile-id
  variant that backs package-driven tracker selection in the worker). P5-L added
  the frozen `ModelMetricsSpec` dataclass + an optional `metrics` field on
  `ModelSpec` (parsed by `_parse_model_metrics`; `metric_source` +
  `upstream_measured` required, the rest nullable) — published-reference model
  benchmarks, never Upstream-measured.
- `model_descriptor.py` — BYOM Phase 0. Declarative `ModelDescriptor`
  (`DescriptorInput`/`Output`/`Artifact`/`Provenance`) + `descriptor_from_spec(spec)`,
  a normalized view over a registry `ModelSpec` (input geometry + preprocessing,
  output/NMS decoding, class map, precision, artifacts, license provenance). No
  JSON/DB — generated from the registry; `DescriptorValidationError` on bad
  shape. The `decoder` field selects a `DecoderSpec` (see
  `runtimes/deepstream_runtime/decoders.py`). Consumed by
  `build_model_runtime_plan`. Dev notes: `docs/byom-descriptors.md`.
- `compatibility.py` — `validate_package_compatibility`: package ↔ registry
  cross-checks (unknown ids, runtime support, class subsets, tracker-requiring
  outputs, status gating; planned/deprecated components never resolvable).
- `resolver.py` — `resolve(package_id)`; rejects unknown packages, unsupported
  runtimes, and incompatible registry combinations clearly
  (`PackageResolutionError`).

### `app/runtimes/`
- `base.py` — `Runtime` ABC + `WorkerState`; documents the lifecycle and the
  reserved-unused `lagging`.
- `mock_runtime/worker.py` — `MockWorker`: the state machine + status pulse +
  3 observation pulses; never touches `internal_rtsp_url`.
- `mock_runtime/fake_observations.py` — pure deterministic payload builders.
- `metadata_source.py` — metadata-source seam shared by the DeepStream modes.

#### `app/runtimes/deepstream_runtime/`
The DeepStream probe + analytics runtime: docker-backed workers for the
decode / inference / metadata / tracker / analytics modes.
- `worker.py` — `DeepStreamWorker`: per-mode lifecycle, the observation
  emit paths (line crossing / ROI occupancy / zone / dwell), status payload.
- `process_manager.py` — `ProcessManager` protocol + the docker-backed
  managers (bounded probes run under a `timeout` wrapper).
- `inference.py` / `decoders.py` / `*_parser/` — `build_model_runtime_plan`
  (descriptor → decoder → PGIE config bytes), custom bbox parsers.
- `pyds_metadata_app.py` — the in-container pyds app (bind-mounted; imports
  gi/pyds, only resolvable inside the container; excluded from mypy).

#### `app/runtimes/crowdcount_runtime/`
The CrowdCount counting runtime: one counting container per session. The
container owns counting + the counts-of-record posts to crowdcount-server;
Upstream owns the lifecycle + processor status. Emits **no** Upstream
observations (the `ObservationWriter` is accepted but unused).
- `worker.py` — `CrowdCountWorker`: guard the RTSP URL, post `starting`,
  launch, attach on the `CROWDCOUNT_READY` marker (bounded by
  `crowdcount_attach_timeout_seconds`), heartbeat `processing`, map the exit
  to `stopped`/`failed`. Withholds the service secret when a camera has no
  `external_id` (probe mode) — a secret with an empty device id is the one
  combination the container rejects.
- `process_manager.py` — `CrowdCountProcessManager`: `docker run` with the
  session inputs + credentials in a per-run 0600 `--env-file` (keeps them
  out of `ps`/argv; the resolved env IS still visible to `docker inspect`).
  Owns the `CC_*` env contract — keep in lock-step with the container's
  `counter_worker.py`.

### `app/supervisor/`
- `session_poller.py` — `SessionPoller.fetch_eligible`.
- `worker_manager.py` — reconcile loop, spawn/stop/reap, quarantine,
  `active_count`, bounded `_shutdown_all`.
- `processor_status.py` — `build_status_request` (tenant-free, tz-aware).

## `workers/overhead_people_counter/`
The counting container's own source tree — **not** part of the `app` package
and not on the supervisor's import path or in its dependency set. Its deps
(cv2/torch/ultralytics/requests) are installed in the image, built with
`docker build -t crowdcount-counter:dev workers/overhead_people_counter`.
Ported from crowdcount-counter-server `prod`, which it no longer depends on;
the counting behavior is the unmodified CrowdCount prod code.
- `Dockerfile` — CUDA 12.8 runtime + torch cu128 stable + torchreid from
  source + baked OSNet ReID weights. Entrypoint: `run_counter_worker.py`.
- `scripts/run_counter_worker.py` — container entrypoint; `--help` prints the
  `CC_*` contract.
- `src/worker/counter_worker.py` — the single-session entrypoint: parse +
  validate the `CC_*` env (**the contract of record**), build one
  `SingleSessionPeopleCounter`, post device online/offline, print
  `CROWDCOUNT_READY`, own the signal/exit-code lifecycle. Honors a stop
  requested during startup, not just in the frame loop.
- `src/worker/perception_publisher.py` — best-effort background publisher of
  detection boxes + the counting line to the perception-fanout sidecar
  (dashboard overlay). Never blocks or breaks the counting loop.
- `src/processors/people_counter_processor.py` — the counter itself: YOLO v6
  + BoT-SORT + OSNet ReID + the doorway crossing state machine.
- `src/processors/base_processor.py` — stream connect/retry + the frame loop;
  `redact_stream_url` (container-side mirror of `app/redact.py`).
- `src/api/crowdcount_api_client.py` — signed client for crowdcount-server:
  counts-of-record, device status, `direction_flipped`. Service auth
  (`X-CC-Service-Signature`) or per-device auth. One Idempotency-Key per
  logical operation, reused across that operation's retries.
- `src/config/server_config.py` — video/tracker/frame-buffer/inference tuning
  constants (the orchestrator-era settings are gone; Upstream owns the
  stream, the GPU and the lifecycle).
- `models/` — `crowdcount-v6.pt` weights + `botsort.yaml`.

## `tests/`
- `conftest.py` — `UPSTREAM_*` env defaults, settings-cache clear,
  `settings`/`fast_settings` fixtures, `eligible_session_dict` builder.
- `_fakes.py` — `FakeApi` (recording + injectable faults), satisfies
  `ProcessorApi`.
- `test_config / test_api_client / test_session_poller / test_worker_manager /
  test_mock_runtime / test_observation_writer / test_package_resolver /
  test_shutdown / test_logging_redaction` — the M8.1 core.
- `test_crowdcount_runtime.py` — package resolution + dispatch, env-file /
  argv hygiene, credential forwarding, and the worker lifecycle against a
  fake process manager (no Docker, no GPU, no network).
- `test_crowdcount_worker_api_client.py` — the counting container's API
  client (idempotency keying, the status auth verdict). Imports the
  container tree against a faked `requests`: its real deps live in the
  image, and adding them here would put them on the live supervisor's
  `uv run` sync path.
- The DeepStream suites (`test_deepstream_*`, `test_config_parity`, the
  observation/geometry/perception suites) cover the analytics runtime.
