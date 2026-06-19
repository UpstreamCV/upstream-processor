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
  models are `experimental` (in-repo prep, no GPU verification yet). Plus the
  live-verified IOU tracker profile + P5-E's authored-but-unverified
  `experimental` NvDCF profile (ReID-free perf config) + still-planned NvSORT,
  and the nvdsanalytics probe profile.
- `registries.py` — strict typed loaders/validators for the three component
  registries (`RegistryValidationError` on duplicates / missing fields / bad
  enums) + runnable-entry lookup helpers used by the DeepStream config hooks
  (`find_runnable_deepstream_tracker_by_algorithm` and P5-E's by-profile-id
  variant that backs package-driven tracker selection in the worker).
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

### `app/supervisor/`
- `session_poller.py` — `SessionPoller.fetch_eligible`.
- `worker_manager.py` — reconcile loop, spawn/stop/reap, quarantine,
  `active_count`, bounded `_shutdown_all`.
- `processor_status.py` — `build_status_request` (tenant-free, tz-aware).

## `tests/`
- `conftest.py` — `UPSTREAM_*` env defaults, settings-cache clear,
  `settings`/`fast_settings` fixtures, `eligible_session_dict` builder.
- `_fakes.py` — `FakeApi` (recording + injectable faults), satisfies
  `ProcessorApi`.
- `test_config / test_api_client / test_session_poller / test_worker_manager /
  test_mock_runtime / test_observation_writer / test_package_resolver /
  test_shutdown / test_logging_redaction` — 39 tests.
