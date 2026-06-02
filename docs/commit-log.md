# Commit log

Reverse chronological. Newest at top.

---

## 2026-06-02 - M7D: Apply selected_classes by post-inference per-object filtering

### Summary
The DeepStream worker now **applies** the operator-saved per-camera
`selected_classes` — not by reconfiguring the detector, but by **post-inference
filtering of the per-object output paths**. Inference still runs across all
classes (`nvinfer`/PGIE class-id config and `nvdsanalytics` untouched); the
allow-list drops non-matching objects before they reach the USER-FACING
per-object surfaces. Scope is `selected_classes` ONLY; model switching,
`processing_fps`, and tracker switching remain stamp-only, and M7C
`confidence_threshold` apply is unchanged/unregressed.

- NEW `app/processing/class_filter.py` — pure, unit-tested helper.
  `CLASS_ID_TO_LABEL = {0:car,1:bicycle,2:person,3:roadsign}` (TrafficCamNet
  order, fallback when an object has no label). `canonical_class(label,
  class_id)` prefers `label.strip().lower()` else the id map else `None`.
  `filter_objects_by_class(objects, selected_classes)` returns the input
  UNCHANGED when the list is empty/`None`; otherwise keeps only objects whose
  canonical class is in the normalized allow-list (unresolvable-class objects
  excluded). Non-mutating, order-preserving, deterministic.
- `resolve_session_processing_config` (`resolver.py`) gains
  `selected_classes_apply_supported: bool = False`. When `True` and a non-empty
  list is saved, it is applied (echoed into `applied.selected_classes`,
  `applied_count += 1`) and the honest warning
  `selected_classes_analytics_counts_not_filtered` is added — so a
  selected-classes-applied config is always at least `partially_applied`. When
  unsupported, the list stays `selected_classes_requested_not_applied` with
  `applied.selected_classes == []` (prior behavior). `applied` is now
  `replace(defaults, confidence_threshold=..., selected_classes=...)`.
- `worker.py`: computes `selected_classes_apply_supported = self._mode in
  (_METADATA_MODE, _TRACKER_MODE, _ANALYTICS_MODE)` (the per-object modes),
  passes it to the resolver, and derives the per-worker filter
  `self._selected_classes = tuple(resolved.applied.selected_classes)` (single
  source of truth; no global state).

### What IS filtered (USER-FACING per-object surfaces)
- The perception snapshot — `_build_perception_payload` filters `frame.objects`
  and builds the snapshot from a `frame.model_copy(update={"objects": ...})`
  (the shared frame is never mutated in place).
- The zone / dwell observation paths — `_emit_zone_transitions` filters the
  objects fed to `zone_state.absorb_object(...)`; dwell follows zone. Track
  continuity is preserved (class is stable per track).

### What is NOT filtered (out of scope, disclosed in lineage)
- nvdsanalytics AGGREGATE counts — line-crossing (`_emit_line_crossings`) and
  ROI-occupancy (`_emit_roi_occupancies`) come from `frame.source["analytics"]`
  (per-line / per-ROI totals across ALL classes, computed in-container). Not
  per-class filterable post-inference; left byte-unchanged. The
  `selected_classes_analytics_counts_not_filtered` lineage warning is the honest
  signal.
- RAW status debug counters (`class_counts` / `objects_seen` via
  `counters.absorb(frame)`) — kept raw (debug = what the detector produced).

### No-filter path
When `selected_classes` is `[]` (or no config), `filter_objects_by_class`
returns the objects unchanged — byte/behavior-identical to before M7D.

### Not in scope
No model switching, no `processing_fps` throttle, no tracker switching, no
nvinfer/PGIE class-id config change, no nvdsanalytics class-id config change, no
geometry/reliability change. No api-server / dashboard / DB / migration. M7C
confidence apply unchanged.

---

## 2026-06-02 - M7C: Apply confidence_threshold to the live PGIE config

### Summary
The DeepStream worker now **applies** the operator-saved per-camera
`confidence_threshold` to the live nvinfer PGIE config — the first piece of the
processing config that is genuinely applied rather than only stamped. Scope is
`confidence_threshold` ONLY: model switching, `selected_classes` / class
filtering, `processing_fps`, and dynamic tracker switching remain stamp-only.

- `build_pgie_config` (`inference.py`) gains an optional
  `confidence_threshold: float | None = None`. A float overrides BOTH
  `pre-cluster-threshold` entries (`[class-attrs-all]` and `[class-attrs-0]`);
  `None` leaves the output byte-identical to the prior hard-coded 0.2 / 0.4
  defaults. No other PGIE field (topk / nms / interval / num-classes / paths)
  changes.
- `DeepStreamMetadataProcessManager.__init__` gains
  `confidence_threshold: float | None = None` (stored as
  `self._confidence_threshold`), forwarded to `build_pgie_config` in
  `_prepare_work_dir`. Default keeps existing callers/tests unaffected.
- `resolve_session_processing_config` (`resolver.py`) gains
  `confidence_apply_supported: bool = False`. When `True` and a threshold is
  saved, it is applied (echoed into `applied.confidence_threshold`, no warning,
  `applied_count += 1`); otherwise it stays
  `confidence_threshold_requested_not_applied`. `applied` is now
  `replace(defaults, confidence_threshold=applied_confidence)`. Status:
  `partially_applied` (warnings + something applied), `requested_not_applied`
  (warnings only), `applied` (something applied, no warnings), else
  `applied_defaults`. `partially_applied` / `applied` are now emitted.
- The worker passes `confidence_apply_supported = self._mode in
  (_METADATA_MODE, _TRACKER_MODE, _ANALYTICS_MODE)` to the resolver, and
  forwards the resolved APPLIED value
  (`self._resolved_processing_config.applied.confidence_threshold`) into the
  `DeepStreamMetadataProcessManager` — the single source of truth, so the
  stamped lineage and the live PGIE config can never disagree.

### Applied only where it can take effect
Confidence is applied only in the DeepStream metadata / tracker / analytics
modes (the ones that build a PGIE config). Inference / decode / probe modes and
the `inference.py:586` `inference_probe` call site keep calling
`build_pgie_config` with no confidence arg (defaults to `None`) — unchanged.

### Per-camera safety
The value flows per-worker (per session) through the process-manager instance —
never via global `Settings` or a module global. No cross-camera leakage.

### Not in scope
No model switching, no `selected_classes` application, no `processing_fps`
throttle, no dynamic tracker switching. No DB migration, no dashboard change, no
api-server change. Tracker config, analytics class-id / geometry, `package_id`,
account `processing_enabled`, reliability all untouched.

---

## 2026-06-01 - M7B: Processor stamps applied_processing_config (STAMP ONLY)

### Summary
The DeepStream worker now consumes the eligible-session `processing_config`
envelope at startup and stamps a requested-vs-applied
`applied_processing_config` lineage block onto every status payload. This is
**STAMP ONLY** — no runtime behavior changes. Nothing from the config is
applied: the model, tracker, confidence threshold, processing FPS, and class
filter all stay at the worker's current defaults. The block records what the
operator *requested* alongside what the runtime *actually applies* so the
dashboard can show the gap before a later milestone wires real application.

- the eligible-session envelope gains a nullable `processing_config`
  `{ "revision": int, "config": {...} }` field (mirrors `geometry_config`),
  `null` when the camera has no `camera_processing_configs` row,
- the processor re-validates the envelope locally (`extra="forbid"` drift
  policy, mirroring `session_geometry.py`) against a clone of the api-server's
  `CameraProcessingConfig`,
- the resolver compares each requested field against the static applied
  defaults (`trafficcamnet_v1` / `[]` all classes / `iou` / no threshold / no
  fps override) and records per-field `*_requested_not_applied` warnings,
- every processor status post carries `payload.applied_processing_config`
  with `source` / `revision` / `status` / `reason` / `requested` / `applied`
  / `warnings`.

### Resolution policy (mirrors geometry, STAMP ONLY)
- **No saved config** (`null`) → `source="default"`,
  `status="applied_defaults"`, all-null `requested`, `applied`=defaults.
- **Saved config, valid** → `source="operator_saved"`; `requested` echoes the
  saved values; `applied` stays at defaults (never echoes a requested value);
  `status="requested_not_applied"` when any field diverges, else
  `applied_defaults`.
- **Saved config, invalid** (malformed envelope) →
  `source="fallback_default_on_error"`,
  `status="fallback_default_on_error"`, short safe `reason`. Unlike geometry's
  `geometry_invalid`, this does **NOT** set `error_code` — M7B applies
  nothing, so a bad processing config must not flag a worker error; it lives
  only in the lineage block.
- `status="partially_applied"` is reserved for a later milestone and is NEVER
  emitted in M7B.

### Files touched
- `app/observations/schemas.py` — `EligibleSession.processing_config:
  dict | None` (additive, `extra="ignore"`).
- `app/processing/__init__.py` — new package marker (empty).
- `app/processing/session_processing.py` — `SessionProcessingConfig` /
  `SessionProcessingConfigEnvelope` re-validation mirror of the api-server's
  `CameraProcessingConfig`.
- `app/processing/resolver.py` — `RequestedProcessingConfig` /
  `AppliedProcessingConfig` / `ResolvedProcessingConfig`,
  `resolve_session_processing_config` (pure, never raises), `DEFAULT_APPLIED`,
  `MODEL_KEY_RUNTIME` (lineage-only static map).
- `app/runtimes/deepstream_runtime/worker.py` — resolve in `__init__`
  (defensive, never raises, no `error_code`) + stamp
  `payload.applied_processing_config` in `_build()` alongside
  `applied_geometry`.
- `docs/architecture.md`, `docs/commit-log.md` — this note.

### What this milestone did NOT change
- `inference.py` / `metadata_probe.py` / pgie / tracker / nvinfer interval /
  nvdsanalytics class-id / analytics class normalization — untouched.
- No multi-class ever passed to the runtime filter; no observation/snapshot
  count changes; account-level `processing_enabled`, `package_id` resolution,
  and the dashboard are untouched.
- The only observable changes: a new nullable envelope field + an additive
  status-payload key.

---

## 2026-05-28 - Decouple perception upload from the frame loop + configurable GPU device

### Summary
Two changes driven by a live FPS investigation. An operator running a 30 FPS
camera on an idle-GPU box saw detections cap at a dead-regular ~3 FPS on the
dashboard. Root cause: the latest-perception-snapshot POST was `await`ed
**inline in the per-frame consumer loop**. The POST is a ~330ms round-trip to
the cloud API, and the configured emit interval (0.1s) was *shorter* than that
round-trip — so the loop posted back-to-back and became entirely
network-round-trip-bound at ~3 FPS. Critically this throttled not just the
overlay but the whole loop, so observation sampling (line crossings, ROIs,
zones, dwell) was also pinned to 3 FPS. Neither the GPU (idle) nor the source
(30 FPS) nor inference (`interval=0`, every frame) was the limit.

### 1. Decouple the perception upload (the FPS fix)
The frame loop now *builds* a snapshot (cheap, local) and drops it in a single
coalescing slot; a background sender task drains the slot and does the POST
**off the hot path**. Effects:

- The consumer loop runs at the full source rate again → observations sample
  at 30 FPS instead of 3.
- "Coalescing" = only the freshest snapshot is ever sent; if a POST is in
  flight, intermediate snapshots are dropped, so the overlay always shows the
  latest frame and never a growing backlog.
- The single sequential sender still uploads at ~round-trip rate (~3 FPS), but
  non-blocking — pushing the *overlay* past that needs bounded concurrency or
  lower API latency, deliberately deferred.

Behavior preserved: `ApiSessionGoneError` from the upload is relayed via the
slot's `fatal` field and re-raised by the loop so a vanished session still
ends the worker quietly; every other error is counted
(`perception_snapshot_write_errors`) and swallowed so a perception failure
never disturbs observations. The sender is **flushed** (not cancelled) at
consumer teardown — stop is signalled, it drains the final pending snapshot,
and the flush is awaited (bounded by `_PERCEPTION_FLUSH_TIMEOUT=5s`) *before*
the terminal status post so the final perception counters are accurate.
`_emit_perception_snapshot` is split into `_build_perception_payload` (loop,
no network) and `_post_perception_payload` (sender, the round-trip). New test
`test_slow_perception_post_does_not_block_frame_loop` holds the upload open and
asserts a later frame's line crossing still emits — with the old inline design
it would deadlock.

### 2. Configurable GPU device (`UPSTREAM_DEEPSTREAM_GPUS`)
`docker run --gpus all` was hard-coded in all four DeepStream command builders
(decode, inference, metadata/analytics). Added a `deepstream_gpus` setting
(default `"all"`) threaded through `worker.py` into every builder so a worker
can be pinned to a specific card, e.g. `UPSTREAM_DEEPSTREAM_GPUS="device=1"`
exposes only physical GPU 1 (all DeepStream elements at gpu-id=0 then map onto
it). Note the docker semantics: a bare integer is a *count* of GPUs, not an
index — use `device=N`. Does not affect FPS (GPUs were idle); it isolates the
processor onto a chosen card. Each builder gets a device-pinning test.

---

## 2026-05-27 - P6-D: End-to-end geometry lineage verification

### Summary
P6-C is live-verified ad-hoc (tail the log, grep the temp INI, eyeball the
dashboard). P6-D formalizes that proof as a single repeatable command that
exits with a machine-checkable verdict code, so the same loop can be run on
demand without rebuilding a stream.

The verification reads three witnesses from the api-server (no shell access
to the processor host required) plus an optional fourth from disk:

1. `GET /v1/cameras/{id}/processor-status/latest` — the new caller-authed
   read endpoint that exposes the full `processor_statuses` row, including
   the `payload.applied_geometry` lineage block P6-C stamps.
2. `GET /v1/cameras/{id}/observations/recent?type=<...>` (already existed)
   for each of `processor.line_crossing`, `processor.roi_occupancy`,
   `processor.zone_entered`, `processor.dwell_started`. Each row's
   `payload.source.geometry_source/geometry_revision` and
   `payload.line_id`/`payload.roi_id` are checked against expected slugs.
3. `GET /v1/cameras/{id}/perception/latest` (already existed) — an
   independent witness from a different table; the snapshot embeds the
   applied geometry, so its `snapshot.geometry.lines[].id` and
   `snapshot.geometry.rois[].id` must match expected.
4. Optional: `--analytics-config-path /tmp/.../analytics_config.txt`
   parses the live nvdsanalytics INI and pins that operator slugs are
   present AND the probe-default `roi-entry-roi=64;36;…` is not leaked
   alongside them.

### Verdict matrix
- **PASS** (exit 0) — every witness confirms operator lineage.
- **PARTIAL** (exit 1) — processor status + ROI/zone/dwell + snapshot pass,
  but `line_crossing` rows are absent (most common: operator walked
  parallel to a vertical saved line). Not a P6-C bug; re-walk perpendicular.
- **FAIL** (exit 2) — lineage is wrong somewhere: `geometry_source=default`
  on a row, revision mismatch, unknown slug, `error_code=geometry_invalid`
  on status, or probe-default ROI leaked into the runtime INI.
- **BLOCKED** (exit 3) — processor never reported (`processor_status_not_found`),
  HTTP 4xx/5xx, or `UPSTREAM_API_TOKEN` missing.

The processor-status-blocked case vetoes any PARTIAL/PASS (we can't credibly
judge the rest without that witness). A missing perception snapshot is a
soft block on just that witness — it does not veto PASS by itself.

### Files touched

In `upstream-api-server`:
- `src/upstream_api/schemas/processor.py` — new `ProcessorStatusRecord` (full
  row, including `payload`/`fps_*`/`error_*` that the write-side
  `ProcessorStatusResponse` deliberately omitted) + `LatestProcessorStatusResponse`.
- `src/upstream_api/services/processor.py` — new `fetch_latest_processor_status`
  + `_LATEST_PROCESSOR_STATUS_SQL` with an optional `stream_session_id`
  filter (tri-state predicate, matches every row when null).
- `src/upstream_api/routers/cameras.py` — new `GET /v1/cameras/{id}/processor-status/latest`
  route. Caller-authed, account-scoped, 404
  `processor_status_not_found` mirroring `perception/latest`.
- `tests/test_processor_status_latest.py` — 10 new tests covering happy
  path, `applied_geometry` payload flow-through, `error_code` round-trip,
  query scoping (account, camera, optional stream_session_id), 401, 404
  on missing row, 404 on cross-account, and the SQL-predicate-pin
  regression guard.
- `docs/api-contracts.md` — endpoint documented alongside the existing
  observations/perception reads, with the `applied_geometry` shape and the
  `error_code=geometry_invalid` degrade contract spelled out.

In `upstream-processor`:
- `scripts/verify_geometry_lineage.py` — argparse CLI, self-contained read
  client using `httpx` + Bearer auth, pure verdict logic factored out for
  unit testing, exit codes 0/1/2/3, `--debug` dumps with `rtsp://...`
  scrubbed via `app.redact.scrub_rtsp_in_text`. Token sourced from
  `UPSTREAM_API_TOKEN` env-only; never printed.
- `tests/test_verify_geometry_lineage.py` — 35 tests:
  pure verdict-logic tests for each `check_*` helper covering the full
  verdict matrix (PASS / PARTIAL / FAIL / BLOCKED on each path),
  `aggregate_verdict` rules, `respx`-mocked end-to-end tests proving
  Bearer auth + URL routing + the stream_session_id query param threads
  through, redaction test proving `--debug` masks RTSP paths and never
  prints the bearer token, and exit-code pin.
- `docs/p6-d-geometry-lineage-verification.md` — operator-facing
  step-by-step checklist (save geometry → restart processor → confirm
  `geometry resolved` line for the new session → walk the perpendicular
  pattern → run the verify script → read the verdict).

### Testing/checks
- `uv run ruff check .` / `ruff format --check .` — clean.
- `uv run mypy app` — clean.
- `uv run pytest -q` — all pass.

### Known caveats / follow-ups
- **No CI hook yet.** The script is run-on-demand. A future CI job could
  run a canned synthetic walk and gate on FAIL/BLOCKED (not PARTIAL — line
  crossings depend on physical motion).
- **Dashboard "applied geometry" Monitor-tab chip** — deferred per the
  P6-D guardrails. The new processor-status/latest endpoint makes it cheap
  to add as a follow-up; it would consume the same `payload.applied_geometry`
  block the verify script reads.
- **Strict-direction enforcement, mid-session reload, reliability findings,
  rectangle/box ROI authoring** — all still tracked as P6-C follow-ups; not
  touched by P6-D.

---

## 2026-05-27 - P6-C: Processor consumes saved geometry

### Summary
The DeepStream analytics worker now applies operator-saved geometry from the
eligible-session envelope (P6-A wire-carry) to its nvdsanalytics runtime
config. Before this commit the worker always rendered the hard-coded probe
default (one near-full-frame ROI + one horizontal line) regardless of what
the dashboard saved. After this commit:

- saved counting lines and polygon ROIs flow from the dashboard through the
  `/v1/internal/processor/eligible-sessions` envelope into the nvdsanalytics
  INI on the *next eligible session* (restart-time only; mid-session reload
  is out of scope),
- normalized `[0,1]` operator coordinates are converted to runtime pixels
  via the session's approved frame dimensions (truncate, then clamp to
  `[0, dim-1]`),
- the rendered INI emits one `roi-<id>=` / `line-crossing-<id>=` key per
  saved geometry (multi-line / multi-ROI works),
- nvdsanalytics direction-reference points are computed per-line as the
  perpendicular to the segment (was hard-coded vertical offsets; new math
  reduces to the prior values for the horizontal probe default, so the P5-C
  golden test stays byte-identical),
- the perception snapshot now embeds the *applied* geometry so the
  dashboard overlay matches the rules nvdsanalytics is actually running,
- every processor status post carries `payload.applied_geometry` with
  `source` / `revision` / `line_count` / `roi_count`,
- emitted line-crossing / ROI-occupancy observations carry
  `source.geometry_source` and `source.geometry_revision` lineage so a
  count can be correlated to a specific dashboard save.

### Fallback policy
- **No saved geometry** → probe default, `applied_geometry.source="default"`.
- **Operator-saved geometry, valid** → applied,
  `applied_geometry.source="operator_saved"` + the saved revision number.
- **Operator-saved geometry, invalid** (bad coordinate space, malformed
  envelope, all `enabled=false`) → fall back to probe default *and* surface
  `error_code="geometry_invalid"` on processor status. The camera keeps
  running so it doesn't go dark while operators fix the save. The supervisor
  would otherwise thrash on a persistently-bad config (fail → retry → poll
  same envelope → fail), which is why "hard fail" was rejected.

### Files touched
- `app/geometry/__init__.py` — new package marker.
- `app/geometry/session_geometry.py` — pydantic models mirroring the
  api-server's `GeometryConfig` wire contract (slug IDs, normalized point
  bounds, polygon/line cardinality, `extra="forbid"` to surface api drift).
- `app/geometry/resolver.py` — `resolve_session_geometry()` envelope →
  `AnalyticsGeometry` conversion with `ResolvedGeometry` lineage wrapper.
- `app/runtimes/deepstream_runtime/inference.py` — `build_analytics_config`
  accepts an optional pre-built `AnalyticsGeometry`, loops over rois/lines,
  emits perpendicular line-crossing reference points.
- `app/runtimes/deepstream_runtime/metadata_probe.py` —
  `DeepStreamMetadataProcessManager` accepts `analytics_geometry`; forwarded
  to both validation + INI-write call sites.
- `app/runtimes/deepstream_runtime/worker.py` — resolves geometry once in
  `__init__` (analytics mode only), forwards to PM factory + perception
  snapshot, injects `applied_geometry` into `_build()` status payload, sets
  `error_code="geometry_invalid"` on degrade.
- `app/observations/observation_adapter.py` — `deepstream_source_block`
  accepts `geometry_source` / `geometry_revision` kwargs (flat, omitted when
  None — keeps probe-only test fixtures unchanged).
- `tests/test_session_geometry_models.py` — new pydantic validation tests.
- `tests/test_session_geometry_resolver.py` — new resolver behavior tests
  covering null, valid, invalid, all-disabled, IDs/labels, pixel clamp.
- `tests/test_worker_geometry_resolution.py` — new worker integration tests
  for resolution, status payload, factory forwarding, error_code policy.
- `tests/test_perception_geometry.py` — new multi-line / multi-ROI INI
  rendering tests + diagonal-line perpendicular reference math; the P5-C
  golden test remains.
- `tests/test_observation_adapter_contract.py` — `deepstream_source_block`
  geometry-lineage variants.
- `docs/api-contracts.md` *(api-server repo)* — P6-A "consumed in P6-C" note
  past-tensed.

### Testing/checks
- `uv run ruff check .` / `ruff format --check .` — clean.
- `uv run mypy app` — clean.
- `uv run pytest -q` — all pass.

### Known caveats / follow-ups
- **`direction` not enforced at runtime.** Operator-saved
  `direction: "in"|"out"` is preserved in observation lineage (and is
  validated in the pydantic model) but nvdsanalytics still runs `mode=loose`
  — a count fires on any crossing. Strict-direction enforcement requires
  swapping the perpendicular reference-point order on `"out"` and switching
  to `mode=strict`; deferred to a follow-up.
- **ID collision with the probe default.** If an operator saves an ROI with
  `id=entry-roi` (the probe-default ID), observation rows from the two
  sources can't be distinguished by ID alone. Not a runtime issue, but
  downstream historical-analysis consumers should join on
  `source.geometry_source` + `source.geometry_revision` to disambiguate.
- **Restart-time only.** A mid-session dashboard save bumps the revision
  but the running worker keeps the old geometry. The worker picks up the
  new revision on its next eligible session. Mid-session reload would
  require shutting the analytics container down and restarting it cleanly;
  out of scope for P6-C.
- **No dashboard "applied geometry" chip yet.** The lineage flows in the
  processor status payload (`processor_statuses.payload.applied_geometry`)
  and on observation `source` blocks, but there's no `/v1/cameras/{id}/
  processor-status` surface for the dashboard to consume cleanly. The
  follow-up is to add that endpoint + a Monitor-tab chip.
- **P6-D end-to-end verification** (run a live session, walk through the
  scene, confirm observations reference the saved IDs and revision) is a
  separate milestone.

---

## 2026-05-22 - P6-A: Geometry config in eligible-session schema

### Summary
Lock-step wire-contract rename to match the api-server P6-A geometry config
API. The eligible-session placeholder field `zones_lines` (always null, never
consumed) is renamed to `geometry_config`. The api-server now populates it
with the camera's dashboard-authored ROI/line geometry as a
`{"revision": <int>, "config": {...}}` envelope (or null when none is saved).

This is a **schema/contract change only** — no runtime behavior changes. The
field is kept loosely typed (`dict[str, Any] | None`) and is **not consumed**:
the DeepStream worker continues to use the hard-coded
`compute_analytics_geometry()` probe default. Actually applying the saved
geometry is P6-C.

### Files touched
- `app/observations/schemas.py` — `EligibleSession.zones_lines` →
  `geometry_config: dict[str, Any] | None`.
- `tests/conftest.py` — `eligible_session_dict()` fixture key rename.
- `tests/test_api_client.py` — two new cases: `geometry_config` defaults to
  None when absent; a `{revision, config}` envelope parses and round-trips.
- `docs/api-contracts.md` — eligible-sessions example + field note.

### Testing/checks
- `uv run pytest -q` — all pass.
- `uv run ruff check .` / `ruff format --check .` — clean.
- `uv run mypy app` — clean.
- No runtime / DeepStream / observation behavior changed.

### Follow-up TODOs
- P6-C: the DeepStream worker loads `geometry_config` from the eligible
  session, converts normalized coords to pixels, and builds the
  nvdsanalytics config from it (replacing the probe default).

---

## 2026-05-21 - P5-C: Publish Latest Perception Snapshots

### Summary
The analytics worker now publishes a low-rate (~1/s) **latest
perception snapshot** — a JSON description of what it currently
sees: detected objects (bbox pixel + normalized, tracker id, class,
confidence, ROI labels) plus the configured ROI/line geometry. It
POSTs to the dedicated ``POST /v1/internal/processor/perception-
snapshots`` api-server endpoint, which upserts one latest row per
camera. This is a **dedicated latest-state path, separate from
observations** (the append-only event log): separate package,
builder, API-client method, counters — nothing added to
ObservationType / ObservationWriter. A future dashboard overlay
(P5-D) consumes it. **No DB migration in this repo.**

### What changed
- New ``app/perception/`` package: ``schemas.py`` (the
  ``PerceptionSnapshotPayload`` request + response envelope) and
  ``builder.py`` (``build_perception_snapshot`` — a pure, fully
  defensive ``NeutralFrameMetadata`` → snapshot-dict converter;
  object list capped at ``perception_snapshot_max_objects``).
- ``app/runtimes/deepstream_runtime/inference.py`` — extracted
  ``compute_analytics_geometry`` + geometry dataclasses
  (``GeometryPoint`` / ``RoiGeometry`` / ``LineGeometry`` /
  ``AnalyticsGeometry``) as the single source of the ROI/line
  numbers. ``build_analytics_config`` now consumes it; its INI
  output is byte-identical (pinned by a golden test).
- ``app/api_client/`` — ``post_perception_snapshot`` on
  ``UpstreamApiClient`` + the ``ProcessorApi`` Protocol.
- ``app/config.py`` — ``perception_snapshot_enabled`` (default
  true), ``perception_snapshot_interval_seconds`` (1.0),
  ``perception_snapshot_max_objects`` (50).
- ``app/runtimes/deepstream_runtime/worker.py`` — analytics-mode
  consumer posts a snapshot at most once per interval, AFTER the
  observation emits, best-effort (a failure increments a counter
  and is swallowed — it never crashes the loop or disturbs
  observations). New ``_MetadataCounters`` perception fields +
  status-payload keys (``perception_snapshots_enabled`` /
  ``_posted`` / ``_write_errors_count`` / ``_interval_seconds`` /
  ``last_perception_snapshot_at``).

### Files
- ``app/perception/__init__.py`` / ``schemas.py`` / ``builder.py`` (new).
- ``app/runtimes/deepstream_runtime/inference.py`` — geometry helper.
- ``app/api_client/base.py`` / ``upstream_api.py`` — new method.
- ``app/config.py`` — three ``UPSTREAM_PERCEPTION_SNAPSHOT_*`` settings.
- ``app/runtimes/deepstream_runtime/worker.py`` — emit path + counters.
- ``tests/test_perception_geometry.py`` / ``test_perception_builder.py``
  / ``test_perception_emit.py`` (new); ``tests/test_api_client.py`` /
  ``test_config.py`` / ``tests/_fakes.py`` / ``tests/conftest.py`` —
  perception coverage + ``FakeApi.post_perception_snapshot``.
- ``.env.example`` — three new env vars.

### Test runs
``uv run pytest`` — all pass (57 new). ``uv run ruff check .`` /
``ruff format --check .`` — clean. ``uv run mypy app`` — clean (strict).

### Strict non-goals
- No dashboard overlay / bounding-box drawing (P5-D).
- No geometry editor, no dashboard-driven geometry, no rules/events.
- Latest-state only — no historical snapshots. Observations unchanged.

### Deploy note
Deploy the api-server P5-C endpoints first; the processor emitter is
best-effort, so a mixed-version window only produces error-counter
noise, never a crash.

---

## 2026-05-21 - P4-E: Dwell Observations

### Summary
Layers time-based observations on top of P4-D's stable zone
membership. ``processor.dwell_started`` fires immediately after
the matching ``zone_entered``; ``processor.dwell_updated`` emits
periodically while the tracked object remains inside;
``processor.dwell_exceeded`` fires once when the dwell duration
crosses a configured threshold. Re-entering a zone produces a NEW
``dwell_session_id`` with a fresh threshold. Coordinated two-repo
change (api-server widens the Literal first; processor adds
DwellState + worker emit path in lockstep). **No DB migration.**

### What changed
- **upstream-api-server** (separate commit `feat: allow processor
  dwell observations`): widens ``ProcessorObservationType`` with
  three new strings; adds three positive tests + re-aims the
  future-types guard at track / direction / telemetry / debug.
- **upstream-processor**:
  - ``app/observations/schemas.py`` — widen ``ObservationType``
    Literal + ``OBSERVATION_TYPES`` tuple.
  - ``app/observations/observation_taxonomy.py`` — flip the
    three dwell rows from ``deferred`` to ``api_supported``;
    update ``dwell_started`` / ``dwell_updated`` row payloads.
  - ``app/observations/observation_adapter.py``:
    - new ``CANDIDATE_DWELL_*`` constants in the tuple;
    - ``DwellEvent`` frozen dataclass + three idempotency-key
      builders;
    - ``_ActiveDwell`` per-(track, zone) bookkeeping (started_at,
      last_update_at, update_sequence, exceeded_emitted);
    - ``DwellState`` adapter — ingests zone events via
      ``absorb_zone_event`` and per-frame ticks via ``tick``.
      Single-shot exceeded; per-emit update_sequence; disappearance
      does NOT close the session.
  - ``app/config.py`` — two new ``Settings`` fields:
    ``deepstream_dwell_update_interval_seconds: float = 30.0`` and
    ``deepstream_dwell_exceeded_threshold_seconds: float = 60.0``
    (env-overridable; both validated > 0).
  - ``app/runtimes/deepstream_runtime/worker.py``:
    - imports DwellState;
    - instantiates ``dwell_state`` parallel to the other states;
    - ``_emit_zone_transitions`` gains an optional ``dwell_state``
      arg and feeds zone events into DwellState after each
      successful zone emit, writing ``dwell_started`` events
      immediately;
    - new ``_emit_dwell_tick`` consumer-loop helper runs each
      frame to emit ``dwell_updated`` + ``dwell_exceeded`` events;
    - new ``_emit_one_dwell_event`` helper shared by the
      zone-event-driven path and the tick path;
    - ``_MetadataCounters`` gains four counters
      (started / updated / exceeded / dwell_write_errors);
    - analytics status payload gains six new fields
      (dwell_observations_enabled, the three per-type counters,
      the write-error counter, the two configured timers).
  - ``.env.example`` documents the two new env vars.
  - ``tests/conftest.py`` ``fast_settings`` adds sub-second
    defaults for both dwell timers.

### Tests
- **363 passing** in upstream-processor (323 prior + 40 net new).
  ruff + ruff-format + ``mypy app`` clean.
- **260 passing** in upstream-api-server (257 prior + 3 new).
- ``tests/test_dwell_state.py`` — NEW, 25 pure-adapter tests.
- ``tests/test_deepstream_dwell_emit.py`` — NEW, 15 worker tests.
- ``tests/test_observation_adapter_contract.py`` — split P4-D
  ``test_dwell_track_telemetry_types_still_unsupported`` into a
  new ``test_dwell_types_now_in_server_whitelist`` (positive) and
  ``test_direction_track_telemetry_types_still_unsupported`` (the
  re-aimed guard).
- ``tests/test_observation_taxonomy.py`` — three new
  ``test_dwell_*_is_now_api_supported`` invariants.
- ``tests/test_deepstream_roi_occupancy_emit.py`` /
  ``tests/test_deepstream_zone_transition_emit.py`` — drop the
  three dwell types from their now-stale future-types guards.

### Accuracy caveat
TrafficCamNet misses frames; nvtracker can drop/reassign IDs;
nvdsanalytics ROI labels can flap. **P4-E is judged by DwellState
correctness given stable zone events, not by perfect scene
accuracy.** Live verification is a smoke test. The classification
rubric: detection / tracker / analytics / zone-state issues are
NOT P4-E regressions — only a DwellState bug is.

### Out of scope (next milestones)
- ``processor.dwell_ended`` (zone_exited already signals end).
- Server-side idempotency.
- Per-zone configurable dwell thresholds.
- Dashboard-driven thresholds.
- Direction / track-update / telemetry types.

---

## 2026-05-21 - P4-D Follow-up: ROI Label / Analytics Class Filter Fix

### Summary
P4-D's live verification produced zero zone events because the
upstream metadata pipeline never attached ``roi_labels`` to any
tracked object. Triage at the time blamed detection / analytics
ambiguously; a second look at the live evidence pinpointed
**geometry** as the real root cause: operator bbox bottom-center
was at ``y=719`` on a 720-tall frame, while our ROI ended at
``y=648`` (90% of 720). nvdsanalytics uses the bbox bottom-center
as the per-object ROI reference point, so the object was outside
the ROI and ``roiStatus`` stayed empty. ``class-id=-1`` was
already in the config; the class-filter hypothesis was wrong.

This commit widens the probe-default ROI, makes both the
geometry margin and the class filter env-configurable for future
production tuning, and adds analytics-mode diagnostics so the
next live triage requires no guesswork.

### What changed
- ``app/config.py`` — two new ``Settings`` fields:
  ``deepstream_analytics_roi_margin_pct: float = 0.05`` (env:
  ``UPSTREAM_DEEPSTREAM_ANALYTICS_ROI_MARGIN_PCT``) and
  ``deepstream_analytics_class_ids: str = "all"`` (env:
  ``UPSTREAM_DEEPSTREAM_ANALYTICS_CLASS_IDS``).
- ``app/runtimes/deepstream_runtime/inference.py``:
  - New helper ``normalize_analytics_class_ids(value)`` — the
    single source of truth for the ``class-id=`` string;
    accepts ``all`` / ``-1`` / ``""`` / single integer; rejects
    comma-lists + non-integer typos with clear errors.
  - ``build_analytics_config(...)`` gains
    ``roi_margin_pct: float = 0.05`` and ``class_ids: str = "all"``
    keyword args; the previous hard-coded 20%/10% margin and
    ``class-id=-1`` strings are now parameterized; range check on
    margin (``[0.0, 0.5)``); class_ids passed through the helper.
- ``app/runtimes/deepstream_runtime/metadata_probe.py`` —
  ``DeepStreamMetadataProcessManager`` constructor + work-dir
  builder thread the new knobs through; defaults are backward-
  compatible.
- ``app/runtimes/deepstream_runtime/worker.py``:
  - ``_default_pm_factory`` passes the settings into the PM.
  - ``_metadata_payload`` analytics block surfaces five new
    diagnostic fields: ``analytics_class_ids``,
    ``analytics_roi_margin_pct``,
    ``analytics_object_class_id_counts``,
    ``objects_with_roi_labels_count``, ``roi_labels_sample``.
- ``.env.example`` documents the two new env vars next to the
  existing analytics knobs.

### Tests
- **319 passing** in upstream-processor (307 prior + 12 net new).
  ruff + ruff-format + ``mypy app`` clean.
- ``tests/test_deepstream_analytics_probe.py`` — 9 new tests
  covering 5% default margin (with regression guard on the broken
  20%/10% values), explicit margin override (0.0 = full frame),
  invalid margin rejection (negative / 0.5 / 1.0), default
  ``class_ids="all"`` maps to ``-1``, explicit single-int filter,
  ``-1`` / ``""`` alias handling, comma-list rejection,
  non-integer rejection, settings-default pin.
- ``tests/test_deepstream_zone_transition_emit.py`` — 3 new
  worker-level diagnostics tests: status payload exposes both
  settings + per-class object counts + ``objects_with_roi_labels_count``;
  P4-D live failure pattern recreated showing zero zone events
  with diagnostic counters that classify the issue as analytics-
  side; env override propagates.

### Why this is NOT a P4-D adapter regression
The state machine never saw "track observed inside ROI" because
the upstream metadata didn't attach ``roi_labels``. The adapter
behaved correctly on the input it received (Case A: never seen
inside → no-op). The unit tests already prove every adapter
transition; this commit fixes the metadata-side gap so the
adapter has the inputs it needs.

### Operator-facing change
The probe ROI is now wider by default. Production scenes that
need narrower geometry can override via env. The class filter is
unchanged from the operator's perspective (``-1`` was already
the default) but the knob now exists for production tuning.

### Out of scope (still deferred)
- Dashboard-driven per-camera ROI / line geometry.
- Multi-class filter in a single nvdsanalytics section.
- Dwell observations (P4-E).
- Server-side idempotency.

---

## 2026-05-21 - P4-D: Zone Entered / Zone Exited Observations

### Summary
Combines P3's stable ``tracker_id`` with P3.5's per-object ROI
labels to emit per-track zone transition events. When the same
tracked object is stably inside an ROI for N frames, emit
``processor.zone_entered``; when the same tracked object is
observed without the ROI label for N frames, emit
``processor.zone_exited``. **Object disappearance does NOT emit
zone_exited** — a tracker drop-frame is indistinguishable from a
real exit, so absence is treated as silent. Coordinated two-repo
change (api-server widens the Literal first; processor adds the
adapter + worker emit path in lockstep).

### What changed
- **upstream-api-server** (separate commit `feat: allow processor
  zone transition observations`): widens
  ``ProcessorObservationType`` with the two strings; adds two
  positive tests + ``test_future_observation_types_still_rejected``
  guarding dwell / direction / track / telemetry. No DB
  migration.
- **upstream-processor**:
  - ``app/observations/schemas.py`` — widen ``ObservationType``
    Literal + ``OBSERVATION_TYPES`` tuple.
  - ``app/observations/observation_taxonomy.py`` — flip both
    ``processor.zone_entered`` and ``processor.zone_exited`` rows
    from ``deferred`` to ``api_supported``.
  - ``app/observations/observation_adapter.py``:
    - new ``CANDIDATE_ZONE_*`` constants in the tuple;
    - new ``ZoneTransitionEvent`` frozen dataclass + two
      idempotency-key builders;
    - new ``ZoneMembershipState`` dataclass (~150 lines): per
      ``(track_id, zone_id)`` committed + pending debounce + per
      ``(track_id, zone_id, transition_type)`` sequence counter.
      First-observed-outside is a no-op (Case A). Inside-flap
      clears pending (Case B). Transitions accumulate then commit
      + emit when stable (Case C). Disappearance never decrements
      / never times out / never emits.
  - ``app/config.py`` — new
    ``deepstream_zone_membership_stability_frames: int = 5``
    (env-overridable via
    ``UPSTREAM_DEEPSTREAM_ZONE_MEMBERSHIP_STABILITY_FRAMES``).
  - ``app/runtimes/deepstream_runtime/worker.py``:
    - imports ``ZoneMembershipState``;
    - instantiates ``zone_state`` parallel to ``line_state`` and
      ``roi_state``;
    - new ``_emit_zone_transitions`` consumer-loop helper mirrors
      ``_emit_roi_occupancies`` (same error-handling, same writer
      contract, same counters surface);
    - ``_MetadataCounters`` gains
      ``zone_entered_observations_emitted`` /
      ``zone_exited_observations_emitted``;
    - analytics status payload gains
      ``zone_transition_observations_enabled``, the two per-type
      counters, and ``zone_membership_stability_frames``.
  - ``.env.example`` — documents the new knob.
  - ``tests/conftest.py`` — ``fast_settings`` adds
    ``deepstream_zone_membership_stability_frames=1`` so existing
    tests get immediate-transition behavior without per-test
    overrides.

### Tests
- **307 passing** in upstream-processor (267 prior + 40 net new).
  ruff + ruff-format + ``mypy app`` clean.
- **257 passing** in upstream-api-server (254 prior + 3 new).

### Accuracy caveat
TrafficCamNet misses frames; nvtracker drops/reassigns IDs;
nvdsanalytics labels flap. **P4-D is judged by adapter correctness
given the metadata it sees, not by scene accuracy.** Live
verification is a smoke test. When live results look messy,
classify cause as detection / tracker / analytics / adapter —
only adapter is a P4-D regression.

### Out of scope (next milestones)
- Dwell observations (P4-E). Builds on ``ZoneMembershipState``'s
  ``stable_membership`` map.
- Object-disappearance-emits-exit policy.
- Per-zone configurable thresholds.
- Server-side idempotency column.
- Direction / track-update / telemetry types.

---

## 2026-05-20 - P4-C Cleanup: ROI Debounce / Hysteresis

### Summary
Pure observability fix on top of P4-C: TrafficCamNet's detection
count flaps on edge-case frames, and P4-C's undamped on-change
policy turned each flap into a real observation (205 POSTs in one
180s cycle, including bursts of 8 POSTs/3s during active operator
movement). Adds a per-ROI debounce: a new count must hold for
``change_stability_frames`` consecutive frames (default 5, ≈166ms
at 30fps) before committing as a change. First-emit and the 30s
heartbeat interval are unchanged in cadence. **No new observation
types, no API change, no DB migration.**

### What changed
- ``app/observations/observation_adapter.py``:
  - ``RoiOccupancyState`` gains ``change_stability_frames: int = 1``
    (legacy P4-C behavior) + ``pending_count_by_roi`` /
    ``pending_seen_frames_by_roi`` accumulator fields.
  - ``__post_init__`` rejects ``change_stability_frames < 1``.
  - ``_classify_emit`` now returns
    ``tuple[reason, snapshot, total, class_counts] | None``. The
    change branch updates the pending counter on every frame
    (even when it doesn't emit), and the interval branch reports
    the LAST STABLE count when a candidate is pending. The
    pending state survives the heartbeat so the eventual commit
    still happens.
- ``app/config.py``: adds
  ``deepstream_roi_change_stability_frames: int = 5`` (production
  default; env-overridable via
  ``UPSTREAM_DEEPSTREAM_ROI_CHANGE_STABILITY_FRAMES``).
- ``app/runtimes/deepstream_runtime/worker.py``:
  - Threads ``change_stability_frames`` into ``RoiOccupancyState``.
  - Status payload surfaces ``roi_change_stability_frames`` so
    operators can tell "no observations: steady scene" apart from
    "no observations: debounce too aggressive".
- ``tests/conftest.py`` — ``fast_settings`` adds
  ``deepstream_roi_change_stability_frames=1`` so existing P4-C
  emit tests stay valid without per-test overrides.
- ``tests/test_observation_adapter_contract.py`` — 9 new tests for
  debounce semantics (threshold rejection, immediate-change
  backward compat, suppression, commit, flap-clears-pending,
  candidate-switch, heartbeat-with-stable-count,
  eventually-commits-after-heartbeat, sequence-only-on-emit).
- ``tests/test_deepstream_roi_occupancy_emit.py`` — 6 new tests
  for the worker integration (flap suppression, stable commit,
  heartbeat-with-stable-count under debounce, Settings default
  is 5, invalid threshold rejected, status payload exposes
  threshold).
- ``.env.example`` documents both
  ``UPSTREAM_DEEPSTREAM_ROI_MIN_INTERVAL_SECONDS`` and the new
  ``UPSTREAM_DEEPSTREAM_ROI_CHANGE_STABILITY_FRAMES``.

### Backward compatibility
- ``RoiOccupancyState`` constructor default (``change_stability_frames=1``)
  matches the pre-cleanup behavior — every direct caller of the
  state machine in P4-A's tests keeps working with zero changes.
- Production wiring sets the value to 5 via Settings; existing
  emit tests use ``fast_settings`` (= 1) so the worker-level emit
  contracts are unchanged in tests.
- The status payload added one new field
  (``roi_change_stability_frames``); other payload keys unchanged.
- ``processor.line_crossing`` unchanged.

### Expected production impact
ROI observation volume during active scenes should drop sharply
(the 0↔1↔0 flap pattern that produced 8 POSTs/3s now suppresses
unless the count holds for ≥5 frames). The interval heartbeat
still ensures observers see fresh state at least every 30 s; a
truly stable population change emits ~one observation per state
transition instead of per flap.

### Out of scope (next milestones)
- Server-side ``idempotency_key`` column + UNIQUE constraint.
- ``processor.zone_entered`` / ``zone_exited`` / ``dwell_exceeded``
  (P4-D; needs a new ``ZoneMembershipState`` adapter).
- Per-class ROI breakdown (probe-side change).
- Dashboard-driven ROI geometry.

### Tests
- **267 passing** (252 prior + 15 new). ruff + ruff-format +
  ``mypy app`` clean. No Docker / GPU / pyds / network.

---

## 2026-05-20 - P4-C: ROI Occupancy Observation Write Path

### Summary
Closes P4-B's deferred item: the DeepStream analytics worker now also
emits ``processor.roi_occupancy`` observations to the api-server's
``observations`` table. Emit policy is **first-seen + on-change +
30-second interval heartbeat** — never per-frame. The P4-B
line-crossing path is unchanged and runs in parallel; the worker's
two emit helpers operate independently and the status payload
exposes both counters. Coordinated with the api-server commit
``feat: allow processor ROI occupancy observations`` which widens
the ``ProcessorObservationType`` Literal in lock-step. No DB
migration.

### What changed (api-server side, separate repo)
- ``src/upstream_api/schemas/processor.py`` — Literal widened
  with ``"processor.roi_occupancy"``.
- ``tests/test_processor.py`` — new
  ``test_observation_roi_occupancy_accepted``; existing
  ``test_observation_line_crossing_accepted`` +
  ``test_observation_invalid_type_rejected`` are unchanged.

### What changed (this repo)
- ``app/observations/schemas.py`` — ``ObservationType`` Literal +
  ``OBSERVATION_TYPES`` tuple gain ``"processor.roi_occupancy"``.
- ``app/observations/observation_taxonomy.py`` — flips the
  ``processor.roi_occupancy`` row from ``deferred`` to
  ``api_supported``.
- ``app/observations/observation_adapter.py`` —
  ``RoiOccupancyState._should_emit`` (was ``bool``) becomes
  ``_classify_emit`` returning ``str | None`` (the emit reason or
  None). ``absorb_roi_counts`` merges
  ``{"emit_reason": <reason>}`` into the event's ``debug`` dict
  so consumers can tell first / change / interval apart without
  re-deriving. ``RoiOccupancyEvent`` itself unchanged.
- ``app/config.py`` — adds
  ``deepstream_roi_min_interval_seconds: float = 30.0`` so
  operators can tune the heartbeat cadence and tests can override
  it via ``fast_settings``.
- ``app/runtimes/deepstream_runtime/worker.py``:
  - Imports ``RoiOccupancyState``.
  - ``_drive_metadata_probe`` instantiates ``roi_state`` next to
    ``line_state`` when ``analytics_mode + writer`` is true.
  - Consumer loop calls a new ``_emit_roi_occupancies(...)``
    immediately after the existing line-crossing emit.
  - The new helper mirrors ``_emit_line_crossings`` exactly:
    pulls ``frame.source["analytics"]["roi_counts"]``
    defensively, coerces to ``dict[str, int]``, hands to
    ``roi_state.absorb_roi_counts(...)``, emits one
    ``processor.roi_occupancy`` observation per returned event.
    Error handling identical: ``ApiSessionGoneError`` propagates,
    ``ApiValidationError`` / ``TRANSIENT_API_ERRORS`` are logged
    + counted + continue.
  - ``_MetadataCounters`` gains
    ``roi_occupancy_observations_emitted: int = 0``.
  - ``_metadata_payload`` surfaces the new counter alongside
    ``line_crossing_observations_emitted``.
- ``tests/test_deepstream_roi_occupancy_emit.py`` (new, 18 tests).
- ``tests/test_observation_adapter_contract.py`` — six existing
  ROI tests gain ``emit_reason`` assertions; two new tests pin
  ``debug.emit_reason`` location + non-overwrite of caller debug;
  the P4-B ``test_roi_occupancy_still_not_in_server_whitelist``
  guard is replaced by
  ``test_roi_occupancy_now_in_server_whitelist`` +
  ``test_zone_dwell_track_types_still_unsupported`` (regression
  on the next-deferred types).
- ``tests/test_observation_taxonomy.py`` — new
  ``test_roi_occupancy_is_now_api_supported``.
- ``tests/test_deepstream_line_crossing_emit.py`` — P4-B's
  ``test_roi_occupancy_still_unsupported_by_writer`` flips;
  ``test_no_roi_occupancy_writes_in_p4b`` is renamed to
  ``test_line_crossing_unchanged_by_p4c`` and asserts the line
  emit count is unaffected.

### Idempotency / dedupe
- Same model as P4-B: client-side
  ``RoiOccupancyState.sequence_by_roi`` + bounded internal
  watermarks. Deterministic key
  ``sess:<session_id>:processor.roi_occupancy:<roi_id>:<sequence>``
  embedded in ``payload.idempotency_key``.
- Sequence increments per emit (NOT per frame); 50 frames of an
  unchanged ROI still produce only 1 emit + idempotency key
  ending ``:1``.

### Status payload addition (analytics + writer)
```
"observations_enabled": true,
"observations_emitted_count": <int>,
"line_crossing_observations_emitted": <int>,
"roi_occupancy_observations_emitted": <int>,
"observation_write_errors_count": <int>,
"last_observation_at": "<iso8601>" | null
```

The two type-specific counters track independently so an operator
can tell line- vs ROI-throughput apart from one status row.

### Out of scope (next milestones)
- ``processor.zone_entered`` / ``zone_exited`` /
  ``dwell_exceeded`` (need a new ``ZoneMembershipState`` that
  diffs per-track ROI membership across frames).
- Per-class ROI breakdown (probe-side change to aggregate
  per-object ``roiStatus`` lists).
- Server-side ``idempotency_key`` column + UNIQUE constraint.
- Dashboard-driven ROI geometry — still hand-authored in
  ``build_analytics_config``.
- Direction / track-update / telemetry types.

### Tests
- This repo: **252 passing** (230 prior + 22 net new: 18 ROI emit
  + 2 emit_reason + 1 roi-api-supported + 1 zone/dwell guard).
  ruff + ruff-format + ``mypy app`` clean.
- api-server: **254 passing** (253 prior + 1 new). ruff +
  ruff-format + ``mypy src`` clean.
- Live verification on crowdcountS2 is an operator step in
  ``docs/local-dev.md`` — depends on the api-server change being
  deployed first.

---

## 2026-05-20 - P4-B: Line-Crossing Observation Write Path

### Summary
Closes the loop the P4-A / P4-A.5 milestones designed: the
DeepStream analytics worker now turns each ``objLCCumCnt`` increment
into one ``processor.line_crossing`` row in the api-server's
``observations`` table via the existing
``POST /v1/internal/processor/observations``. Coordinated across two
repos (this commit covers the processor side; the api-server side
lands separately as ``feat: allow processor line crossing
observations``). Event-only — ROI occupancy, zone events, dwell,
direction all remain deferred per the P4-A.5 taxonomy.

### What changed (api-server side, separate repo)
- ``src/upstream_api/schemas/processor.py`` — ``ProcessorObservationType``
  Literal widened with ``"processor.line_crossing"``. No DB
  migration (``observations.type`` is plain ``text``, no CHECK).
- ``tests/test_processor.py`` — adds
  ``test_observation_line_crossing_accepted``; existing
  ``test_observation_invalid_type_rejected`` is unchanged.

### What changed (this repo)
- ``app/observations/schemas.py`` — ``ObservationType`` Literal +
  ``OBSERVATION_TYPES`` tuple gain ``"processor.line_crossing"``.
- ``app/observations/observation_taxonomy.py`` — taxonomy row's
  ``status`` flips from ``proposed_first_api_add`` to
  ``api_supported``.
- ``app/runtimes/deepstream_runtime/worker.py``:
  - ``DeepStreamWorker.__init__`` accepts optional
    ``writer: ObservationWriter | None``.
  - New ``_emit_line_crossings`` helper extracts
    ``frame.source["analytics"]["line_crossing_cumulative"]``, hands
    it to ``LineCrossingState.absorb_cumulative(...)``, and emits
    one ``processor.line_crossing`` observation per returned event
    via ``ObservationWriter.emit``.
  - ``_drive_metadata_probe`` instantiates ``LineCrossingState`` +
    pre-builds the ``deepstream_source_block`` once per worker when
    in analytics mode with a writer present; non-analytics modes
    ignore the writer.
  - Per-emit error handling: ``ApiSessionGoneError`` propagates
    (outer ``run()`` ends quietly); ``ApiValidationError`` is logged
    + counted (wire-contract drift; do not crash);
    ``TRANSIENT_API_ERRORS`` is logged + counted. No retry loop in
    v0 — the deterministic ``idempotency_key`` makes future retry
    safe to add.
  - ``_MetadataCounters`` gains ``observations_emitted``,
    ``line_crossing_observations_emitted``,
    ``observation_write_errors``, ``last_observation_at``.
  - ``_metadata_payload`` surfaces those fields only when a writer
    is wired (``observations_enabled=true``); otherwise
    ``observations_enabled=false`` and the counters are omitted.
- ``app/supervisor/worker_manager.py`` — ``build_worker`` passes the
  same writer it gives ``MockWorker`` into ``DeepStreamWorker``.
- ``tests/test_deepstream_line_crossing_emit.py`` (new, 15 tests):
  WorkerManager wires the writer; 0→1 emits one; 0→1→2 emits two
  with distinct deterministic idempotency keys; repeated /
  rewound cumulative emits nothing while watermark holds; payload
  top-level is runtime-neutral (DeepStream vocabulary only under
  ``source`` / ``debug``); no ROI occupancy writes; tracker- /
  metadata-only modes don't emit; ``writer=None`` is safe + status
  payload signals ``observations_enabled=False``; transient API
  error path ends in ``stopped`` with counters surfaced + no
  stream-key leakage; emit call surface matches the wire contract.
- Updated test invariants:
  - ``tests/test_observation_adapter_contract.py``:
    ``test_candidate_types_not_in_server_whitelist_yet`` replaced
    by ``test_line_crossing_now_in_server_whitelist`` +
    ``test_roi_occupancy_still_not_in_server_whitelist``.
  - ``tests/test_observation_taxonomy.py``:
    ``test_exactly_one_type_proposed_for_first_api_add`` relaxed
    to ``test_at_most_one_type_proposed_for_first_api_add``;
    ``test_first_api_add_is_line_crossing`` replaced by
    ``test_line_crossing_is_now_api_supported``.

### Idempotency / dedupe
- Client-side ``LineCrossingState.last_cumulative_by_line`` plus a
  bounded ``recently_emitted_keys`` set. Deterministic key:
  ``sess:<session_id>:processor.line_crossing:<line_id>:<cumulative_count>``.
- No server-side ``idempotency_key`` column yet (would be a P4-C
  optional change). Today duplicates are impossible within a
  container lifetime; the cumulative counter resets when the
  container restarts every ~180 s but the key stays unique per
  ``(session_id, line, cumulative_count)`` so a future server
  UNIQUE constraint is safe to add later.

### Container boundary unchanged
The pyds container still doesn't talk to the API. The webhook
secret stays in the outer worker. Adapter + writer run in the host
asyncio consumer loop, not inside docker.

### Status payload addition (analytics + writer)
```
"observations_enabled": true,
"observations_emitted_count": <int>,
"line_crossing_observations_emitted": <int>,
"observation_write_errors_count": <int>,
"last_observation_at": "<iso8601>" | null
```

Tracker / metadata / decode / inference modes are unchanged; their
payloads do NOT advertise observation fields.

### Out of scope (next milestones)
- ``processor.roi_occupancy`` write path (P4-C).
- ``processor.zone_entered`` / ``zone_exited`` / ``dwell_exceeded``
  (P4-C+; need a new ``ZoneMembershipState`` adapter).
- Server-side ``idempotency_key`` column + UNIQUE.
- Dashboard-driven line geometry — still hand-authored in
  ``build_analytics_config``.
- Direction debounce, track-update sampling, frame-summary telemetry.

### Tests
- This repo: 230 passing (214 prior + 16 net new from line-crossing
  + invariant rework). ruff + ruff-format + ``mypy app`` clean.
- api-server: 253 passing (252 prior + 1 new). ruff + ruff-format +
  ``mypy src`` clean.
- Live verification on crowdcountS2 is the operator step in
  ``docs/local-dev.md`` — depends on the api-server change being
  deployed first.

---

## 2026-05-20 - P4-A.5: Observation Taxonomy + Emission Policy

### Summary
Locks in which ``processor.*`` observation types UpStream will support,
in what order, and under what emission policy — before any api-server
widening or worker wiring happens. Adds a typed taxonomy table as the
single source of truth and a strict invariant test suite that keeps
the table, the P4-A adapter, and ``schemas.OBSERVATION_TYPES`` from
drifting. Docs-and-data-only milestone; **no api-server change, no
worker wiring, no production POST**.

### What changed
- `app/observations/observation_taxonomy.py` (new, ~270 lines):
  - ``ObservationCategory`` Literal: ``event`` | ``state`` | ``telemetry``
    (the three emission-policy classes).
  - ``ObservationStatus`` Literal: ``api_supported`` |
    ``proposed_first_api_add`` | ``deferred`` (drives the api-server
    widening order).
  - Frozen ``ObservationSpec`` dataclass with type / category / status
    / description / payload_keys / idempotency_key_pattern /
    source_signal + requirements flags + dependencies.
  - ``INITIAL_TAXONOMY`` tuple — 13 ``processor.*`` types covering
    line crossing (the first to ship), zone entered / exited /
    dwell / direction, ROI / zone / queue occupancy, per-track
    state / update, and telemetry-style frame summary / detection
    seen / scene quality warning.
  - Derived tuples ``API_SUPPORTED_TYPES``,
    ``PROPOSED_FIRST_API_ADD_TYPES``, ``DEFERRED_TYPES`` so callers
    don't need to filter the table.
  - ``spec_for(type)`` and ``types_in_category(category)`` lookup
    helpers.
- `tests/test_observation_taxonomy.py` (new, 24 tests):
  - Taxonomy non-empty, every type unique + ``processor.*`` namespace,
    every category / status valid.
  - **Exactly one** type marked ``proposed_first_api_add`` (the
    load-bearing P4-A.5 decision) and it is
    ``processor.line_crossing``.
  - ``proposed_first_api_add`` and ``api_supported`` mutually
    exclusive.
  - Taxonomy ``api_supported`` flag must be in lock-step with
    ``schemas.OBSERVATION_TYPES`` (drift fails the test).
  - Both P4-A ``CANDIDATE_*`` adapter constants have matching
    taxonomy rows.
  - Every idempotency key pattern starts with
    ``sess:<session_id>:processor.<thing>:`` and every event-style
    key has a uniqueness token; state keys never use
    ``<frame_num>``.
  - Zone-dependent types require zone geometry + tracker_id;
    ``dwell_exceeded`` has explicit ``dependencies`` on
    ``zone_entered`` / ``zone_exited``.
  - ``payload_keys`` does NOT redundantly include the universal
    ``idempotency_key`` / ``source`` / ``debug``.
  - ``ObservationSpec`` is frozen.
- `docs/architecture.md`: adds the **P4-A.5 — Observation Taxonomy +
  Emission Policy** section with the table, the first-API-addition
  decision, payload + idempotency rules, adapter state requirements
  (implemented + deferred), explicit deferred list, and the P4-B
  next-step plan.
- `docs/commit-log.md`: this entry.

### Initial type table (status snapshot)
- **proposed_first_api_add** (1): ``processor.line_crossing``
- **deferred** (12): ``zone_entered``, ``zone_exited``,
  ``dwell_exceeded``, ``direction_detected``, ``roi_occupancy``,
  ``zone_occupancy``, ``queue_length``, ``track_state``,
  ``track_update``, ``detection_seen``, ``frame_summary``,
  ``scene_quality_warning``.
- **api_supported** (0; the three ``processor.mock_*`` types in
  ``schemas.OBSERVATION_TYPES`` live outside the
  ``processor.<thing>`` taxonomy because they're test scaffolding).

### Out of scope (next milestones)
- ``processor.line_crossing`` API addition + worker wiring +
  live-verify is **P4-B**.
- ``processor.roi_occupancy`` API addition + adapter wiring is
  **P4-C** (after we have live ROI volume measurements).
- ``zone_entered`` / ``zone_exited`` / ``dwell_exceeded`` need a new
  ``ZoneMembershipState`` adapter (per-(track, zone) presence diff).
- Dashboard-driven ROI / line geometry is a separate later
  milestone; P4-B uses P3.5's hand-authored line.

### Tests
- 214 passing (190 prior + 24 new taxonomy invariants). ruff +
  ruff-format + ``mypy app`` clean. No Docker / GPU / pyds / API
  network in tests.

---

## 2026-05-20 - P4-A: Observation Adapter Contract Probe

### Summary
Defines the contract for the first two real processor observation
types (``processor.line_crossing``, ``processor.roi_occupancy``) and
the deterministic idempotency / adapter-state machinery that will
turn P3.5's analytics counters into discrete observation rows —
**without** writing any observations or touching any worker. Pure
contract-level milestone; the matching api-server PR + worker wiring
land in P4-B.

### What changed
- `app/observations/observation_adapter.py` (new) — pure / dependency-
  free module:
  - ``CANDIDATE_LINE_CROSSING_TYPE`` / ``CANDIDATE_ROI_OCCUPANCY_TYPE``
    string constants (deliberately NOT in
    ``schemas.OBSERVATION_TYPES`` yet).
  - ``LineCrossingEvent`` + ``LineCrossingState.absorb_cumulative(...)``
    — turns ``objLCCumCnt`` deltas into one event per crossing with
    a deterministic
    ``sess:<id>:processor.line_crossing:<line>:<cum>``
    idempotency key. Handles rewind (no spurious emit, watermark
    held) and replay (bounded ``recently_emitted_keys``).
  - ``RoiOccupancyEvent`` + ``RoiOccupancyState.absorb_roi_counts(...)``
    — designed but not wired. Emits on first observation, on count
    change, and on ``min_interval_seconds`` elapse; otherwise quiet.
  - ``build_*_idempotency_key`` pure helpers.
  - ``deepstream_source_block(...)`` — stable provenance dict so
    worker code doesn't inline DeepStream literals at the emit site.
- `tests/test_observation_adapter_contract.py` (new) — 31 tests
  covering type-registry guards, idempotency determinism, the full
  cumulative→discrete event mapping including the 0→1→2 pattern from
  the P3.5 live run, the 0→3 jump case, repeat / rewind / empty
  defenses, runtime-neutral payload shape (DeepStream vocabulary
  under ``source`` / ``debug``), optional-field omission,
  ``RoiOccupancyState`` emit conditions (on-change + interval +
  not-every-frame), malformed-input safety, and a regression that
  the adapter module does NOT import ``ObservationWriter`` /
  ``UpstreamApiClient`` (pure by design).
- `docs/architecture.md` — adds the **P4-A — Observation Adapter
  Contract Probe** section: api-server schema findings, candidate
  type names, payload shape, idempotency strategy, adapter state
  design, ``ObservationWriter`` review, what the api-server PR
  must contain for P4-B, what's deliberately NOT in P4-A.
- `docs/commit-log.md` — this entry.

### API contract findings (verified offline against `upstream-api-server` @ 7a6057a)
- ``POST /v1/internal/processor/observations`` exists and is
  webhook-authed. Strict ``Literal`` validation on ``type``;
  ``processor.line_crossing`` 422s today.
- DB column ``observations.type`` is plain ``text`` (no CHECK), so
  widening the whitelist is a one-string Python change on each side
  — no migration.
- No ``idempotency_key`` column, no unique constraint. Duplicate
  POSTs create duplicate rows. Client-side dedupe via
  ``LineCrossingState`` is the only mechanism in v0; the
  idempotency_key lives in ``payload`` so a future server-side
  dedupe migration is cheap.

### Idempotency keys (deterministic, session-scoped)
- Line crossing: ``sess:<session_id>:processor.line_crossing:<line_id>:<cumulative_count>``.
- ROI occupancy: ``sess:<session_id>:processor.roi_occupancy:<roi_id>:<sequence>``.
- ``track_id`` intentionally OMITTED from the line-crossing key
  (``objLCCumCnt`` is per-LINE aggregate, not per-track); attached
  as optional ``payload.track_id`` context.

### Out of scope (next milestones)
- ``ObservationWriter.emit(...)`` integration in the analytics
  worker (P4-B).
- ``schemas.OBSERVATION_TYPES`` widening + the matching api-server
  one-string PR (P4-B).
- Live verification of a real ``processor.line_crossing`` row
  landing in the ``observations`` table (P4-B).
- ``processor.roi_occupancy`` wiring (P4-C).
- ``idempotency_key`` column + ``UNIQUE`` constraint on the api-server
  (optional P4-C; not required for P4-B because client-side state
  dedupes within a container lifetime).

### Tests
- 190 passing (159 prior + 31 new contract tests). ruff + ruff-format
  + ``mypy app`` clean. No Docker / GPU / pyds / api-server in tests.

---

## 2026-05-20 - P3.5: Analytics Metadata Probe

### Summary
Splices `nvdsanalytics` into the tracker pipeline so the SDK's ROI /
line-crossing / overcrowding / direction metadata is attached to
`NvDsFrameMeta` (frame-level `objInROIcnt` / `objLCCumCnt` / `ocStatus`)
and `NvDsObjectMeta` (per-object `roiStatus` / `lcStatus` /
`dirStatus`); the pyds pad probe walks `frame_user_meta_list` /
`obj_user_meta_list`, folds the result under `frame.source["analytics"]`
and `object.debug["analytics"]`, and the worker surfaces bounded
counters + a sample of distinct labels in the status payload. New
package `deepstream.analytics_probe.v0` selects the mode via
most-specific-wins ordering (analytics > tracker > metadata > inference
> decode). v0 ships a hand-authored ROI + line-crossing config aimed
at 1280×720; dashboard-driven geometry is a later milestone. **No
observations, no metadata adapter, no rules/events, no API contract
change.**

### What changed
- `app/packages/package_registry.json` — new
  `deepstream.analytics_probe.v0` (`supports.analytics_probe: true`).
- `app/config.py` — `deepstream_analytics_labels_sample_limit` (20).
- `app/runtimes/deepstream_runtime/inference.py` — pure
  `build_analytics_config(frame_width, frame_height, roi_label,
  line_label)` returning an INI body with one `[roi-filtering-stream-0]`
  + one `[line-crossing-stream-0]` section. Central rectangle ROI
  (20-80% × 10-90% of frame), horizontal line across the middle of the
  ROI (x ∈ 25-75%), direction reference perpendicular to the line in
  `mode=loose`. Rejects non-positive dimensions / empty labels.
- `app/runtimes/deepstream_runtime/metadata_probe.py` —
  `analytics_enabled` flag on the docker command builder and the
  `DeepStreamMetadataProcessManager`; both raise `ValueError` if asked
  to enable analytics without tracker (nvdsanalytics operates on
  tracked objects). When set, `_prepare_work_dir` also writes
  `analytics_config.txt` into the per-run temp dir; container name
  prefix flips from `upstream-dst-` to `upstream-dsa-`. The
  in-container app additionally gets `--analytics-config-path`.
- `app/runtimes/deepstream_runtime/pyds_metadata_app.py` — optional
  `nvdsanalytics` element appended after `nvtracker` when
  `--analytics-config-path` is set (requires `--tracker-config-path`),
  the pad probe attaches to `analytics.src`, and the probe walks
  `frame_user_meta_list` /  `obj_user_meta_list` for
  `NVDS_*_META_NVDSANALYTICS` user-meta via
  `pyds.NvDsAnalyticsFrameMeta.cast(...)` /
  `pyds.NvDsAnalyticsObjInfo.cast(...)`. Frame-level analytics
  (`objInROIcnt`, `objLCCumCnt`, `objLCCurrCnt`, `ocStatus`) land under
  `frame.source["analytics"]`; per-object (`roiStatus`, `lcStatus`,
  `dirStatus`, `ocStatus`) under `object.debug["analytics"]`. Summary
  line gains analytics counters too.
- `app/observations/metadata_models.py` — `NeutralObjectMetadata`
  gained `debug: dict[str, Any] = Field(default_factory=dict)` so the
  per-object analytics blob has a typed home without leaking
  DeepStream-specific keys at the top level.
- `app/runtimes/deepstream_runtime/worker.py` — `_ANALYTICS_MODE`,
  mode dispatch ordering (analytics > tracker > metadata > inference >
  decode), `_default_pm_factory` flips `tracker_enabled` +
  `analytics_enabled` for analytics mode. `_MetadataCounters` gains
  analytics counters (`has_analytics_metadata`, `roi_metadata_count`,
  `line_metadata_count`, `overcrowding_metadata_count`,
  `direction_metadata_count`, `objects_with_roi_metadata`,
  `objects_with_line_metadata`, `analytics_labels: set[str]`,
  `analytics_labels_sample: list[str]` (FIFO-bounded)). The status
  payload helper adds the analytics fields only in `analytics_probe`
  mode; metadata / tracker payloads are unchanged.
- `tests/test_package_resolver.py` — analytics-package resolver case.
- `tests/test_deepstream_analytics_probe.py` — 14 new tests
  (no Docker / GPU / pyds / gi).
- `.env.example`, `docs/architecture.md`, `docs/local-dev.md`,
  `docs/commit-log.md`, `app/runtimes/deepstream_runtime/README.md`,
  `README.md` — documentation.

### Analytics config + pyds research (in-container)
- Sample config: `/opt/nvidia/deepstream/deepstream-8.0/sources/apps/
  sample_apps/deepstream-nvdsanalytics-test/config_nvdsanalytics.txt`
  (INI; `[property]`, `[roi-filtering-stream-N]`,
  `[line-crossing-stream-N]`, `[overcrowding-stream-N]`,
  `[direction-detection-stream-N]`).
- Plugin: `nvdsanalytics` GStreamer element, library
  `/opt/nvidia/deepstream/deepstream/lib/gst-plugins/
  libnvdsgst_dsanalytics.so`. No `ll-lib-file` knob needed (analytics
  is single-implementation).
- Python API confirmed via `python3 -c "import pyds; ..."` in the
  DS 8.0 + pyds container:
  - `pyds.NvDsAnalyticsFrameMeta` exposes `objCnt`, `objInROIcnt`,
    `objLCCumCnt`, `objLCCurrCnt`, `ocStatus`, `unique_id` + `cast`.
  - `pyds.NvDsAnalyticsObjInfo` exposes `roiStatus`, `lcStatus`,
    `ocStatus`, `dirStatus`, `objStatus`, `unique_id` + `cast`.
  - `pyds.NVDS_FRAME_META_NVDSANALYTICS` /
    `pyds.NVDS_OBJ_META_NVDSANALYTICS` are the user-meta type tags.
  - `NvDsFrameMeta.frame_user_meta_list` /
    `NvDsObjectMeta.obj_user_meta_list` walk the user-meta linked
    lists, casting each entry via `pyds.NvDsUserMeta.cast(...)`.

### Out of scope (next milestones)
- `ObservationWriter` integration + `POST /observations` (P4).
- Final metadata adapter (P4).
- Dashboard-driven ROI / line geometry (operator-authored zone /
  line config, currently hand-rolled in `build_analytics_config`).
- Rules / events on top of the analytics counters.

---

## 2026-05-20 - P3: Tracker Metadata Probe

### Summary
Splices `nvtracker` (NvTracker low-level IOU) into the pyds metadata
pipeline so `NvDsObjectMeta.object_id` becomes a real stable tracker id
across frames. New package `deepstream.tracker_probe.v0` selects the new
mode in `DeepStreamWorker` (most-specific-wins ordering:
tracker > metadata > inference > decode). Same supervisor, same pyds
container image, same SIGINT-driven shutdown — only the GStreamer
pipeline, the in-container CLI, and the status payload gain a small
surface. **No observations.** No metadata adapter, no nvdsanalytics, no
line crossing / zones, no API contract change.

### What changed
- `app/packages/package_registry.json` — new
  `deepstream.tracker_probe.v0` (`supports.tracker_probe: true`),
  separate from the four existing packages so operators can compare
  `metadata_probe` (detection metadata, no tracker) vs `tracker_probe`
  (detection metadata + stable tracker IDs) side by side.
- `app/config.py` — five new env-overridable knobs:
  `deepstream_tracker_lib_path`,
  `deepstream_tracker_algorithm` (`iou` for v0),
  `deepstream_tracker_width` (960), `deepstream_tracker_height` (544),
  `deepstream_tracker_ids_sample_limit` (10).
- `app/runtimes/deepstream_runtime/inference.py` — pure
  `build_tracker_config(algorithm="iou")` helper: returns a
  `%YAML:1.0` low-level config modelled on the DS 8.0 sample
  `config_tracker_IOU.yml` (BaseConfig / TargetManagement /
  TrajectoryManagement / DataAssociator). Unknown algorithms are
  rejected up front with a clear `ValueError`.
- `app/runtimes/deepstream_runtime/metadata_probe.py` — pure
  `build_metadata_docker_command` accepts `tracker_enabled` +
  `tracker_lib_path` / `tracker_width` / `tracker_height`; when set,
  the inner `bash -lc` grows
  `--tracker-config-path <wd>/tracker_config.yml --tracker-lib-path ...
  --tracker-width N --tracker-height N`. The
  `DeepStreamMetadataProcessManager` gained the same fields, writes
  `tracker_config.yml` into the per-run temp dir alongside `pgie.txt` +
  `pyds_metadata_app.py`, and flips the container name prefix from
  `upstream-dsm-` to `upstream-dst-` for tracker runs so the two modes
  are distinguishable in `docker ps`. Shutdown hardening (P2 cleanup)
  applies unchanged.
- `app/runtimes/deepstream_runtime/pyds_metadata_app.py` — optional
  `nvtracker` element spliced between `nvinfer` and `nvvideoconvert`
  when `--tracker-config-path` is set; the pad probe attaches to the
  **tracker src pad** in that case (so `NvDsObjectMeta.object_id` is
  populated). New CLI args: `--tracker-config-path`,
  `--tracker-lib-path`, `--tracker-width`, `--tracker-height`. Same
  near-zero-work probe (no HTTP, no async, no `ObservationWriter`).
- `app/runtimes/deepstream_runtime/worker.py` — new
  `_TRACKER_MODE = "tracker_probe"` and mode ordering
  `tracker > metadata > inference > decode`. The default PM factory
  for tracker mode passes the tracker knobs through. `_MetadataCounters`
  gained `unique_tracker_ids: set[str]` and a FIFO-bounded
  `tracker_ids_sample` (cap from settings). `_metadata_payload` adds
  `tracker_enabled`, `tracker_algorithm`, `unique_tracker_ids_count`,
  `tracker_ids_sample` in tracker mode; metadata-probe mode is
  unchanged (no tracker fields), so operators can tell the two modes
  apart from the status row alone.
- `tests/test_package_resolver.py` — new resolver case for
  `deepstream.tracker_probe.v0`.
- `tests/test_deepstream_tracker_probe.py` — 13 new tests covering:
  package resolution, mode dispatch
  (tracker > metadata > inference > decode), `build_tracker_config`
  shape + unknown-algorithm rejection, the docker command builder with
  and without tracker args, the real PM's `tracker_config.yml` write +
  cleanup + container-name prefix flip + pre-flight algorithm
  validation, the full `tracker_probe` lifecycle with tracker_ids
  `"101", "101", "102"` → `unique_tracker_ids_count = 2` and bounded
  sample `["101", "102"]`, bounded sample under 6 unique IDs, and a
  regression that `metadata_probe` mode does NOT add tracker fields.
  No Docker / GPU / pyds / gi.
- `.env.example`, `docs/architecture.md`, `docs/local-dev.md`,
  `app/runtimes/deepstream_runtime/README.md` — document the new
  package, knobs, payload shape, and live-verification checklist.

### Tracker choice (and what we rejected)
NvTracker low-level **IOU** is the v0 default:

- No ReID model required (NvDCF / NvSORT / NvDeepSORT all need one);
  zero extra files beyond what `libnvds_nvmultiobjecttracker.so` already
  bundles.
- Sufficient to prove the stable-ID path under a fixed-camera person
  stream (the P3 live run confirmed: `tracker_id "2"` persisted across
  1155 consecutive frames, `"5"` × 725, `"0"` × 370 — clearly real
  association, not new IDs per frame).
- We write our own `tracker_config.yml` (a small `%YAML:1.0` body in
  `build_tracker_config`) into the per-run temp dir rather than
  referencing the in-container sample path, so the runtime is
  self-contained and tunables live in this repo.

NvDCF / NvSORT / NvDeepSORT are explicitly NOT supported in v0 — the
algorithm enum rejects them with a clear error so a typo doesn't load a
config that needs a ReID model we haven't shipped.

### Live verification
Run on crowdcountS2 (RTX 5070 Ti, Blackwell sm_120) against a real
validated upstream-agent stream:

- Package override active for the session
  (`forced=deepstream.tracker_probe.v0`).
- Worker log: `mode=tracker_probe`, `model=trafficcamnet tracker=on`.
- pyds container starts as `upstream-dst-2049fb2fac99` (tracker prefix).
- In-container log: `[NvMultiObjectTracker] Initialized` (IOU config
  loaded from `tracker_config.yml`).
- Bounded probe cycle (180s): **2128** frames seen, **2010** with
  detections, labels = `person × 2201, car × 1`, **7** unique tracker
  ids (`"0".."6"`).
- `tracker_id` stability across consecutive frames: `"2"` × 1155
  frames, `"5"` × 725, `"0"` × 370, `"6"` × 210, `"4"` × 106, `"3"` × 3,
  `"1"` × 1 (a tracker that wasn't doing real work would assign a new
  id per frame — every count would be 1).
- Status posts `starting → attached → processing → stopped`. Final
  status: `clean shutdown` at 10:19:13.
- Zero observations posted.
- Zero leftover containers (`docker ps --filter name=upstream-dst-`),
  zero leftover `/tmp/upstream-p3-tracker.*` dirs, secret file 600,
  no raw RTSP / session id / Authorization / Bearer / srt:// in logs.

### Out of scope (next milestones)
- `ObservationWriter` integration + `POST /observations` (P4).
- Final metadata adapter (P4).
- `nvdsanalytics`, line crossing, zones (P3.5 if event semantics are
  needed before observations).
- Dashboard overlay.
- NvDCF / NvSORT / NvDeepSORT tuning (future tracker-config work).

---

## 2026-05-19 - P2 cleanup: metadata worker shutdown hardening

### Summary
Pure reliability fix for the supervisor-managed metadata probe worker
shipped in `dcc61c3`. The live verification exposed three shutdown
issues; none changed the metadata extraction path or the API contract,
all are addressed here. **No feature behaviour change. No new package,
no new env vars, no new observations, no nvtracker, no API change.**

### Root cause analysis
1. `request_stop()` only set `self._stop`. The metadata consumer is
   parked in `asyncio.to_thread(queue.Queue.get)` and only wakes when
   the bridge queue is closed. The reader closed the queue in its
   `finally`, but only after its next 0.5 s poll iteration saw the
   stop flag — i.e. up to 0.5 s of consumer parking on every stop.
2. `_safe_stop_pm()` and the terminal status post were AFTER the inner
   `asyncio.gather`. If the worker task was cancelled (supervisor
   force-stop after the per-worker budget), the gather raised
   CancelledError and the cleanup tail was skipped — leaving docker
   containers + `/tmp/upstream-p2-meta.*` workdirs behind.
3. `DeepStreamMetadataProcessManager.stop()` was not idempotent. The
   outer `run()` `except CancelledError` handler calls `_safe_stop_pm`
   again, which retried `docker stop` on an already-removed container.

### What changed
- `app/runtimes/deepstream_runtime/worker.py`:
  - New instance attribute `_metadata_queue: BoundedBridgeQueue | None`,
    set only while `_drive_metadata_probe` is running.
  - `request_stop()` overridden to set the stop event AND
    **synchronously** call `self._metadata_queue.close()` if a queue is
    active. Closing the queue pushes its close-sentinel, which wakes any
    consumer parked in `asyncio.to_thread(queue.Queue.get)` within
    milliseconds.
  - `_drive_metadata_probe()` now wraps the inner `gather()` in a
    `try / finally`, and the finally `await asyncio.shield(...)` a new
    `_finalize_metadata_probe()` helper that closes the queue,
    `_safe_stop_pm()`s the PM, and posts the terminal status. Shielding
    means docker stop + workdir cleanup + the final status post all
    complete even when the outer task is cancelled.
  - `_metadata_queue` is cleared in the outermost `finally` so subsequent
    `request_stop` calls are no-ops.
- `app/runtimes/deepstream_runtime/metadata_probe.py`:
  - `DeepStreamMetadataProcessManager` gained a `_stopped` guard so a
    second `stop()` call short-circuits. The guarded path still calls
    `_cleanup_work_dir()` defensively (in case the first call was
    cancelled before reaching its `finally`).
  - `_cleanup_work_dir()` nulls `_work_dir` after `shutil.rmtree`, so it
    is fully idempotent regardless of how many times it's called.
- `docs/local-dev.md` — P2 metadata-probe live-verification checklist:
  - **PID discovery fix**: capture the real python PID via
    `pgrep -f "/tmp/upstream-live-p2/venv/bin/python -m app\.main" -n`
    instead of `$!` (which records the nohup wrapper that exits
    promptly after `exec`). The first live run SIGINT'd a dead wrapper
    PID, which made it look like SIGINT was being ignored.
  - Documented the new SIGINT→wait→SIGTERM-backstop loop and the
    operator-side cleanup verification.
- `tests/test_deepstream_metadata_probe.py` — 4 new tests:
  - `test_request_stop_closes_queue_synchronously`: snapshots
    `_metadata_queue.stats().closed` immediately after `request_stop()`,
    proving the close is synchronous, AND verifies the PM observed the
    queue already-closed when `stop()` was called (cleanup ordering).
  - `test_request_stop_wakes_consumer_with_no_more_metadata`: parked
    consumer wakes via close, worker reaches `stopped` even with zero
    metadata frames in flight.
  - `test_pm_stop_is_idempotent`: a second `stop()` does not re-issue
    `docker stop` and leaves the workdir cleaned.
  - `test_pm_keep_temp_files_preserves_workdir`: `keep_temp_files=True`
    intentionally retains the dir + path attribute for operator inspection.

### Not changed
NO change to the decode-probe (M8.2B) or inference-probe (M8.2C1-B)
workers. NO new package, NO new env vars, NO API contract change. NO
ObservationWriter, NO `POST /observations`, NO nvtracker, NO
nvdsanalytics, NO line crossing, NO zones, NO dashboard overlay, NO
metadata adapter. The pyds container's in-container app is unchanged;
the manual P2 pyds probe in `scripts/` is unchanged.

### Testing/checks
`uv run pytest -q` **130 passed** (126 prior + 4 new). `ruff check`,
`ruff format --check`, `mypy app`, `bash -n scripts/*.sh` all clean.

### Follow-up TODOs
- **P3 — Tracker Metadata Probe**: add `nvtracker` to the pipeline so
  `NeutralObjectMetadata.tracker_id` is populated and `has_tracker_id`
  lights up. Required before any line crossing / dwell-time observation
  in P4.

---

## 2026-05-19 - P2: supervisor-managed metadata probe worker

### Summary
Connects the manual P2 pyds metadata pipeline to the existing
`WorkerManager` supervisor lifecycle via a new package
`deepstream.metadata_probe.v0` (`supports.metadata_probe: true`).
`DeepStreamWorker` mode dispatch is now a 3-way (metadata > inference >
decode); the metadata mode spawns the pyds-derived DS 8.0 container,
parses structured `UPSTREAM_METADATA=<json>` / `UPSTREAM_METADATA_SUMMARY=<json>`
stdout lines, validates each frame line with `NeutralFrameMetadata`, and
exercises the real app-side `BoundedBridgeQueue` /
`BridgeQueueMetadataSource` seam from P2 (seam milestone). Status
payload surfaces counters + queue stats. **The container never calls
UpStream APIs** — the webhook secret never enters the container.
**Implementation complete + unit-tested; live verification pending.** NO
observations, NO API contract change.

### What changed
- `app/packages/package_registry.json` — `deepstream.metadata_probe.v0`
  (`supports.metadata_probe`); mock + decode + inference entries unchanged.
- `app/config.py` — new env-overridable settings:
  `deepstream_pyds_image` (default `upstream-deepstream:8.0-pyds-v1.2.2`),
  `deepstream_metadata_probe_timeout_seconds` (60),
  `deepstream_metadata_queue_maxsize` (256),
  `deepstream_metadata_queue_drop_policy` (`drop_newest`),
  `deepstream_metadata_sample_limit` (3).
- `app/runtimes/deepstream_runtime/pyds_metadata_app.py` (new) — the
  **in-container** pyds Python app, bind-mounted into the pyds-derived
  DS 8.0 container; never imported by other `app/` code. Builds the
  proven RTSP -> nvstreammux -> nvinfer -> fakesink pipeline, attaches a
  pad probe to `nvinfer.src`, walks NvDsBatchMeta/NvDsFrameMeta/
  NvDsObjectMeta, emits one `UPSTREAM_METADATA=<json>` per frame +
  terminal `UPSTREAM_METADATA_SUMMARY=<json>`. Near-zero-work pad
  probe (build dict + `sys.stdout.write` + flush; no HTTP / async /
  `ObservationWriter`). The raw RTSP URL is consumed via `RTSP_URL` env
  and redacted in any GST_ERROR output. mypy/ruff skip this module via
  `[[tool.mypy.overrides]]` because gi/pyds only resolve at runtime.
- `app/runtimes/deepstream_runtime/metadata_probe.py` (new) — pure
  `build_metadata_docker_command` + `DeepStreamMetadataProcessManager`.
  Owns a per-run temp work dir (PGIE config + copy of
  `pyds_metadata_app.py`), bind-mounted into the container at the same
  path; runs `docker run --rm --gpus all --network host -v wd:wd -e
  PYTHONDONTWRITEBYTECODE=1 -e RTSP_URL=... bash -lc 'timeout -s INT N
  python3 -B wd/pyds_metadata_app.py …'`. Same docker-stop -> SIGTERM ->
  SIGKILL-after-timeout teardown as the inference PM; shreds the temp
  dir on stop unless `keep_temp_files`.
- `app/runtimes/deepstream_runtime/worker.py` — module-level
  `_METADATA_MODE` + pure `parse_metadata_line` (returns
  `("frame"|"summary", "<json>")` or `None`) + `_MetadataCounters` /
  `_MetadataReaderState` dataclasses + `_queue_stats_dict`. `_mode`
  property now orders metadata > inference > decode. `_default_pm_factory`
  branches to `DeepStreamMetadataProcessManager`. New
  `_drive_metadata_probe` runs reader + consumer asyncio tasks
  concurrently: reader validates via `NeutralFrameMetadata` and pushes
  to `BoundedBridgeQueue.put_from_thread`; consumer iterates
  `BridgeQueueMetadataSource` and feeds counters; posts `attached` on
  first valid frame, `processing` immediately + throttled, terminal
  `stopped` (clean exit + no hard fail) or `failed` (with redacted
  `error_message` + `deepstream_unsupported_sm` /
  `deepstream_trt_error` / `deepstream_exit`).
- `tests/test_deepstream_metadata_probe.py` (new, 13 tests) plus
  additions to `tests/test_config.py` (2 tests) and
  `tests/test_package_resolver.py` (1 test). Covers mode selection
  (and decode/inference regression guard), pure parser routing, pure
  command builder (`--gpus all` / `--network host` / `-v wd:wd` /
  `PYTHONDONTWRITEBYTECODE` / `RTSP_URL` / inner argv), full lifecycle
  with counters + `queue_stats` + `app_summary`, `Unsupported SM` ->
  `failed` with RTSP redaction, nonzero exit without metadata,
  malformed-JSON / schema-invalid lines being skipped, missing RTSP
  fast-fail, graceful `request_stop`, and the real PM's temp-dir write
  + cleanup (monkeypatched `create_subprocess_exec`). NO Docker / GPU /
  pyds / gi in any test.
- `pyproject.toml` — added `[[tool.mypy.overrides]]` for the
  in-container app module (gi/pyds only resolve inside the container).
- Docs: `README.md` (new P2 scope + env table rows), `.env.example`
  (new env vars), `docs/architecture.md` (P2 supervisor-managed
  metadata-probe section), `docs/local-dev.md` (live verification
  checklist), `app/runtimes/deepstream_runtime/README.md` (now describes
  three modes), this entry.

### Not changed
NO `app/`-side `ObservationWriter` integration. NO `POST /observations`.
NO final metadata adapter. NO `nvtracker` (and thus no stable tracker
IDs — `tracker_id` is `None` for now, preserved as a forward-compatible
field). NO `nvdsanalytics`, NO line crossing, NO zone analytics, NO
dashboard overlay, NO rules / events, NO Triton, NO package catalog UI,
NO API contract change. upstream-agent / api-server / streaming-server /
dashboard untouched. The existing decode (M8.2B) and inference
(M8.2C1-B) workers are unchanged; mock runtime unchanged. The webhook
secret never enters the pyds container.

### Testing/checks
`uv run pytest -q` **126 passed** (111 prior + 15 new); `ruff check`,
`ruff format --check`, `mypy app`, `bash -n scripts/*.sh` all clean.

### Follow-up TODOs
- Live verification of the supervisor-managed metadata worker on
  crowdcountS2 (see `docs/local-dev.md` P2 checklist).
- After live-verified: either P3 (tracker metadata probe — add
  `nvtracker` so `tracker_id` is populated and `has_tracker_id=True`),
  or P4 (observation adapter — map `NeutralFrameMetadata` ->
  `ObservationPayload` and wire the `ObservationWriter`).

---

## 2026-05-19 - P2: MetadataSource Protocol + bounded bridge queue (app/ seam)

### Summary
Promotes the neutral metadata shape proven by the manual pyds probe into
typed `app/`-side abstractions: pydantic models, a thread-safe bounded
queue with explicit drop policy, and an async-iterable `MetadataSource`
Protocol. Pure code, no GPU/Docker/live-stream test required. No pyds, gi,
or Gst imports in `app/` or tests. **NO observations, NO API writes, NO
runtime-worker pyds integration yet** — those land in the next milestones.

### What changed
- `app/observations/metadata_models.py` (new) — pydantic v2 models
  (`extra="forbid"`): `BoundingBoxPixels`, `BoundingBoxNormalized`,
  `NeutralObjectMetadata` (class_id / label? / confidence? / bbox_pixels?
  / bbox_normalized? / tracker_id?: str), `NeutralFrame` (NonNegativeInt
  frame_num / width / height / optional timestamp), `NeutralFrameMetadata`
  (runtime / metadata_source / optional stream_session_id / frame /
  objects[] / source + debug dicts). Helpers: `bbox_normalized_from_pixels`
  (same rounding as the manual probe) and `tracker_id_from_pyds`
  (`UNTRACKED_OBJECT_ID -> None`, else stringify uint64). The models accept
  the EXACT dict the manual probe emitted live (asserted by test).
- `app/observations/bridge_queue.py` (new) — `BoundedBridgeQueue` +
  `DropPolicy` (DROP_NEWEST / DROP_OLDEST) + `BridgeQueueStats`. Producer
  side `put_from_thread` is thread-safe and non-blocking (returns bool);
  consumer side `get` is awaitable, returns `NeutralFrameMetadata | None`.
  Internal `queue.Queue` (unbounded slot; maxsize gated under
  `threading.Lock` so DROP_OLDEST can evict atomically) + `asyncio.to_thread`
  for the async path + a close sentinel to wake waiting consumers.
  Designed for a single async consumer (out of scope: fan-out).
- `app/runtimes/metadata_source.py` (new) — `MetadataSource` Protocol
  (`@runtime_checkable`, `__aiter__ -> AsyncIterator[NeutralFrameMetadata]`)
  + `BridgeQueueMetadataSource` (canonical impl wrapping the queue).
- `tests/test_metadata_models.py` (new, 12 tests), `tests/test_bridge_queue.py`
  (new, 13 tests), `tests/test_metadata_source.py` (new, 6 tests): pydantic
  shape/extra/blank-runtime validation, JSON round-trip, both drop
  policies, close-wakes-on-empty, async-wakes-on-late-put, real
  threading.Thread producer + slow asyncio consumer with bounded memory
  and FIFO order, Protocol conformance.
- Docs: architecture.md (P2 `app/` seam section), local-dev.md (developer
  pointer), this entry.

### Not changed
NO `app/` runtime worker pyds integration. NO pad probe in `app/`. NO
ObservationWriter integration. NO `POST /observations`. NO nvtracker, no
nvdsanalytics, no line crossing, no zone analytics, no dashboard, no
rules, no Triton, no API contract change. The DeepStream worker
(decode_probe + inference_probe) is unchanged. The manual P2 pyds probe in
`scripts/` is unchanged (it stays runtime-independent of `app/`).

### Testing/checks
`uv run pytest -q` **111 passed** (79 prior + 32 new). `ruff check`,
`ruff format --check`, `mypy app`, `bash -n scripts/*.sh` — all clean. No
Docker/GPU/pyds/gi imports anywhere in tests.

### Follow-up TODOs
- **P2 — supervisor-managed metadata probe worker**: use the proven manual
  pyds probe path PLUS this MetadataSource/BridgeQueue seam under the real
  processor supervisor lifecycle. Still no observations, no API writes.
- Later: P3 (package / config generator) and P4 (observation adapter).

---

## 2026-05-19 - P2: manual DeepStream pyds metadata probe (live-verified)

### Summary
Out-of-tree, manual-only probe proving the DeepStream metadata seam: pyds
imports under `--gpus all`, a pad probe on `nvinfer.src` walks
`pyds.gst_buffer_get_nvds_batch_meta` → `NvDsFrameMeta` → `NvDsObjectMeta`
and pulls `class_id` / `obj_label` / `confidence` / `rect_params` /
`object_id`. Live against the active stream the probe produced
**671 frames / 673 `person` detections** with bbox + confidence + (correct)
`tracker_id=null`; neutral metadata sample matches the P2 spec exactly;
clean bounded exit (`docker_exit=124`), no leftover container, no raw RTSP
key or secret in any logged/committed file.

DeepStream 8.0 ships no pyds; the probe stands on a thin
`upstream-deepstream:8.0-pyds-v1.2.2` image built from
`docker/Dockerfile.pyds` (FROM `deepstream:8.0-samples-multiarch` +
`libpython3.12t64` + NVIDIA `pyds-1.2.2-cp312-cp312-linux_x86_64.whl`).
Build is reproducible; the in-build `import pyds` check was removed because
`libcuda.so.1` is only injected at run time by the NVIDIA Container Toolkit
(`--gpus all`) — the probe verifies the import naturally on first run.

### What changed
- `docker/Dockerfile.pyds` (new) — pinned base + wheel URL, SHELL `bash -lc`,
  apt: `python3-pip`, `libpython3.12t64`, `curl`, `ca-certificates`; curl
  purged after install; LABEL with pyds version + base.
- `scripts/_pyds_probe_helpers.py` (new) — pure helpers: `redact_rtsp`,
  `scrub_rtsp_inline`, `bbox_normalized`, `build_neutral_object_sample`,
  `UNTRACKED_OBJECT_ID`. No gi/pyds imports → testable on the Mac.
- `scripts/deepstream_pyds_metadata_probe.py` (new) — Python probe: builds
  `rtspsrc(tcp) → rtph264depay → h264parse → nvv4l2decoder → nvstreammux →
  nvinfer → nvvideoconvert → fakesink`; pad probe on `nvinfer.src`;
  near-zero work; bounded by `--seconds`; emits one
  `PYDS_PROBE_SUMMARY=<json>` line; SIGINT/SIGTERM-clean.
- `scripts/probe_deepstream_pyds_metadata.sh` (new) — driver: builds the
  pyds image on demand (`CONFIRM_BUILD=1`), generates a per-run temp
  `pgie.txt` (TrafficCamNet INT8 — same as M8.2C1-A/B), `docker run --rm
  --gpus all --network host -v wd:wd ...`, parses summary, exit-classifies,
  shreds the temp dir. RTSP redacted in every log line; `KEEP_TEMP_FILES=1`
  to retain.
- `tests/test_pyds_probe_helpers.py` (new, 9 tests) — redact / scrub /
  normalize / neutral-sample shape / JSON roundtrip / `UNTRACKED_OBJECT_ID`.
- Docs updated: `docs/architecture.md` (P2 section), `docs/deepstream-
  inference-smoke.md` (P2 pointer), `docs/deepstream-rtsp-probe.md`
  (forward pointer), README + local-dev (run-it pointer).

### Not changed
**No `app/` code touched.** No MetadataSource, no metadata adapter, no
ObservationWriter integration, no observations, no pyds in `app/`, no pad
probes in `app/`, no nvtracker, no nvdsanalytics, no Triton, no line
crossing, no zones, no dashboard, no rules. The runtime supervisor / worker
behavior is unchanged. API contracts unchanged. upstream-agent / api-server
/ streaming-server / dashboard untouched. The webhook secret never left the
API host (this probe doesn't call any UpStream API).

### Testing/checks
`uv run pytest -q` **79 passed** (70 prior + 9 new). `ruff check`,
`ruff format --check`, `mypy app`, `bash -n scripts/*.sh` — all clean.
Live: SSM eligible-sessions HTTP 200 / `validation.passed=true`;
`ffprobe` exit 0 (h264 1280×720 30/1); pyds 1.2.2 image built in ~15 s;
probe run produced `frames_seen=671 / frames_with_objects=671 /
objects_seen=673 / class_counts={"2":673} / label_counts={"person":673}`,
3 neutral-sample frames matching the P2 spec.

### Follow-up TODOs
- P2 (continued) — `MetadataSource` Protocol + bounded bridge queue in
  `app/`, consumed by an eventual ObservationWriter seam. Still no
  observations.
- Later: metadata adapter, tracking, nvtracker, nvdsanalytics, etc.

---

## 2026-05-19 - P1/M8.2C1-B: supervisor-managed DeepStream inference worker

### Summary
The M8.2C1-A-proven DeepStream 8.0 nvinfer pipeline now runs under the
existing WorkerManager as the `deepstream.inference_probe.v0` package. New
`DeepStreamInferenceProcessManager` owns a per-run temp work dir
(deepstream-app + PGIE config + KITTI dir, bind-mounted into the DS8
container) and runs the bounded `deepstream-app` RTSP→nvstreammux→nvinfer
(TrafficCamNet INT8)→fakesink pipeline. `DeepStreamWorker` gained a
mode (from the package's `supports`): `decode_probe` (M8.2B, unchanged) or
`inference_probe`. Status `starting→attached→processing→stopped`, or
`failed` with a redacted message + `deepstream_unsupported_sm`/
`deepstream_trt_error`/`deepstream_exit` code. Detection evidence
(KITTI frame/detection counts + engine-built/PERF log markers) is
status/log-only — **no observations, no pyds, no metadata extraction**.
Default DeepStream image bumped 7.1 → 8.0 (7.1 fails inference on Blackwell).
Implementation complete + unit-tested; **live verification pending**.

### Files touched
- `app/runtimes/deepstream_runtime/inference.py` (new) — pure builders
  (`build_pgie_config`, `build_app_config`, `build_inference_docker_command`)
  + `DeepStreamInferenceProcessManager` (temp dir, `kitti_stats()`, cleanup
  unless `keep_temp_files`, redacted logging).
- `app/runtimes/deepstream_runtime/worker.py` — `_mode`, mode-aware
  `_default_pm_factory`, `run()` dispatch, `_drive_decode()` (verbatim move,
  unchanged behaviour) + `_drive_inference()`, `_infer_payload`,
  `_kitti_stats`; `_post_best_effort` gained `payload`.
- `app/packages/package_registry.json` — `deepstream.inference_probe.v0`
  (`supports.inference_probe`); mock + decode entries unchanged.
- `app/config.py` — `deepstream_image` default → 8.0; new
  `deepstream_inference_timeout_seconds` (60),
  `deepstream_inference_output_dir` (""), `deepstream_inference_model`
  ("trafficcamnet"), `deepstream_keep_temp_files` (False).
- `tests/test_deepstream_inference.py` (new, 14 tests) +
  `tests/test_config.py` / `tests/test_package_resolver.py` additions.
- README / .env.example / docs/architecture / docs/local-dev (M8.2C1-B live
  checklist) / docs/deepstream-inference-smoke / docs/deepstream-rtsp-probe /
  `app/runtimes/deepstream_runtime/README.md`.

### Not changed
No API contract change. No pyds, pad probes, MetadataSource, metadata
adapter, observations, tracking, nvtracker, nvdsanalytics, line crossing,
zones, Triton, dashboard, package catalog, billing. Mock runtime and M8.2B
decode behaviour unchanged (decode loop moved verbatim into `_drive_decode`).
upstream-agent / api-server / streaming-server / dashboard untouched.

### Testing/checks
`uv run pytest -q` **70 passed** (53 prior + 17 new); `ruff check`,
`ruff format --check`, `mypy app`, `bash -n scripts/*.sh` all clean. No
Docker/GPU in the suite (scripted `ProcessManager` double feeds the exact
M8.2C1-A log lines; real PM temp-dir cleanup tested with a monkeypatched
subprocess).

### Follow-up TODOs
- Live verification on crowdcountS2 with an active stream
  (`UPSTREAM_FORCE_PACKAGE_ID=deepstream.inference_probe.v0`,
  `UPSTREAM_DEEPSTREAM_IMAGE=...8.0-samples-multiarch`).
- Later milestones: pyds / NvDsBatchMeta / metadata adapter / observations.

---

## 2026-05-19 - P1/M8.2C1-A: DeepStream 8.0 inference smoke PASSED on Blackwell

### Summary
Retried the M8.2C1-A inference probe on `deepstream:8.0-samples-multiarch`
(DeepStream 8.0.0, **TensorRT 10.9.0.34**, **CUDA 12.8.1**). The Blackwell
sm_120 block is **gone**: nvinfer built a TensorRT engine on the RTX 5070 Ti
(`serialize cuda engine to file: ...successfully`, ~30 s cold), ran at ~30
fps on the live RTSP stream, and TrafficCamNet produced **1571 detections
across 1728 KITTI frames** (class `person`). Clean SIGINT/timeout exit
(`App run successful`), no `Unsupported SM`, no core dump. **M8.2C1-A is
passed; next is M8.2C1-B.** Decode worker (M8.2B) unaffected. Still manual
probe only — no worker change, pyds, metadata, or observations.

### What changed
- `scripts/probe_deepstream_inference.sh` — default image bumped 7.1 → **8.0**
  (7.1 proven non-functional for inference on Blackwell); DEEPSTREAM_IMAGE
  still overridable. **Classifier fixed**: the cold-start
  `deserialize ... failed, try rebuild` line is the normal no-prebuilt-engine
  path and is no longer treated as failure; engine-build success is detected
  via `serialize cuda engine to file: ...successfully` (nvinfer persists the
  engine next to the ONNX, not at our model-engine-file path, so file
  presence is not a reliable signal); added an `[INCONCLUSIVE-DET]` exit-0
  branch for engine-built-but-zero-detections (scene/model mismatch).
- `docs/deepstream-inference-smoke.md` — verdict now PASS on 8.0; 7.1 kept as
  root-cause history; combined 7.1/8.0 evidence table; outcome → M8.2C1-B.
- `docs/commit-log.md`, `docs/deepstream-rtsp-probe.md` — updated.

### Not changed
No inference runtime worker, pyds, NvDsBatchMeta, metadata adapter,
observations, Triton, tracking, dashboard. App behaviour untouched. Probe is
read-only w.r.t. UpStream; processor secret sourced server-side (SSM/API
host), never printed/committed; no raw stream key committed.

### Testing/checks
`bash -n scripts/*.sh` clean; `uv run pytest -q` 53/53; ruff/format/mypy
unaffected (no Python changed). Live on crowdcountS2: SSM eligible-sessions
HTTP 200 COUNT=1; `ffprobe` exit 0; DS 8.0 manual run + fixed script run both
built the engine and produced detections; no leftover containers; temp dirs
removed/shredded.

### Follow-up TODOs
- M8.2C1-B: supervisor-managed inference worker, DeepStream image pinned to
  8.0 (TRT ≥ 10.7 / CUDA ≥ 12.8). Metadata/observations remain later
  milestones.

---

## 2026-05-19 - P1/M8.2C1-A: DeepStream nvinfer inference smoke probe (blocked: Blackwell)

### Summary
Manual-only probe to answer "can a DeepStream container run nvinfer on the
live RTSP stream on the RTX 5070 Ti?" Answer on `deepstream:7.1-samples-multiarch`:
**no.** nvinfer initialized but TensorRT 10.3 cannot build an engine for
Blackwell — `IBuilder::buildSerializedNetwork: Error Code 1: Internal Error
(Unsupported SM: 0xc00)` (compute 12.0 = sm_120), then core dump. Decode
(M8.2A/B) works on the same image because NVDEC bypasses TensorRT. Active
validated stream confirmed (h264 1280x720@30, `validation.passed=true`);
`ffprobe` clean. No runtime/worker change, no observations.

### What changed
- `docs/deepstream-inference-smoke.md` — new. Verdict, evidence table,
  TensorRT-vs-Blackwell explanation, recommended next image (DeepStream 8.x /
  TensorRT ≥ 10.7 / CUDA ≥ 12.8), how-to.
- `scripts/probe_deepstream_inference.sh` — new. RTSP→nvstreammux→nvinfer
  (bundled TrafficCamNet INT8)→fakesink via `deepstream-app`, no display;
  forces a fresh engine build (the SM-arch test); KITTI dump = detection
  evidence (no pyds). Temp config in a shredded `mktemp` dir, RTSP redacted,
  `CONFIRM_PULL=1` gate, exits non-zero on the SM block / any inference fail.

### Not changed
No inference runtime worker, pyds, NvDsBatchMeta, metadata adapter,
observations, Triton, tracking. App behaviour untouched. Probe is read-only
w.r.t. UpStream; processor secret sourced server-side (SSM/API host), never
printed/committed; no raw stream key committed.

### Testing/checks
`bash -n scripts/*.sh` clean; `uv run pytest -q` 53/53; ruff/mypy unaffected
(no Python changed). Live: SSM eligible-sessions HTTP 200 COUNT=1; crowdcountS2
`ffprobe` exit 0; deepstream-app reproduced `Unsupported SM: 0xc00`.

### Follow-up TODOs
- Re-run `scripts/probe_deepstream_inference.sh` with
  `DEEPSTREAM_IMAGE=nvcr.io/nvidia/deepstream:8.x-samples-multiarch` against
  an active stream. Only if it passes → M8.2C1-B inference worker.

---

## 2026-05-19 - P1/M8.2B cleanup: DeepStream shutdown defaults

### Summary
M8.2B live verification needed env overrides for clean DeepStream shutdown
(the M8.1/mock defaults were too tight for a Docker-backed worker). Made the
deployment-safe values first-class so future inference tests don't rely on
ad hoc overrides.

### What changed
- `app/config.py`: `final_status_timeout_seconds` 3.0→**15.0**;
  `overall_shutdown_timeout_seconds` 8.0→**25.0**;
  `deepstream_process_stop_timeout_seconds` 10.0→**5.0**.
  These are global/shared; documented as DeepStream-safe and still fine for
  mock (upper bounds, not delays). Env overrides unchanged.
- `.env.example`: added the three vars with a "shutdown budget" comment.
- README / architecture / local-dev / deepstream-rtsp-probe: noted the new
  defaults.
- `tests/test_config.py`: assert the new defaults + env-override still works.

### Not changed
No inference, no observations, no API-contract change, no runtime
architecture change. Mock + DeepStream tests use `fast_settings` overrides
and are unaffected. Deployment/readiness cleanup before M8.2C1 inference
smoke test.

### Testing/checks
`uv run pytest -q`, `ruff check`, `ruff format --check`, `mypy app` — all green.

---

## 2026-05-19 - P1/M8.2B: Minimal DeepStream Runtime Worker

### Summary
Turns the M8.2A manual decode probe into a supervisor-managed runtime worker.
New `deepstream.decode_probe.v0` package (runtime `deepstream`, decode only,
no observations). `DeepStreamWorker` runs the proven Docker/GStreamer NVDEC
pipeline as a managed subprocess and maps it onto the M8.0 status lifecycle
(`starting → attached → processing → stopped`/`failed`). Resolver now
dispatches by the package's runtime; `UPSTREAM_FORCE_PACKAGE_ID` routes a
mock-defaulted session to the DeepStream package without touching prod
config. RTSP URL is never logged (redacted); secret never handled here.
**No inference, no Triton, no tracking, no nvdsanalytics, no observations.**

### Files touched
- `app/packages/package_registry.json` — added `deepstream.decode_probe.v0`.
- `app/packages/resolver.py` — `resolve(package_id)` (drop runtime arg);
  `SUPPORTED_RUNTIMES={mock,deepstream}`; `PackageSpec.supports`.
- `app/config.py` — `force_package_id` + 5 `deepstream_*` settings.
- `app/supervisor/processor_status.py` — optional `runtime`/`package_id`
  overrides (mock unchanged; DeepStream reports its resolved runtime).
- `app/redact.py` (new) — `redact_rtsp` / `scrub_rtsp_in_text`.
- `app/runtimes/deepstream_runtime/{__init__,process_manager,worker}.py`
  (new) — `ProcessManager` protocol, `DeepStreamProcessManager`,
  `DeepStreamWorker`; `README.md` updated.
- `app/supervisor/worker_manager.py` — `effective_package_id` +
  `build_worker` runtime dispatch; `_Handle.worker: Runtime`.
- `tests/` — rewrote `test_package_resolver`; fixed `test_mock_runtime` /
  `test_worker_manager`; new `test_deepstream_runtime` (fake PM, no Docker).
- `docs/` — README, architecture, local-dev (M8.2B checklist),
  deepstream-rtsp-probe, this log.

### System interaction notes
`SessionPoller → WorkerManager (effective_package_id → resolve →
build_worker) → DeepStreamWorker → DeepStreamProcessManager (docker run
--gpus all --network host …) → fpsdisplaysink lines → POST /status`.
No observation writes.

### Testing/checks
- `uv run pytest -q` — 51/51 pass (was 39; +12).
- `uv run ruff check .` / `ruff format --check .` — clean.
- `uv run mypy app` — clean (28 files).
- Live DeepStream worker run = operator checklist (`docs/local-dev.md`).

### Follow-up TODOs
- Live-verify the worker against an active stream on crowdcountS2.
- Later: first inference probe (re-evaluate DeepStream tag for RTX 5070 Ti
  / Blackwell sm_120 — decode is fine, TensorRT is the open question).

---

## 2026-05-18 - M8.1: Mock Processor Handoff (initial repo)

### Summary
First commit of `upstream-processor`. A pure-asyncio supervisor that polls the
M8.0 `/v1/internal/processor/eligible-sessions` endpoint, runs one mock worker
per eligible session, posts the `starting → attached → processing → stopped`
status lifecycle (or `failed`), and emits the three `processor.mock_*`
observation types. Proves the processor handoff contract before any DeepStream
/ video work. **No video decode, no RTSP, no DeepStream** — `internal_rtsp_url`
is parsed but never read (AST-enforced).

### Files touched
- `app/config.py` — `Settings` (env_prefix `UPSTREAM_`, `SecretStr`).
- `app/errors.py` — typed `ProcessorError`/`ApiError` hierarchy.
- `app/logging_setup.py` — secret-redaction logging filter.
- `app/backoff.py` / `app/clock.py` — interruptible backoff; tz-aware time.
- `app/api_client/{base,upstream_api}.py` — `ProcessorApi` Protocol + async
  typed client (auth header isolated, no body/secret logging).
- `app/observations/{schemas,writer}.py` — contract mirror + observation
  chokepoint.
- `app/packages/{package_registry.json,resolver.py}` — single `mock.detect.v0`
  package; strict resolver.
- `app/runtimes/base.py`, `app/runtimes/mock_runtime/{worker,fake_observations}.py`
  — runtime seam + mock worker.
- `app/supervisor/{session_poller,worker_manager,processor_status}.py` —
  reconcile loop + status builder.
- `app/main.py` — entrypoint + signal handling + exit-code contract.
- `tests/*` — 39 tests; `docs/*`; `pyproject.toml`, `Makefile`, `.env.example`.

### Functions/classes changed
- `WorkerManager.run/_reap/_start_new/_stop_removed/_shutdown_all` — the
  single-writer reconcile + bounded shutdown.
- `MockWorker.run` — the lifecycle state machine; 404→quiet, 422→failed,
  transient→absorbed.
- `UpstreamApiClient` — failure→typed-error taxonomy.

### System interaction notes
`main → SessionPoller.fetch_eligible (GET) → WorkerManager reconcile →
MockWorker (POST /status, POST /observations)`. Auth via
`X-Upstream-Webhook-Secret`; tenant always server-derived from the session.

### Testing/checks
- `uv run pytest -q` — 39/39 pass.
- `uv run ruff check .` / `ruff format --check .` — clean.
- `uv run mypy app` — clean (strict; app code, matching api-server's
  `mypy src` convention).
- Boot smoke: start → transient-survive loop on unreachable API → SIGINT →
  exit 0, secret absent from logs.
- Read-only prod smoke: live `GET /eligible-sessions` parses into our schema.

### Follow-up TODOs
- M8.2: `runtimes/deepstream_runtime/` behind the same supervisor — first
  consumer of `internal_rtsp_url`; introduces real `lagging`.
