# Commit log

Reverse chronological. Newest at top.

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
