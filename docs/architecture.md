# Architecture

Pure asyncio, single event loop, one `asyncio.Task` per session. No threads:
the M8.1 workload is all I/O (HTTP polls + timed posts).

## Components

### `main`
Composition root. Loads `Settings`, configures logging (with the secret
redaction filter), installs SIGINT/SIGTERM handlers (`loop.add_signal_handler`
→ set an `asyncio.Event`), builds the shared `httpx.AsyncClient` /
`UpstreamApiClient`, the `SessionPoller`, the `WorkerManager`, and runs the
reconcile loop. Exit codes: `0` clean, `2` fatal auth, `3` fatal contract,
`1` unexpected.

### `SessionPoller`
Thin wrapper over `GET /eligible-sessions`. Unwraps the `{"sessions":[...]}`
envelope into a list; lets typed errors propagate so the manager owns policy.

### `WorkerManager` (the reconcile loop)
Single-writer over a `dict[session_id → handle]` (asyncio is cooperative, so
no locks). Each tick: poll → reap finished tasks → start workers for newly
eligible sessions → stop workers for sessions that dropped out → sleep.
Guarantees: never two workers for one session; a reappearing session gets a
fresh worker; a crashed worker is reaped and respawned next tick unless it
crash-loops (`max_restarts` within `restart_window` → quarantine); an
unresolvable package is reported `failed` once and quarantined, never spawned.
`ApiAuthError` (401) and a GET `ApiValidationError` (422) bubble out as fatal;
everything else on the poll is transient and backed off.

### `MockWorker` (the runtime, `runtimes/mock_runtime`)
One per session. Lifecycle: `starting → attached → processing` (status pulse
on an interval) with three independent observation pulses
(`mock_frame_seen` / `mock_heartbeat` / `mock_detection`). Terminal: `stopped`
on graceful stop, `failed` on a contract/internal error, or **quiet exit with
no terminal post** on 404 `session_not_found` (the session no longer exists to
post against). Transient API errors are absorbed with per-worker backoff
inside the pulses. The worker **never reads `internal_rtsp_url`** (enforced by
an AST-level test).

### Shutdown
Signal → event set → reconcile loop exits → `_shutdown_all` requests stop on
every worker, then awaits each best-effort final `stopped` under
`final_status_timeout`, all under an overall `wait_for(overall_timeout)` with a
hard `task.cancel()` escape hatch so a dead API cannot hang exit.

## Future: the DeepStream runtime (M8.2+)

`runtimes/base.Runtime` is the seam. M8.2 adds `runtimes/deepstream_runtime/`
implementing the same `run()` / `request_stop()` / `state` / `session_id`
interface. Only then is `internal_rtsp_url` consumed (to open the MediaMTX
RTSP read path). The supervisor, poller, status/observation paths, schemas,
and resolver are unchanged — the runtime is the only new moving part. The
contract-legal `lagging` status is reserved for that real-pipeline runtime;
the mock never emits it.
