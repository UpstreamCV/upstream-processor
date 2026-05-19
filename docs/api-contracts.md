# API contracts consumed (M8.0)

Base: `${UPSTREAM_API_BASE_URL}/v1/internal/processor`
Auth: header `X-Upstream-Webhook-Secret: <UPSTREAM_PROCESSOR_WEBHOOK_SECRET>`
(no `Authorization`/Bearer; never logged). Pinned against api-server commit
`7a6057a`. Our pydantic models in `app/observations/schemas.py` are the single
mirror of this contract.

## `GET /eligible-sessions?limit=<int>`

200 → **object envelope** (not a bare array):

```json
{ "sessions": [ {
  "session_id": "<uuid>", "stream_key": "sess_...", "camera_id": "<uuid>",
  "account_id": "<uuid>", "internal_rtsp_url": "rtsp://...",
  "package_id": "mock.detect.v0", "runtime": "mock",
  "approved": {"codec","width","height","fps","bitrate_kbps"},
  "validation": {"passed","observed","expected","mismatch_fields","created_at"},
  "zones_lines": null, "started_at": "<iso|null>", "processor_eligible": true
} ] }
```

`internal_rtsp_url` is parsed but **never used** in M8.1.

## `POST /status`

Request (no `account_id`/`camera_id` — the server derives tenant from the
session):

```json
{ "processor_id": "...", "stream_session_id": "<uuid>", "runtime": "mock",
  "package_id": "mock.detect.v0",
  "status": "starting|attached|processing|lagging|failed|stopped",
  "fps_input": 30.0, "fps_processed": 30.0,
  "last_frame_at": "<iso>", "last_observation_at": "<iso>",
  "error_code": null, "error_message": null, "payload": {} }
```
`None` fields are omitted (`exclude_none`). 200 → status row echo
(`id,account_id,camera_id,...,created_at,updated_at`).

## `POST /observations`

```json
{ "stream_session_id": "<uuid>",
  "type": "processor.mock_frame_seen|processor.mock_heartbeat|processor.mock_detection",
  "payload": { ... }, "observed_at": "<iso, tz-aware>",
  "processor_id": "...", "package_id": "mock.detect.v0" }
```
200 → `{ "id": "<uuid>", "count": 1 }`.

## Errors (envelope `{"error":{"code","message","details","request_id"}}`)

| HTTP | code | client handling |
|---|---|---|
| 401 | `invalid_webhook_secret` | **fatal** — log once, exit 2 |
| 404 | `session_not_found` | session gone — stop that worker quietly, no terminal post |
| 422 | `validation_error` | contract bug — log loud, no retry; worker posts `failed`; on the GET it is fatal (exit 3) |
| 429 / 5xx / timeout / unparsable-2xx | — | transient — backoff and retry, loop survives |
