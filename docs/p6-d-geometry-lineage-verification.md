# P6-D — End-to-End Geometry Lineage Verification

> Repeatable proof that operator-saved geometry flowed all the way from the
> dashboard editor through to emitted observations. Run this after any
> change that touches the geometry path (geometry editor, api-server
> wire-carry, processor resolver, nvdsanalytics translation, or observation
> emission).

## What this verifies

Five witnesses, three of them from the api-server (no shell access to the
processor host needed) and two from the live runtime:

| Witness | Source | Confirms |
|---|---|---|
| `payload.applied_geometry` on processor status | `GET /v1/cameras/{id}/processor-status/latest` (P6-D) | the worker resolved operator geometry and is running it |
| Geometry lineage on observations | `GET /v1/cameras/{id}/observations/recent?type=…` | every emitted line_crossing / roi_occupancy / zone_entered / dwell_started row carries `payload.source.geometry_source` + `geometry_revision` |
| Saved slugs in observation payloads | same | `payload.line_id` ∈ saved lines, `payload.roi_id` ∈ saved ROIs |
| Embedded geometry in perception snapshot | `GET /v1/cameras/{id}/perception/latest` | the dashboard overlay matches the runtime |
| nvdsanalytics INI on disk *(optional)* | `--analytics-config-path /tmp/.../analytics_config.txt` | the runtime config carries saved slugs and the probe-default ROI did not leak |

## Prerequisites

- Dashboard access (to draw geometry).
- A camera with an active streaming session you can walk in front of.
- SSH to the processor host (for the optional INI witness and to tail the
  resolver log during the test).
- A caller-authed Bearer token exported as `UPSTREAM_API_TOKEN` (NOT the
  processor webhook secret — that's a different auth tier).

## Step-by-step

### 1. Save geometry in the dashboard

Open the geometry editor:

```
/dashboard-next/cameras/<camera-id>/configure/geometry
```

Draw **at least 2 lines and 1 ROI**. Give them distinct slugs (the dashboard
auto-generates `line-1`, `line-2`, `roi-1`, etc.). Save. The sidebar will
show the new revision number — **write it down** (you'll pass it to the
verify script).

### 2. Restart the processor session

The processor only resolves geometry at session start. Bounce the agent (or
restart the streaming session) so the eligible-session envelope is re-served
to the processor. The processor's next poll will spawn a fresh
`DeepStreamWorker`.

### 3. Confirm the resolver picked up the operator save

SSH the processor host and tail the log:

```
ssh crowdcounts2@crowdcountS2 \
  'tail -n 0 -F /tmp/upstream-live-p6c/processor.log' \
  | grep --line-buffered "geometry resolved"
```

Wait for a line shaped like:

```
geometry resolved: camera_id=<uuid> stream_session_id=<uuid> \
  source=operator_saved revision=<N> line_count=<L> roi_count=<R>
```

`<N>` must match the revision you noted in step 1. `<L>` and `<R>` must
match the counts you drew. **Do not start the walk-through until this line
appears for the new session.**

### 4. Walk the test pattern in front of the camera

Pacing matters — observations are debounced; do each step deliberately.

1. **Line 1** — Stand clearly on one side of saved line 1. Walk *perpendicular*
   across it. Pause 2 s. Walk back across it. Pause 2 s.
   - If your line is vertical, that means walking horizontally. Walking
     parallel to the line will not trigger `line_crossing` events — the
     #1 cause of P6-D PARTIAL verdicts.
2. **Line 2** — Same, for saved line 2.
3. **ROI** — Walk into the saved ROI. **Stand still inside it for 10–15
   seconds.** This is long enough to fire `roi_occupancy`, `zone_entered`,
   and `dwell_started`. Then walk out.

### 5. Run the verification script

From any machine (your laptop is fine — the script reads the api-server):

```bash
export UPSTREAM_API_TOKEN=<your bearer token>

cd ~/Documents/Upstream/upstream-processor
uv run python scripts/verify_geometry_lineage.py \
  --camera-id <camera-id> \
  --expected-revision <N> \
  --expected-lines line-1,line-2 \
  --expected-rois roi-1 \
  --since <ISO-8601 just before your walk> \
  --stream-session-id <stream session uuid>
```

Optional: pass `--analytics-config-path /tmp/upstream-p35-analytics.*/analytics_config.txt`
(the live INI) for the runtime-config witness. The path on the server changes
every cycle — copy it from the processor log or `ls -t /tmp/upstream-p35-analytics.*/`.

### 6. Read the verdict

The script prints a per-witness report and exits with a machine-checkable code:

| Exit | Verdict | Meaning |
|---|---|---|
| 0 | **PASS** | every witness confirms operator-saved lineage end-to-end |
| 1 | **PARTIAL** | processor + ROI lineage pass, but `line_crossing` rows are absent (or some expected line slugs weren't crossed). **Re-walk** perpendicular and re-run. |
| 2 | **FAIL** | lineage is wrong somewhere — `geometry_source = default`, revision mismatch, unknown slug, or probe-default leaked into the runtime INI. Treat as a P6-C regression. |
| 3 | **BLOCKED** | can't tell — processor has never reported (404), api-server returned a 4xx/5xx, or `UPSTREAM_API_TOKEN` was missing. |

## Common failure modes

### PARTIAL — "no line_crossing rows in window"

Most common. The operator walked parallel to a vertical saved line. Walk
perpendicular and re-run; usually clears immediately. If multiple line slugs
are expected but only some show up, the script names the missing ones.

### FAIL — `geometry_source = default` on an observation

The resolver did not pick up operator geometry. Either the dashboard save
hadn't been persisted by the time the worker started, or the resolver
crashed silently and fell back. Check the processor log for `geometry
resolved: source=…` and for any `geometry_invalid` `error_code` on the
processor-status row.

### FAIL — `error_code = geometry_invalid` on processor status

The processor parsed the operator save but rejected it (wrong coordinate
space, all-disabled, malformed envelope). Per P6-C, the worker degrades to
the probe default and surfaces this error. Re-open the geometry editor and
re-save; if it persists, the rejection reason is in the `error_message`
column.

### BLOCKED — `processor_status_not_found`

The processor never reported for this camera. Either no session is active,
or the session is active but the processor isn't running, or you passed a
`--stream-session-id` that doesn't match any row. Start the processor and
re-run.

### BLOCKED — `UPSTREAM_API_TOKEN` not set

You're trying to use the processor webhook secret as a Bearer token; those
are different auth tiers. Mint a dashboard API token from the dashboard's
token UI or via `POST /v1/api-tokens` and export it.

## CI integration

The script exits with the verdict code, so a CI job can run it after a
canned synthetic walk and treat anything other than `0` as a regression.
Don't gate on `PARTIAL` (1) in CI — line-crossing depends on physical motion
and is brittle. Gate on `FAIL` (2) and `BLOCKED` (3) only.

## Out of scope

P6-D verifies *that the lineage is wired correctly*. It does NOT verify:

- **Strict direction** — `direction: "in"|"out"` is preserved in lineage but
  nvdsanalytics runs `mode=loose`. Counts fire on any crossing regardless of
  direction.
- **Mid-session reload** — a dashboard save during an active session is
  picked up only on the next session restart.
- **Replay / shadow testing** — comparing the same video against two
  geometry revisions side-by-side.
- **Reliability findings** — alerting when `geometry_source = default`
  appears in production.

All of those are P6-D follow-ups, tracked in the processor commit log.
