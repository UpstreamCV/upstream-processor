# P5-E — NvDCF Tracker Profile Expansion & IOU-vs-NvDCF A/B

**Date:** 2026-06-15
**Status:** `experimental` — NOT promoted to `probe`. This document records what
was built, what was tested, why, and how, plus the gate that remains before
promotion.

---

## 1. Why this milestone exists

P5-D live testing (2026-06-11, `crowdcountS2`) showed strong YOLOX-S person
detection but **IOU track-ID fragmentation** during walking motion. The four
per-track analytics events depend on a single track surviving across frames, so
fragmentation suppressed them — they were never observed live despite good
detection:

- `processor.zone_exited`
- `processor.line_crossing`
- `processor.line_crossing_object`
- `processor.dwell_exceeded`

The codebase's own blocker analysis (`docs/architecture.md`) named **tracker
profile expansion (NvDCF)** as "the smallest credible fix." P5-E adds that
tracker and the machinery to A/B it against IOU.

## 2. What was built (see `docs/commit-log.md` for the full change list)

- **NvDCF tracker profile** — promoted the pre-existing
  `deepstream_nvdcf_balanced_v1` stub from `planned`/empty to `experimental`
  with an authored, **ReID-free** DS 8.0 NvDCF *perf*-style ll-config
  (`app/packages/tracker_registry.json`).
- **NvDCF analytics package** — `deepstream.yolox_analytics_nvdcf_probe.v0`,
  identical to the IOU `deepstream.yolox_analytics_probe.v0` except for the
  tracker profile (`app/packages/package_registry.json`). Sits **beside** the
  IOU package for A/B; `experimental`, `production_ready: false`.
- **Package-driven tracker selection** — the worker now derives the running
  tracker (algorithm + ll-lib / width / height) from the resolved package's
  `tracker_profile_id` (mirroring the package-driven model), instead of the
  global `UPSTREAM_DEEPSTREAM_TRACKER_ALGORITHM` setting (now a fallback only).
  `build_tracker_config` was already generic, so NvDCF needed **no
  rendering-code change**. A package that names a non-runnable tracker **fails
  closed** at process start (never silently falls back to IOU).
- **A/B tooling** — `scripts/_tracker_ab_helpers.py` (pure, tested),
  `scripts/compare_tracker_ab.py` (CLI), `scripts/probe_tracker_ab.sh`
  (real-worker live A/B) and `scripts/probe_tracker_ab_clip.sh` (hands-off
  recorded-clip A/B).

## 3. What we tested, and why

The unit tests (751 passing) verify the **wiring** — package resolves, config
renders, the worker selects NvDCF, lineage/status report it, it fails closed on
a bad profile. They **cannot** verify the only thing the milestone actually
claims, because it is a GPU + scene property, not a code property:

1. Does NvDCF even **run** on the Blackwell (sm_120) / DS 8.0 stack? (Never
   executed there before — only IOU and the detector were proven.)
2. Are the **authored tunables** loadable / does it track at all?
3. Does it **reduce fragmentation** vs IOU?
4. What is the **FPS / VRAM** cost?

We chose a **recorded-clip A/B** rather than staging a live person because (a) a
fair A/B needs *identical input* on both passes, which sequential live runs
cannot give, and (b) it requires nobody in front of the camera. This run targets
questions **#1–#3 (gross)**; it deliberately does **not** exercise the
zone/line/dwell events (no analytics geometry in this path).

## 4. How we ran it

`scripts/probe_tracker_ab_clip.sh` drives DeepStream's stock `deepstream-app`
against the bundled sample clip, twice, with **our** rendered tracker configs:

- Input: `/opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h264.mp4`
  (1,442 frames; a street scene of cars/pedestrians), identical for both passes.
- Detector: the DS sample primary-gie (TrafficCamNet); `[tracker]` enabled with
  `ll-config-file` = our `build_tracker_config("iou")` then
  `build_tracker_config("nvdcf")`, `ll-lib-file` = the in-image
  `libnvds_nvmultiobjecttracker.so`, `tracker-width/height = 960×544`.
- Per-frame track IDs dumped via `kitti-track-output-dir`, parsed by
  `kitti_dir_to_snapshots` and scored by `compare_tracker_ab.py`.
- Host: `crowdcountS2`, **GPU 1** (the near-idle GPU; the live processor on GPU 0
  was not disturbed), image `upstream-deepstream:8.0-pyds-v1.2.2`.

Reproduce (render configs on a dev box, copy with the helper + CLI to the host,
then):

```sh
KEEP_TEMP_FILES=1 GPU_DEVICE=1 PROBE_TIMEOUT=240 bash probe_tracker_ab_clip.sh
```

## 5. Results

Both passes processed all **1,443** frames, no crash, no fail-closed, valid
KITTI output — **NvDCF runs end-to-end on the Blackwell GPU with our config**
(question #1 ✓, #2 ✓).

| metric | IOU | NvDCF |
|---|---|---|
| distinct track IDs (whole clip) | 59 | 309 |
| avg objects/frame | 25.4 | 18.9 |
| longest single-track persistence (frames) | 1442 | 763 |
| switch-proxy (lower = less fragmentation) | 29 | 281 |

FPS was not meaningfully captured: a file source completes far faster than
real time, so the 5s perf interval reported `0`. (FPS/VRAM under a live stream
remains to be measured.)

## 6. Interpretation and caveats

On this clip NvDCF allocated **~5× more track IDs** than IOU — superficially
"more fragmented." **Do not read this as "NvDCF is worse" for the real use
case.** Three confounders:

1. **Wrong content.** The sample is small, distant, fast, overlapping **cars** —
   not a close, large, slow **walking person**. NvDCF's visual correlation
   filter behaves very differently on the two regimes.
2. **Unverified tunables.** The high ID churn indicates our authored
   `minTrackerConfidence` / `probationAge` / `maxShadowTrackingAge` are too
   aggressive (terminate + reacquire with fresh IDs). These must be reconciled
   against the in-container DS 8.0 `config_tracker_NvDCF_perf.yml` and tuned.
3. **Unequal detector gate.** The byte-locked IOU profile uses
   `minDetectorConfidence: 0` (keeps everything); the NvDCF perf profile uses
   `0.1894` + tracker-confidence filtering — so they are not tracking the same
   detection set. IOU's "persistence = the entire 1442-frame clip" is itself a
   **red flag for IOU** (one ID likely riding/merging across many cars by
   overlap), not a quality win.

**Net:** NvDCF viability on the production GPU is **proven**; fragmentation
reduction for the person/zone use case is **neither proven nor disproven** by
this clip.

## 7. What this did NOT cover

- Zone/line/dwell **event recall** (this path has no analytics/geometry).
- The **person** use case (sample clip is car traffic).
- FPS/VRAM under a **live** stream.
- The two deliberate design policies that independently cap
  `zone_exited` / `dwell_exceeded` regardless of tracker — "track disappearance
  ≠ zone exit" (P4-D) and "disappearance ≠ close dwell session" (P4-E).

## 8. Gate before promoting `experimental → probe`

Promote `deepstream_nvdcf_balanced_v1` AND
`deepstream.yolox_analytics_nvdcf_probe.v0` only after, on `crowdcountS2`:

- stable processor startup; YOLOX detection preserved;
- **measurably better track continuity than IOU on representative content**
  (a clip / live scene with a person walking) — after reconciling + tuning the
  NvDCF ll-config;
- `line_crossing` + `line_crossing_object` fire; `zone_exited` reachable;
  `dwell_exceeded` from a real walker (or the design-policy limitation
  documented);
- no observation flood; acceptable FPS/VRAM; no RTSP/secret leakage.

Never `supported` without a separate license + deployment review
(`production_ready` stays `false`).

## 9. Reproduction artifacts

- Live A/B runbook + smoke checklist + promotion criteria:
  `docs/local-dev.md` → "P5-E — IOU vs NvDCF tracker A/B".
- Clip A/B driver: `scripts/probe_tracker_ab_clip.sh`.
- Pure aggregation + KITTI parsing (unit-tested):
  `scripts/_tracker_ab_helpers.py`, `tests/test_tracker_ab_probe.py`.
- Comparison CLI: `scripts/compare_tracker_ab.py`.
