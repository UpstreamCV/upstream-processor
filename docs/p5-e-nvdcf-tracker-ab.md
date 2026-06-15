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
- Clip A/B driver: `scripts/probe_tracker_ab_clip.sh` (P5-F added `CLIP_HOST`
  to point it at a host-recorded clip).
- Pure aggregation + KITTI parsing (unit-tested):
  `scripts/_tracker_ab_helpers.py`, `tests/test_tracker_ab_probe.py`.
- Comparison CLI: `scripts/compare_tracker_ab.py`.

---

## P5-F update (2026-06-15) — config reconciliation + validation attempt

### A1. NvDCF config reconciled to the DS 8.0 perf sample
Read the in-container `/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_tracker_NvDCF_perf.yml` and transcribed it **verbatim** into `deepstream_nvdcf_balanced_v1.settings`. The P5-E authored settings were materially off; key corrections:

| key | P5-E (authored) | P5-F (DS 8.0 perf) |
|---|---|---|
| `BaseConfig.minDetectorConfidence` | 0.1894 | **0.043** |
| `TargetManagement.minIouDiff4NewTarget` | 0.3777 | **0.7418** |
| `TargetManagement.minTrackerConfidence` | 0.2079 | **0.4009** |
| `TargetManagement.probationAge` | 3 | **2** |
| `TargetManagement.maxShadowTrackingAge` | 30 | **51** |
| `DataAssociator.minMatchingScore4Overall` | 0.0 | **0.429** |
| `DataAssociator` (+tentative keys) | missing | **added** `tentativeDetectorConfidence`, `minMatchingScore4TentativeIou` |
| `StateEstimator.processNoiseVar*` | 2.0/1.0/0.1/4.0/16.0 | **1.511/1.3159/0.03/3.0283/8.1505** |
| `VisualTracker.visualTrackerType` | 1 (NvDCF_legacy) | **2 (NvDCF_VPI)** |
| `ReID` section | `{reidType:0}` placeholder | **removed** (perf sample is ReID-free) |

IOU unchanged (byte-identical golden still passes). NvDCF stays `experimental`.

### A2. Reconciled config re-run (sample clip, GPU 1) — viability re-confirmed
The reconciled config (incl. `visualTrackerType: 2`/VPI) **loads and runs** on the Blackwell GPU (1443 frames, no VPI error). On the (non-representative) car clip: IOU 60 distinct IDs / switch-proxy 30; reconciled NvDCF 410 / 368 — higher, but this is the expected effect of the sample's low `minDetectorConfidence` (0.043) keeping many flickering distant-car detections, and the crude switch-proxy is dominated by detection churn on dense scenes. Not a person-tracking verdict.

### A3. Representative person A/B — RUN (operator-supplied clip)
Live capture was initially blocked (no eligible upstream session; mediamtx ready-paths flapped 15→0; camera-code paths 400 on read — readable paths are dynamic `sess_…` keys). The operator then supplied a **45-second clip of people crossing the mid-frame** (`Test_vid_copy.mp4`), which was run through both packages' trackers via `probe_tracker_ab_clip.sh` (`CLIP_HOST` + `EXTRA_MOUNT`).

Detector choice mattered: the DS sample **TrafficCamNet** detected people only sparsely (~177 person-lines / 681 frames) — a traffic detector, not representative — so the verdict run used **YOLOX-S** (the package's real detector, conf 0.25), which detected people densely (peak 5–6 concurrent). Identical input both passes; tracker-fragmentation only (no analytics events in this path).

**Result (YOLOX detector, 681 frames / ~45s):**

| metric | IOU | reconciled NvDCF |
|---|---|---|
| distinct track IDs | **18** | **33** |
| peak concurrent people | 6 | 5 |
| longest track persistence (frames) | **323** | 245 |
| switch-proxy (lower = less fragmentation) | **12** | 28 |
| substantial tracks (≥15 frames) | 16 | 17 |
| FPS | fast | ~134 (avg 154) |

NvDCF produced **~83% more distinct track IDs** than IOU (33 vs 18) and shorter max persistence — i.e. ~16 short-lived spurious tracks that IOU's stickier IOU-association absorbs into fewer, longer tracks. NvDCF ran cleanly and fast (no perf/stability issue).

### A4. Decision — **Outcome B: stay `experimental` (do NOT promote)**
On representative people content the reconciled NvDCF (NVIDIA's own perf preset) did **not** improve track continuity over IOU — it **fragmented more**. So the P5-E hypothesis ("NvDCF reduces IOU fragmentation, which should unblock the weak per-track events") is **not supported** by this test. `deepstream_nvdcf_balanced_v1` and `deepstream.yolox_analytics_nvdcf_probe.v0` remain `experimental` / `production_ready: false`.

Caveats: one 45s clip; tracker-fragmentation only (the zone/line/dwell **events** were not exercised — that needs the analytics pipeline + geometry); "more IDs" is a fragmentation proxy (without ground truth it could partly reflect NvDCF refusing to merge distinct people that IOU over-merges into one 323-frame track). But the direction is clear and repeatable.

**Implication (Outcome C signal):** since a stickier tracker did not obviously help, the weak `zone_exited` / `line_crossing` / `dwell_exceeded` events are plausibly capped by the deliberate P4-D/P4-E "disappearance ≠ exit/close" lifecycle policies rather than by the tracker. Recommend a follow-up **P5-Gap — Analytics Lifecycle Policy Fixes** investigation before further tracker-swap effort; NvDCF would need people-specific tuning (the perf preset is generic) to be worth revisiting, or explore NvSORT.
