# TEAM ROLES

**Checklist item 4.** Fill in the names, paste this into your group chat, pin it.

| Role | Name | Owns |
|---|---|---|
| **A — Modelling** | Dip | Framing, losses, training, ablations, evaluation |
| **B — Data** | _______ | Earth Engine, masks, auxiliary channels, the cache |
| **C — Decision layer & writing** | _______ | Assets at risk, Naria case study, figures, the paper |

**Dip takes A.** The contribution of this project is the loss design and the change-prediction reframe, and that is where your ML background is decisive. Do not let yourself get pulled into Earth Engine debugging on Day 1 — it is the most seductive way to lose the critical path.

---

## The split in one line each

- **A** turns arrays into results.
- **B** turns satellites into arrays.
- **C** turns results into a paper that wins.

They meet at exactly two interfaces: `DATA_CONTRACT.md` (B→A) and the results CSV (A→C). Everything else is independent, which is what lets three people work at 18 hours a day without colliding.

---

## Why this split and not another

| Alternative | Why not |
|---|---|
| Everyone does a bit of everything | Three people debugging the same Earth Engine auth error. The classic failure. |
| Split by river (one person per reach) | Triples the pipeline work and produces three incompatible codebases. |
| Split by day | Nobody owns anything end to end, so nobody notices when it breaks. |
| A does data too, B does modelling | Puts the person with the ML expertise on the non-critical path. |

**The split above is by *artefact*, not by task.** Each person owns files nobody else writes to. Merge conflicts approach zero.

---

## Day by day

### A — Modelling (the critical path)

| Day | Work | Hands off |
|---|---|---|
| 1 | Build `dataset.py` against **B's stub** (contract §9) — do not wait for real data. Then `metrics.py` (M1–M9) and persistence B0. Unit-test the delta invariant. U-Net skeleton. | Persistence numbers on every metric → C |
| 2 | **The whole bet.** B3 JamUNet-style baseline (confirm M1 < 1.0). A1 delta framing. Losses. A2 α-sweep. | Results table v1 → C |
| 3 | Wire in B's aux channels (A5). OpenSTL B4/B5 — *90-minute hard stop*. A3, A4, A7. Bankline displacement M5. | — |
| 4 | 5-seed ensemble, calibration. A8 spatial holdout, A9 cross-river. C6 bridge break. **Touch test set once at h16, then freeze.** | Final numbers + `P(erosion)` rasters → C |
| 5 | Support C on the Naria hindcast. Stop modelling. | — |
| 6–7 | Write Methods and Results. Red-team on Day 7. | — |

**A's non-negotiables:** never select a checkpoint on overall accuracy; never use a random split; touch the test set once.

### B — Data

| Day | Work | Hands off |
|---|---|---|
| 1 | **h0: register for Earth Engine — before anything else.** Then: write the stub (10 min, unblocks A immediately), `reaches.yaml`, smoke test, export P1+P2, `build_stack.py`, **contact sheet and inspect every year**. | **Cache + `meta.json` by h6** → A |
| 2 | `build_aux.py`: distance-to-bank, curvature, erosion history. Hunt for Hardinge Bridge / Bahadurabad discharge. | — |
| 3 | Finish aux incl. `dist_bridge_km`. Deliver aux stack. | Aux stack by h5 → A |
| 4 | Own masks 2022–2025 (MNDWI + local Otsu). **Validate against GSW on 2015–2021 overlap.** Export J1 — one command. | Extended cache → A |
| 5 | Geospatial support for C's asset overlay: reprojection, union boundaries, raster–vector joins. | — |
| 6–7 | Write the Study Area and Data sections. | — |

**B's non-negotiables:** the stub ships in the first ten minutes; never bypass the grid-alignment check in `build_stack.py`; never interpolate a missing year.

### C — Decision layer and writing

| Day | Work | Hands off |
|---|---|---|
| 1 | Read JamUNet, *From Pixels to People*, Ritu et al. 2023. Draft the abstract (playbook §18). Pull the ACAPS and Start Fund Naria briefing notes. | Abstract draft → all |
| 2 | Acquire Open Buildings v3 **and 2.5D Temporal** (confirmed available), OSM Bangladesh extract, union boundaries. Build the Naria ground-truth table: 2,000+ houses, 500 businesses, 4,200–5,000 homeless, six named unions. | — |
| 3 | `assets_at_risk.py` skeleton against A's stub predictions. Figure templates, colour-blind-safe palette. | — |
| 4 | Related Work section written. Figure pipeline ready so Day 5 is assembly, not authoring. | — |
| 5 | **Naria hindcast and the union watchlist.** All figures. **Submit the abstract at h14–16.** | Submitted abstract |
| 6–7 | Write Intro, Related Work, Discussion, Limitations, Conclusion. Build the deck — lead with Naria, not the method. | Final paper |

**C's non-negotiables:** the abstract is submitted on Day 5, not Day 7; no claim goes in the abstract that A has not verified.

---

## If you are only two people

Drop C. Reassign:

- **A** takes modelling, evaluation, **and the Naria hindcast** (it is analysis, and it is the winning result).
- **B** takes data, the asset layer, figures, **and the paper**.

Cut, in this order: OpenSTL (B4/B5), the horizon experiment (A7), the 2022–2025 mask extension. **Never cut:** A1, A2, the Naria hindcast, or the abstract deadline.

---

## Sync protocol

**Daily standup, 20 minutes, end of each day.** Three questions only:

1. What landed in the repo today?
2. What is blocking me tomorrow?
3. Has anything changed in `DATA_CONTRACT.md`?

**Two hard checkpoints:**

- **Day 2, hour 18 — the go/no-go.** Has the delta framing improved M1 over the state framing? If yes, everything after is elaboration. If no, you have five days to change course rather than one. Diagnose that night; do not push on and hope.
- **Day 4, hour 18 — the freeze.** Modelling stops. The test set has been touched once. Numbers are final. Days 5–7 are writing and the decision layer. Nobody trains anything after this without all three agreeing.

---

## Rules that prevent the predictable disasters

1. **Nobody edits another person's files.** Open a PR or ask. A single merge conflict in `dataset.py` at hour 90 costs more than the feature.
2. **`DATA_CONTRACT.md` changes require A and B to agree in writing, with the version bumped.** Silent shape changes are the most expensive bug available to you.
3. **Every result lands in `results/` as CSV, with the config name and seed.** If it is not in a CSV it does not exist and cannot go in the paper.
4. **Fix and record every seed.** With ~28 runs and three people, unreproducible numbers cost a day of confusion on Day 6.
5. **Checkpoint to Drive every epoch.** Colab will disconnect. Assume it.
6. **Nobody works past hour 18.** Seven consecutive 18-hour days is already at the limit. Errors made at hour 19 get found at hour 100.

---

## The three things that decide the outcome

1. **B ships the stub in the first ten minutes**, so A is never blocked by Earth Engine.
2. **A writes `metrics.py` and persistence before any model.** If you cannot measure the failure, you cannot show you fixed it.
3. **C submits the abstract on Day 5.** Not Day 7.
