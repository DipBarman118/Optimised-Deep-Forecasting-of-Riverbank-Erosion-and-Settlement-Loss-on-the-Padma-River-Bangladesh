# DATA CONTRACT v1.0

**Checklist item 5.** Commit this before anyone writes code. It is the interface between **B (data)** and **A (modelling)**, and it exists so the two of you can work in parallel on Day 1 without ever having seen each other's files.

**The rule:** B guarantees these arrays exist with exactly these shapes, dtypes and conventions. A writes code assuming they do. Neither of you asks the other what a file looks like — you both read this.

**Changing the contract requires both A and B to agree, in writing, in this file, with the version bumped.** A silent change is the single most expensive thing that can happen to a three-person team on a seven-day deadline.

---

## 1. Files

All paths relative to `cache_dir` from `configs/reaches.yaml`. `{R}` is a reach ID (`P1`, `P2`, `J1`). `{y0}`, `{y1}` are the first and last year.

| File | dtype | Shape | Contents |
|---|---|---|---|
| `{R}_waterclass_{y0}_{y1}.npy` | `uint8` | `[T, H, W]` | Raw GSW class: `0` no data, `1` not water, `2` seasonal, `3` permanent |
| `{R}_water_{y0}_{y1}.npy` | `uint8` | `[T, H, W]` | **`1` = water, `0` = land** |
| `{R}_valid.npy` | `uint8` | `[H, W]` | `1` = pixel has data in every year |
| `{R}_aux_{y0}_{y1}.npy` | `float32` | `[T, C, H, W]` | Auxiliary channels, order fixed in `{R}_meta.json` |
| `{R}_meta.json` | JSON | — | The authoritative description of everything above |
| `{R}_grid.json` | JSON | — | Written by `gee_export.py`; the intended export grid |

---

## 2. Conventions — non-negotiable

These are the ones that cause silent, hard-to-find bugs. Read them twice.

1. **Water is 1. Land is 0.** Never the reverse, in any array, at any stage.
2. **Axis order is `[T, H, W]`.** Time first. Auxiliaries are `[T, C, H, W]` — channel second, matching PyTorch `NCHW` after indexing a timestep.
3. **Row 0 is north.** The affine transform has negative `e`. If you `flipud` anything, the river flows the wrong way.
4. **Year `t` lives at index `meta["years"].index(t)`.** Never assume index 0 is 1988. Missing years shift everything.
5. **No sentinel values inside arrays.** No-data is expressed only through `{R}_valid.npy`. Do not put `255` or `-1` in the water mask.
6. **All reaches share one CRS** (`EPSG:32645`). Required for the cross-river transfer test (A9) to mean anything.
7. **Every year of a reach shares one pixel grid.** `build_stack.py` enforces this and exits non-zero if violated. Do not bypass the check.

---

## 3. Delta encoding — the core of the project

A computes this from `water`; B does not produce it. Both must agree on it.

```python
# delta[t] describes the transition from year t to year t+1
# Shape [T-1, H, W], dtype uint8

0 = stable      water[t] == water[t+1]
1 = erosion     water[t] == 0  and  water[t+1] == 1     land became water
2 = accretion   water[t] == 1  and  water[t+1] == 0     water became land
```

**Invariant that must be unit-tested on Day 1, hours 6–9:**

```python
recon = water[t].copy()
recon[delta[t] == 1] = 1
recon[delta[t] == 2] = 0
assert (recon == water[t + 1]).all()
```

If this fails, everything downstream is meaningless. Write the test before the model.

---

## 4. `{R}_meta.json` schema

```json
{
  "contract_version": "1.0",
  "reach_id": "P1",
  "name": "Padma proper — Aricha/Goalundo to Chandpur",
  "river": "Padma",
  "crs": "EPSG:32645",
  "transform": [xmin, 30.0, 0.0, ymax, 0.0, -30.0],
  "shape": [34, 1400, 4100],
  "years": [1988, 1989, "...", 2021],
  "missing_years": [],
  "water_classes": [2, 3],
  "water_convention": "1 = water, 0 = land",
  "delta_encoding": {"0": "stable", "1": "erosion", "2": "accretion"},
  "row0": "north",
  "rho_mean_annual_change": 0.0214,
  "valid_fraction": 0.981,
  "split": {"train": [1988, 2012], "val": [2013, 2016], "test": [2017, 2021]},
  "tile": 256,
  "stride": 128,
  "aux_channels": ["dist_to_bank", "curvature", "erosion_hist_3y",
                   "discharge_norm", "dist_bridge_km", "gsw_occurrence"],
  "git_commit": "a1b2c3d",
  "created_utc": "2026-08-01T09:14:22+00:00"
}
```

`transform` is in **GDAL order** (`rasterio`'s `.to_gdal()`): `[xmin, pixel_width, 0, ymax, 0, -pixel_height]`. Note this differs from `rasterio`'s native `Affine` ordering — convert explicitly, never by hand.

---

## 5. Auxiliary channels

> **Memory correction — this changed after measuring the real grids.**
>
> Measured reach sizes are larger than first estimated, because a diagonal river inside a rectangle is mostly empty floodplain:
>
> | Reach | Grid | Per year | 34-year `uint8` stack |
> |---|---|---|---|
> | P1 | 3816 × 3223 | 12.3 Mpx | **418 MB** |
> | P2 | 5802 × 3030 | 17.6 Mpx | **598 MB** |
> | J1 | 2472 × 5586 | 13.8 Mpx | **470 MB** |
>
> P1 + P2 ≈ **1.0 GB** as `uint8`. That fits Colab's ~12.7 GB RAM comfortably.
>
> **But a materialised `[T, C, H, W] float32` aux stack does not.** Six float32 channels over P1 alone is ~10 GB and will kill the session. So the aux design below splits into *stored static* and *computed dynamic*. Do not write a full 4-D aux array.

### 5a. Stored — static channels only

File `{R}_auxstatic.npy`, dtype **`float16`**, shape `[C_static, H, W]`. Order fixed by `meta["aux_static"]`.

| Channel | Units | Range | Notes |
|---|---|---|---|
| `gsw_occurrence` | % | 0–100 | Long-run GSW occurrence. Time-invariant |
| `dist_bridge_km` | km | 0–200 | Distance to Padma Bridge river training works. **C6 only** |

P1 cost: 2 × 12.3 Mpx × 2 bytes ≈ **49 MB**. Acceptable.

### 5b. Computed on the fly — dynamic channels

Derived by **A** inside the dataset, **per 256 × 256 tile**, from `water[t]`. Never precomputed reach-wide. A distance transform on a 256 × 256 tile is sub-millisecond; the full-reach version costs 10 GB of RAM you do not have.

| Channel | Units | Derivation |
|---|---|---|
| `dist_to_bank` | pixels, signed | `scipy.ndimage.distance_transform_edt` on the tile. Negative inside water, positive on land |
| `curvature` | 1/pixel | Local channel-centreline curvature. Outer banks of bends erode |
| `erosion_hist_3y` | count 0–3 | From `water[t-3:t]` slices already in RAM |

### 5c. Per-year scalars

File `{R}_scalars.json` — `{"discharge_norm": {"1988": 0.62, ...}}`. Broadcast to a plane at tile-construction time. Storing a constant plane per year is 34 × 12.3 Mpx of identical values, which is pure waste.

**If a channel cannot be produced, B records it in `meta["aux_missing"]` and A substitutes zeros.** B never silently drops a channel — that changes `C_static` and breaks A's model input shape at load time, on Day 3, with no explanation.

---

## 6. Splitting

**Temporal only. Never random.** Randomly splitting year-windows leaks future river states into training.

| Split | Years | Use |
|---|---|---|
| train | 1988–2012 | Fitting |
| val | 2013–2016 | Checkpoint selection, hyperparameters |
| test | 2017–2021 | **Touch once**, Day 4 hour 16 |
| extended | 2022–2025 | B's own masks. Also the C6 window |

A sample is the window `(t-3, t-2, t-1, t) -> delta[t]`. **The window is assigned to the split containing year `t+1`** — the year being predicted. Getting this backwards leaks one year of test data into training.

Windows spanning a `missing_years` gap are **dropped**, never interpolated.

---

## 7. Tiling

- Tile `256 × 256`, stride `128` (50% overlap)
- A tile is kept only if `valid` covers ≥ 95% of it
- Tile positions are computed **once** from `[H, W]` and shared across all years, so a tile index refers to the same ground location in every year
- Tiles are **not** pre-materialised to disk. The full stack is ~139 MB and lives in RAM; slice on the fly

---

## 8. Who writes what

| File | Owner | Consumer | Due |
|---|---|---|---|
| `configs/reaches.yaml` | **B** | everyone | Day 1 h1 |
| `{R}_grid.json` | **B** (`gee_export.py`) | B | Day 1 h3 |
| `{R}_waterclass`, `{R}_water`, `{R}_valid`, `{R}_meta.json` | **B** (`build_stack.py`) | **A** | **Day 1 h6** |
| `{R}_aux` | **B** (`build_aux.py`) | **A** | Day 3 h5 |
| `src/dataset.py` (delta, tiling, splits) | **A** | A | Day 1 h9 |
| `src/metrics.py` | **A** | A, C | Day 1 h12 |
| Open Buildings / OSM / union boundaries | **C** | C | Day 2 |

---

## 9. The stub that unblocks A immediately

**B writes this in the first ten minutes, before any real export.** It lets A build and test the entire dataset, metrics and training pipeline against synthetic data while B is still waiting on Earth Engine. A is never blocked.

```python
# data/make_stub.py — synthetic reach with a migrating channel
import numpy as np, json, os

T, H, W = 34, 400, 1200
cache = "data/cache"
os.makedirs(cache, exist_ok=True)
rng = np.random.default_rng(0)

water = np.zeros((T, H, W), np.uint8)
for t in range(T):
    for x in range(W):
        centre = H/2 + 60*np.sin(x/180) + 1.6*t      # channel migrates with t
        half   = 34 + 9*np.sin(x/90 + t/5)
        lo, hi = int(centre-half), int(centre+half)
        water[t, max(0,lo):min(H,hi), x] = 1
water |= (rng.random((T, H, W)) < 0.0015)            # speckle

np.save(f"{cache}/STUB_water_1988_2021.npy", water)
np.save(f"{cache}/STUB_waterclass_1988_2021.npy", (water*3 + (1-water)).astype(np.uint8))
np.save(f"{cache}/STUB_valid.npy", np.ones((H, W), np.uint8))
json.dump({
    "contract_version": "1.0", "reach_id": "STUB", "name": "synthetic",
    "river": "stub", "crs": "EPSG:32645",
    "transform": [700000.0, 30.0, 0.0, 2600000.0, 0.0, -30.0],
    "shape": [T, H, W], "years": list(range(1988, 1988+T)), "missing_years": [],
    "water_classes": [2, 3], "water_convention": "1 = water, 0 = land",
    "delta_encoding": {"0":"stable","1":"erosion","2":"accretion"},
    "row0": "north", "valid_fraction": 1.0,
    "split": {"train":[1988,2012],"val":[2013,2016],"test":[2017,2021]},
    "tile": 256, "stride": 128, "aux_channels": [],
}, open(f"{cache}/STUB_meta.json","w"), indent=2)
print("stub written — A is unblocked")
```

Because the stub channel migrates at a known rate, A can sanity-check that the model learns *something* before real data arrives. **A model that cannot beat persistence on the stub has a bug, not a hard problem.**

---

## 10. Changelog

| Version | Date | Change | Agreed by |
|---|---|---|---|
| 1.0 | 1 Aug 2026 | Initial contract | A, B |
