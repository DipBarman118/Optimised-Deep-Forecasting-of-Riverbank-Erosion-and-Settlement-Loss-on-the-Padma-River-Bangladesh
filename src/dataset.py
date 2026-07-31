#!/usr/bin/env python3
"""
src/dataset.py — windows, delta targets, tiling, temporal splits.

Day 1, hours 6-9. Implements DATA_CONTRACT.md sections 3, 6 and 7.

    python src/dataset.py --reach STUB          # runs the self-tests

The two things this file exists to get right:

  1. The DELTA INVARIANT. delta must reconstruct water[t+1] from water[t]
     exactly. If it does not, everything downstream is meaningless.

  2. THE SPLIT. A window is assigned to the split containing the year being
     PREDICTED (t+1), not the last input year (t). Getting this backwards
     leaks one year of test data into training and is undetectable later.

Both are asserted in self_test(). Run it before you trust anything.
"""

import argparse
import json
import os

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset
    HAS_TORCH = True
except ImportError:                                    # self-tests run without torch
    HAS_TORCH = False
    Dataset = object

STABLE, EROSION, ACCRETION = 0, 1, 2
CLASS_NAMES = ["stable", "erosion", "accretion"]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
class ReachData:
    """Holds one reach's arrays in RAM. See the memory note in DATA_CONTRACT §5."""

    def __init__(self, cache_dir, reach):
        with open(os.path.join(cache_dir, f"{reach}_meta.json")) as f:
            self.meta = json.load(f)
        self.reach = reach
        self.years = self.meta["years"]
        self.water = np.load(os.path.join(cache_dir, self.meta["files"]["water"]))
        self.valid = np.load(os.path.join(cache_dir, self.meta["files"]["valid"]))

        assert self.water.dtype == np.uint8, "contract §1: water must be uint8"
        assert self.water.ndim == 3, "contract §2: water must be [T, H, W]"
        assert self.water.max() <= 1, "contract §2: water must be binary, 1 = water"
        assert self.water.shape[0] == len(self.years), "T must match len(years)"
        assert self.valid.shape == self.water.shape[1:], "valid must be [H, W]"

        self.T, self.Hh, self.Ww = self.water.shape
        self.delta = compute_delta(self.water)

        # Static aux, if B has produced it (contract §5a)
        self.aux_static = None
        p = os.path.join(cache_dir, f"{reach}_auxstatic.npy")
        if os.path.exists(p):
            self.aux_static = np.load(p).astype(np.float32)

    def year_index(self, year):
        """Contract §2 rule 4 — never assume index 0 is 1988."""
        return self.years.index(year)

    def __repr__(self):
        return (f"<ReachData {self.reach} T={self.T} H={self.Hh} W={self.Ww} "
                f"years={self.years[0]}-{self.years[-1]}>")


# ---------------------------------------------------------------------------
# Delta — contract §3
# ---------------------------------------------------------------------------
def compute_delta(water):
    """
    water : uint8 [T, H, W], 1 = water
    return: uint8 [T-1, H, W]  0 stable · 1 erosion (land->water) · 2 accretion
    """
    a, b = water[:-1].astype(bool), water[1:].astype(bool)
    d = np.zeros_like(water[:-1], dtype=np.uint8)
    d[~a & b] = EROSION
    d[a & ~b] = ACCRETION
    return d


def apply_delta(water_t, delta_t):
    """Inverse of compute_delta for a single step. Must recover water[t+1]."""
    out = water_t.copy()
    out[delta_t == EROSION] = 1
    out[delta_t == ACCRETION] = 0
    return out


# ---------------------------------------------------------------------------
# Tiling — contract §7
# ---------------------------------------------------------------------------
def tile_origins(H, W, tile, stride, valid=None, min_valid=0.95):
    """
    Tile positions, computed ONCE from [H, W] and reused for every year, so a
    tile index refers to the same ground location in all years.
    """
    ys = list(range(0, max(H - tile, 0) + 1, stride))
    xs = list(range(0, max(W - tile, 0) + 1, stride))
    if ys and ys[-1] + tile < H:
        ys.append(H - tile)
    if xs and xs[-1] + tile < W:
        xs.append(W - tile)

    out = []
    for y in ys:
        for x in xs:
            if valid is not None:
                if valid[y:y + tile, x:x + tile].mean() < min_valid:
                    continue
            out.append((y, x))
    return out


# ---------------------------------------------------------------------------
# Windows and splits — contract §6
# ---------------------------------------------------------------------------
def window_indices(reach: ReachData, k, split, split_cfg=None, drop_gaps=True):
    """
    A window at index i uses input years [i-k+1 .. i] and predicts delta[i],
    i.e. the transition from years[i] to years[i+1].

    THE SPLIT IS DECIDED BY years[i+1] — the year being predicted.
    """
    cfg = split_cfg or reach.meta["split"]
    if split == "all":
        lo, hi = -np.inf, np.inf
    else:
        lo, hi = cfg[split]

    missing = set(reach.meta.get("missing_years", []))
    out = []
    for i in range(k - 1, reach.T - 1):
        target_year = reach.years[i + 1]
        if not (lo <= target_year <= hi):
            continue
        if drop_gaps:
            span = reach.years[i - k + 1: i + 2]
            # contract §6: never interpolate; drop windows spanning a gap
            if missing & set(range(span[0], span[-1] + 1)):
                continue
            if span[-1] - span[0] != len(span) - 1:
                continue
        out.append(i)
    return out


# ---------------------------------------------------------------------------
# On-the-fly dynamic aux — contract §5b
# ---------------------------------------------------------------------------
def signed_distance(mask, clip=40.0):
    """
    Signed distance to the bankline, in pixels. Negative inside water.
    Computed PER TILE — never reach-wide (that is the 10 GB mistake).
    """
    from scipy import ndimage
    m = mask.astype(bool)
    if m.all() or (~m).all():
        return np.zeros(m.shape, np.float32)
    d_out = ndimage.distance_transform_edt(~m)     # land: distance to water
    d_in = ndimage.distance_transform_edt(m)       # water: distance to land
    return (np.clip(d_out - d_in, -clip, clip) / clip).astype(np.float32)


def erosion_history(delta, i, n=3):
    """Times each pixel eroded in the previous n transitions, scaled to 0-1."""
    lo = max(0, i - n)
    if lo == i:
        return np.zeros(delta.shape[1:], np.float32)
    return ((delta[lo:i] == EROSION).sum(0) / n).astype(np.float32)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class RiverWindows(Dataset):
    """
    Returns
        x : float32 [k + C, tile, tile]   k water masks, then aux channels
        y : int64   [tile, tile]          delta class
        w : float32 [tile, tile]          validity weight (0 = ignore in loss)
    """

    def __init__(self, reach: ReachData, k=4, split="train",
                 tile=None, stride=None, aux=("dist", "hist"), augment=False):
        self.r = reach
        self.k = k
        self.split = split
        self.aux = tuple(aux)
        self.augment = augment
        self.tile = tile or reach.meta["tile"]
        self.stride = stride or reach.meta["stride"]

        self.windows = window_indices(reach, k, split)
        self.tiles = tile_origins(reach.Hh, reach.Ww, self.tile, self.stride,
                                  reach.valid)
        self.index = [(i, j) for i in self.windows for j in range(len(self.tiles))]

        self.n_aux = (("dist" in self.aux)
                      + ("hist" in self.aux)
                      + (reach.aux_static.shape[0] if reach.aux_static is not None else 0))
        self.in_channels = k + self.n_aux

    def __len__(self):
        return len(self.index)

    def describe(self):
        yrs = [self.r.years[i + 1] for i in self.windows]
        return (f"{self.split:<5} windows={len(self.windows):>3} "
                f"tiles={len(self.tiles):>4} samples={len(self):>6}  "
                f"target years {min(yrs) if yrs else '-'}-{max(yrs) if yrs else '-'}")

    def __getitem__(self, n):
        i, j = self.index[n]
        y0, x0 = self.tiles[j]
        s = (slice(y0, y0 + self.tile), slice(x0, x0 + self.tile))

        chans = [self.r.water[i - self.k + 1 + c][s].astype(np.float32)
                 for c in range(self.k)]

        if "dist" in self.aux:
            chans.append(signed_distance(self.r.water[i][s]))
        if "hist" in self.aux:
            chans.append(erosion_history(self.r.delta, i)[s])
        if self.r.aux_static is not None:
            for c in range(self.r.aux_static.shape[0]):
                chans.append(self.r.aux_static[c][s])

        x = np.stack(chans)
        y = self.r.delta[i][s].astype(np.int64)
        w = self.r.valid[s].astype(np.float32)

        if self.augment and np.random.rand() < 0.5:
            # Horizontal flip only. Flipping along the flow axis reverses the
            # river and creates physically impossible training examples.
            x, y, w = x[:, :, ::-1].copy(), y[:, ::-1].copy(), w[:, ::-1].copy()

        if HAS_TORCH:
            return torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(w)
        return x, y, w


# ---------------------------------------------------------------------------
# Self-tests
# ---------------------------------------------------------------------------
def self_test(cache_dir, reach_id, k=4):
    print("=" * 68)
    print(f"SELF-TEST  ·  reach={reach_id}")
    print("=" * 68)

    r = ReachData(cache_dir, reach_id)
    print(r)

    # --- contract §3: the delta invariant ---------------------------------
    print("\n[1] delta invariant  (contract §3)")
    for t in range(r.T - 1):
        recon = apply_delta(r.water[t], r.delta[t])
        assert (recon == r.water[t + 1]).all(), f"FAILED at t={t}"
    print(f"    PASS — delta reconstructs water[t+1] for all {r.T-1} transitions")

    # --- class balance: the copy-collapse argument ------------------------
    print("\n[2] class balance  (why symmetric losses fail)")
    tot = r.delta.size
    for c, name in enumerate(CLASS_NAMES):
        n = int((r.delta == c).sum())
        print(f"    {name:<10} {n:>12,}  {100*n/tot:6.3f}%")
    rho = float((r.delta != STABLE).mean())
    print(f"\n    rho = {rho:.4f}  ->  predicting 'stable' everywhere scores "
          f"{(1-rho)*100:.2f}% accuracy.")
    print("    That is the copy solution. Quote this number in the paper.")

    # --- contract §6: no temporal leakage ---------------------------------
    print("\n[3] temporal split  (contract §6)")
    seen = {}
    for sp in ("train", "val", "test"):
        idx = window_indices(r, k, sp)
        yrs = sorted(r.years[i + 1] for i in idx)
        seen[sp] = set(yrs)
        rng = f"{yrs[0]}-{yrs[-1]}" if yrs else "empty"
        print(f"    {sp:<6} {len(idx):>3} windows  target years {rng}")
    assert not (seen["train"] & seen["test"]), "LEAK: train/test share a target year"
    assert not (seen["train"] & seen["val"]), "LEAK: train/val share a target year"
    assert not (seen["val"] & seen["test"]), "LEAK: val/test share a target year"
    if seen["train"] and seen["test"]:
        assert max(seen["train"]) < min(seen["test"]), "LEAK: train after test"
    print("    PASS — splits disjoint and ordered; no future leaks into the past")

    # --- datasets ----------------------------------------------------------
    print("\n[4] datasets")
    for sp in ("train", "val", "test"):
        ds = RiverWindows(r, k=k, split=sp)
        print(f"    {ds.describe()}")
    ds = RiverWindows(r, k=k, split="train")
    x, y, w = ds[0]
    xs = tuple(x.shape) if not HAS_TORCH else tuple(x.shape)
    print(f"\n    sample x {xs} float32   ({ds.k} water + {ds.n_aux} aux)")
    print(f"           y {tuple(y.shape)} int64 in {{0,1,2}}")
    print(f"           w {tuple(w.shape)} float32")
    print(f"    in_channels = {ds.in_channels}  <- your model's first conv")

    print("\n" + "=" * 68)
    print("ALL PASS.  Next: src/metrics.py, then analysis/baselines.py.")
    print("=" * 68)
    return r


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reach", default="STUB")
    ap.add_argument("--cache", default=os.path.join(
        os.path.dirname(__file__), "..", "data", "cache"))
    ap.add_argument("--k", type=int, default=4)
    a = ap.parse_args()
    self_test(a.cache, a.reach, a.k)
