#!/usr/bin/env python3
"""
data/make_stub.py — synthetic reach, so A is never blocked by Earth Engine.

    python data/make_stub.py

Writes a STUB reach to the cache that obeys DATA_CONTRACT.md exactly. Build and
test dataset.py, metrics.py, the losses and the whole training loop against it
while B is still waiting for Earth Engine approval.

The channel migrates southward at a known, constant rate, so the task is
genuinely learnable. That gives you a hard diagnostic:

    A MODEL THAT CANNOT BEAT PERSISTENCE ON THE STUB HAS A BUG, NOT A HARD
    PROBLEM. Fix it here, on synthetic data, before real data arrives.

Run this in the first ten minutes of Day 1.
"""

import json
import os

import numpy as np

T, H, W = 34, 400, 1200
YEAR0 = 1988
CACHE = os.path.join(os.path.dirname(__file__), "cache")


def main():
    os.makedirs(CACHE, exist_ok=True)
    rng = np.random.default_rng(0)

    water = np.zeros((T, H, W), np.uint8)
    x = np.arange(W)

    for t in range(T):
        # Sinuous channel that migrates south ~1.6 px/yr (~48 m/yr at 30 m),
        # which is the right order of magnitude for the Padma.
        centre = H / 2 + 60 * np.sin(x / 180.0) + 1.6 * t
        half = 34 + 9 * np.sin(x / 90.0 + t / 5.0)          # width breathes
        lo = np.clip((centre - half).astype(int), 0, H)
        hi = np.clip((centre + half).astype(int), 0, H)
        for col in range(W):
            water[t, lo[col]:hi[col], col] = 1

    # Speckle — classification noise, so the task is not trivially clean
    water |= (rng.random((T, H, W)) < 0.0015).astype(np.uint8)

    waterclass = np.where(water == 1, 3, 1).astype(np.uint8)   # 3 permanent, 1 land
    valid = np.ones((H, W), np.uint8)
    years = list(range(YEAR0, YEAR0 + T))

    # rho — the copy-collapse number, on the stub
    rho = float(np.mean([(water[t] != water[t - 1]).mean() for t in range(1, T)]))

    np.save(f"{CACHE}/STUB_water_{years[0]}_{years[-1]}.npy", water)
    np.save(f"{CACHE}/STUB_waterclass_{years[0]}_{years[-1]}.npy", waterclass)
    np.save(f"{CACHE}/STUB_valid.npy", valid)

    meta = {
        "contract_version": "1.0",
        "reach_id": "STUB",
        "name": "synthetic migrating channel",
        "river": "stub",
        "crs": "EPSG:32645",
        "transform": [700000.0, 30.0, 0.0, 2600000.0, 0.0, -30.0],
        "shape": [T, H, W],
        "years": years,
        "missing_years": [],
        "water_classes": [2, 3],
        "water_convention": "1 = water, 0 = land",
        "delta_encoding": {"0": "stable", "1": "erosion", "2": "accretion"},
        "row0": "north",
        "rho_mean_annual_change": rho,
        "valid_fraction": 1.0,
        "split": {"train": [1988, 2012], "val": [2013, 2016], "test": [2017, 2021]},
        "tile": 256,
        "stride": 128,
        "aux_static": [],
        "aux_missing": ["gsw_occurrence", "dist_bridge_km"],
        "files": {
            "water": f"STUB_water_{years[0]}_{years[-1]}.npy",
            "waterclass": f"STUB_waterclass_{years[0]}_{years[-1]}.npy",
            "valid": "STUB_valid.npy",
        },
        "source": "synthetic — data/make_stub.py",
    }
    with open(f"{CACHE}/STUB_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"STUB written to {CACHE}")
    print(f"  shape {water.shape}  ({water.nbytes / 1e6:.0f} MB)")
    print(f"  water fraction     {water.mean():.3f}")
    print(f"  rho (annual change) {rho:.4f}  ->  persistence accuracy {(1-rho)*100:.2f}%")
    print("\nA is unblocked. Next: src/dataset.py, then src/metrics.py.")


if __name__ == "__main__":
    main()
