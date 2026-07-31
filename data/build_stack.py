#!/usr/bin/env python3
"""
data/build_stack.py — GeoTIFFs -> .npy cache, and the grid-alignment gate.

    python data/build_stack.py --reach P1 --tif-dir /content/drive/MyDrive/padma_erosion
    python data/build_stack.py --reach P1 --tif-dir ... --plot   # visual QC (item 11)

This script exists to enforce the data contract in DATA_CONTRACT.md. It refuses
to write a cache if anything is inconsistent, because a silent one-pixel grid
offset between two years turns into a stripe of fake erosion that you will not
notice until Day 6.

Outputs, all in cache_dir:
    {reach}_waterclass_{y0}_{y1}.npy   uint8  [T,H,W]   raw GSW class 0-3
    {reach}_water_{y0}_{y1}.npy        uint8  [T,H,W]   1 = water, 0 = land
    {reach}_valid.npy                  uint8  [H,W]     1 = usable pixel
    {reach}_meta.json                  the contract for this reach

Requires:  pip install rasterio numpy pyyaml matplotlib
"""

import argparse
import glob
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone

import numpy as np
import rasterio
import yaml

CONFIG = os.path.join(os.path.dirname(__file__), "..", "configs", "reaches.yaml")


def git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def load_cfg(path):
    with open(path) as f:
        return yaml.safe_load(f)


def collect(tif_dir, reach_id):
    """Find {reach}_gsw_{year}.tif, excluding smoke-test files."""
    hits = []
    for p in sorted(glob.glob(os.path.join(tif_dir, f"{reach_id}_gsw_*.tif"))):
        if "_smoke" in p:
            continue
        m = re.search(rf"{reach_id}_gsw_(\d{{4}})\.tif$", os.path.basename(p))
        if m:
            hits.append((int(m.group(1)), p))
    return sorted(hits)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reach", required=True)
    ap.add_argument("--tif-dir", required=True, help="Downloaded/mounted Drive folder")
    ap.add_argument("--config", default=CONFIG)
    ap.add_argument("--plot", action="store_true", help="Write a QC contact sheet")
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    defaults = cfg["defaults"]
    reach_cfg = cfg["reaches"][args.reach]
    cache_dir = cfg["project"]["cache_dir"]
    os.makedirs(cache_dir, exist_ok=True)

    water_classes = reach_cfg.get("water_classes", defaults["water_classes"])

    # ---- grid spec written by gee_export.py -------------------------------
    gpath = os.path.join(cache_dir, f"{args.reach}_grid.json")
    grid = None
    if os.path.exists(gpath):
        grid = json.load(open(gpath))
        print(f"Grid spec: {gpath}")
    else:
        print(f"WARNING: {gpath} missing. Deriving the grid from the first "
              f"GeoTIFF instead — alignment cannot be checked against intent.")

    files = collect(args.tif_dir, args.reach)
    if not files:
        sys.exit(f"No GeoTIFFs matching {args.reach}_gsw_YYYY.tif in {args.tif_dir}")
    print(f"Found {len(files)} files: {files[0][0]}-{files[-1][0]}")

    # ---- gate 1: every raster shares one grid -----------------------------
    ref_shape = ref_transform = ref_crs = None
    arrays, years = [], []

    for year, path in files:
        with rasterio.open(path) as src:
            if ref_shape is None:
                ref_shape, ref_transform, ref_crs = src.shape, src.transform, src.crs
                print(f"Reference grid: {ref_shape}  {ref_crs}")
            else:
                if src.shape != ref_shape:
                    sys.exit(
                        f"\nSHAPE MISMATCH in {os.path.basename(path)}\n"
                        f"  expected {ref_shape}, got {src.shape}\n\n"
                        f"Re-export with an explicit crsTransform. Do not resample "
                        f"to force agreement — the offset is real and would become "
                        f"fake erosion along one edge of every change map.\n"
                    )
                if not np.allclose(
                    np.array(src.transform).astype(float),
                    np.array(ref_transform).astype(float),
                    atol=1e-6,
                ):
                    sys.exit(
                        f"\nTRANSFORM MISMATCH in {os.path.basename(path)}\n"
                        f"  expected {ref_transform}\n  got      {src.transform}\n\n"
                        f"Sub-pixel misregistration. Re-export this year.\n"
                    )
                if src.crs != ref_crs:
                    sys.exit(f"CRS MISMATCH in {path}: {src.crs} vs {ref_crs}")
            arrays.append(src.read(1).astype(np.uint8))
            years.append(year)

    # ---- gate 2: agrees with the grid gee_export intended -----------------
    if grid is not None:
        want = (grid["height"], grid["width"])
        if ref_shape != want:
            sys.exit(
                f"\nRasters are {ref_shape} but {gpath} specifies {want}.\n"
                f"The config changed after export. Re-export or restore the config — "
                f"do not proceed with a mismatch.\n"
            )
        print("Grid matches the exported spec.")

    # ---- gate 3: no missing years -----------------------------------------
    expected = set(range(min(years), max(years) + 1))
    missing = sorted(expected - set(years))
    if missing:
        print(f"\nWARNING: missing years {missing}")
        print("Sequences spanning a gap are invalid. Either re-export, or make")
        print("dataset.py skip any window containing a gap. Do NOT interpolate —")
        print("an interpolated year is a fabricated observation.\n")

    wc = np.stack(arrays)                                   # [T,H,W] uint8, 0-3
    T, H, W = wc.shape

    # ---- binary water + validity ------------------------------------------
    water = np.isin(wc, water_classes).astype(np.uint8)     # 1 = water
    valid = (wc != 0).all(axis=0).astype(np.uint8)          # class 0 = no data

    valid_frac = valid.mean()
    print(f"\nStack {wc.shape}  ({wc.nbytes / 1e6:.0f} MB)")
    print(f"Water classes {water_classes} -> water mean {water.mean():.3f}")
    print(f"Valid pixels  {valid_frac:.3f}")
    if valid_frac < 0.90:
        print("  WARNING: >10% of pixels have no data in at least one year.")
        print("  Usually Landsat 7 SLC-off striping. Check the per-year report.")

    # ---- the number that drives every design decision ---------------------
    ch = []
    for t in range(1, T):
        m = valid.astype(bool)
        ch.append(float((water[t][m] != water[t - 1][m]).mean()))
    rho = float(np.mean(ch))

    print(f"\nMean annual change rate  rho = {rho:.4f}  ({rho * 100:.2f}% of pixels)")
    print(f"  Persistence upper-bound accuracy: {(1 - rho) * 100:.2f}%")
    print("  This is the copy-collapse argument, measured on your own data.")
    print("  Quote it in the paper. If rho > 0.05 your signal is unusually strong;")
    print("  if rho < 0.01 lean even harder on the change-restricted metrics.")

    print("\nPer-year detail:")
    print(f"  {'year':<6}{'water%':>9}{'valid%':>9}{'changed%':>11}")
    for i, y in enumerate(years):
        c = f"{ch[i-1]*100:>10.2f}%" if i else f"{'—':>11}"
        print(f"  {y:<6}{water[i].mean()*100:>8.2f}%{(wc[i]!=0).mean()*100:>8.2f}%{c}")

    # ---- write -------------------------------------------------------------
    y0, y1 = years[0], years[-1]
    tag = f"{args.reach}_{y0}_{y1}"
    p_wc = os.path.join(cache_dir, f"{args.reach}_waterclass_{y0}_{y1}.npy")
    p_w = os.path.join(cache_dir, f"{args.reach}_water_{y0}_{y1}.npy")
    p_v = os.path.join(cache_dir, f"{args.reach}_valid.npy")
    p_m = os.path.join(cache_dir, f"{args.reach}_meta.json")

    np.save(p_wc, wc)
    np.save(p_w, water)
    np.save(p_v, valid)

    meta = {
        "contract_version": "1.0",
        "reach_id": args.reach,
        "name": reach_cfg["name"],
        "river": reach_cfg["river"],
        "crs": str(ref_crs),
        "transform": [float(v) for v in ref_transform.to_gdal()],
        "shape": [T, H, W],
        "years": years,
        "missing_years": missing,
        "water_classes": water_classes,
        "water_convention": "1 = water, 0 = land",
        "delta_encoding": {"0": "stable", "1": "erosion", "2": "accretion"},
        "row0": "north",
        "rho_mean_annual_change": rho,
        "valid_fraction": valid_frac,
        "split": defaults["split"],
        "tile": defaults["tile"],
        "stride": defaults["stride"],
        "files": {
            "waterclass": os.path.basename(p_wc),
            "water": os.path.basename(p_w),
            "valid": os.path.basename(p_v),
        },
        "source": "JRC/GSW1_4/YearlyHistory",
        "git_commit": git_commit(),
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    with open(p_m, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nWrote:\n  {p_wc}\n  {p_w}\n  {p_v}\n  {p_m}")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        n = len(years)
        cols = 6
        rows = int(np.ceil(n / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.6, rows * 2.6))
        for ax, (i, y) in zip(np.atleast_1d(axes).ravel(), enumerate(years)):
            ax.imshow(water[i], cmap="Blues", vmin=0, vmax=1, interpolation="nearest")
            ax.set_title(str(y), fontsize=8)
            ax.axis("off")
        for ax in np.atleast_1d(axes).ravel()[n:]:
            ax.axis("off")
        fig.suptitle(f"{args.reach} — annual water masks (classes {water_classes})",
                     fontsize=11)
        fig.tight_layout()
        out = os.path.join(cache_dir, f"{tag}_contactsheet.png")
        fig.savefig(out, dpi=130, bbox_inches="tight")
        print(f"  {out}")

    print(f"""
NOW LOOK AT THE CONTACT SHEET. Every year, one at a time. You are checking:

  · Does the channel look like a river in all {len(years)} frames?
  · Any year that is mostly blank, mostly blue, or striped?
      blank/blue -> cloud or classification failure, drop or re-derive the year
      striped    -> Landsat 7 SLC-off, expected post-2003, note it in the paper
  · Does the river visibly migrate across the sequence? If nothing moves, your
    bbox is probably off the active channel.
  · Do the migration directions match the literature — north bank eroding,
    south bank accreting on the Padma?

This takes fifteen minutes and is the highest-value fifteen minutes of Day 1.
Bad data found now costs an hour. Found on Day 5, it costs the project.

Next:  src/dataset.py, then src/metrics.py + the persistence baseline —
       both BEFORE any model.
""")


if __name__ == "__main__":
    main()
