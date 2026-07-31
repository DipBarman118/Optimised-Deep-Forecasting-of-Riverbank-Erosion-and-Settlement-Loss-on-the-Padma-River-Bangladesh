#!/usr/bin/env python3
"""
data/gee_export.py — Earth Engine export, parameterised by reach.

Covers checklist items 7 and 8.

    # Item 8: smoke test. One year, ~10 km box. Run this FIRST.
    python data/gee_export.py --reach P1 --smoke

    # Full export, all years, one reach
    python data/gee_export.py --reach P1
    python data/gee_export.py --reach P2

    # Day 4: adding the Jamuna is this, and nothing else
    python data/gee_export.py --reach J1

    # Watch running tasks
    python data/gee_export.py --status

Every export for a given reach lands on an IDENTICAL pixel grid, because the
grid is computed once from the config and passed explicitly as crsTransform.
If you let Earth Engine infer the grid from `region` + `scale`, exports can
differ by a pixel between years — and every one of those pixels becomes fake
erosion in your change map. build_stack.py re-checks this and fails loudly.

Requires:  pip install earthengine-api pyproj pyyaml
"""

import argparse
import json
import math
import os
import sys
import time

import ee
import yaml
from pyproj import Transformer

CONFIG = os.path.join(os.path.dirname(__file__), "..", "configs", "reaches.yaml")


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
def load_config(path=CONFIG):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if "REPLACE" in cfg["project"]["ee_project"]:
        sys.exit(
            "\nSet project.ee_project in configs/reaches.yaml to your Google "
            "Cloud project ID.\nFind it at https://console.cloud.google.com "
            "(project selector, top bar).\n"
        )
    return cfg


def get_reach(cfg, reach_id):
    if reach_id not in cfg["reaches"]:
        sys.exit(f"Unknown reach '{reach_id}'. Available: {list(cfg['reaches'])}")
    reach = dict(cfg["defaults"])
    reach.update(cfg["reaches"][reach_id])
    reach["id"] = reach_id
    return reach


def init_ee(project):
    try:
        ee.Initialize(project=project)
    except Exception:
        print("Authenticating with Earth Engine (a browser window will open)...")
        ee.Authenticate()
        ee.Initialize(project=project)
    print(f"Earth Engine ready  ·  project={project}")


# --------------------------------------------------------------------------
# The export grid — computed once, reused for every year
# --------------------------------------------------------------------------
def build_grid(bbox, crs, scale):
    """
    Project the lon/lat bbox into `crs`, snap to the `scale` grid, and return
    an affine transform plus raster dimensions.

    Snapping to a multiple of `scale` means the grid origin is reproducible
    from the config alone. Two people running this on different machines get
    byte-identical geometry.
    """
    west, south, east, north = bbox
    tr = Transformer.from_crs("EPSG:4326", crs, always_xy=True)

    # All four corners — the projected box is not axis-aligned with lon/lat
    xs, ys = [], []
    for lon, lat in [(west, south), (west, north), (east, south), (east, north)]:
        x, y = tr.transform(lon, lat)
        xs.append(x)
        ys.append(y)

    xmin = math.floor(min(xs) / scale) * scale
    xmax = math.ceil(max(xs) / scale) * scale
    ymin = math.floor(min(ys) / scale) * scale
    ymax = math.ceil(max(ys) / scale) * scale

    width = int(round((xmax - xmin) / scale))
    height = int(round((ymax - ymin) / scale))

    # GDAL/rasterio affine order: [a, b, c, d, e, f]
    # x = a*col + b*row + c ;  y = d*col + e*row + f
    # Negative e means row 0 is the NORTH edge (standard raster orientation).
    transform = [scale, 0.0, float(xmin), 0.0, -scale, float(ymax)]

    return {
        "crs": crs,
        "transform": transform,
        "width": width,
        "height": height,
        "bounds_proj": [xmin, ymin, xmax, ymax],
    }


def grid_region(grid):
    """The export footprint as an ee.Geometry in the projected CRS."""
    xmin, ymin, xmax, ymax = grid["bounds_proj"]
    return ee.Geometry.Rectangle(
        [xmin, ymin, xmax, ymax], proj=grid["crs"], geodesic=False
    )


# --------------------------------------------------------------------------
# Imagery
# --------------------------------------------------------------------------
def gsw_waterclass(year):
    """
    Raw JRC GSW yearly water class for `year`.

    Band `waterClass`:  0 no data · 1 not water · 2 seasonal · 3 permanent

    We export the RAW class, not a binary mask. Binarising happens locally in
    build_stack.py, so changing `water_classes` in the config is a 10-second
    re-run instead of a re-export. It also preserves the no-data information
    you need to build the validity mask.
    """
    col = ee.ImageCollection("JRC/GSW1_4/YearlyHistory")
    img = col.filter(ee.Filter.eq("year", year)).first()
    return ee.Image(img).select("waterClass").toUint8()


def export_year(reach, grid, year, folder, smoke=False):
    img = gsw_waterclass(year)
    tag = f"{reach['id']}_gsw_{year}" + ("_smoke" if smoke else "")

    task = ee.batch.Export.image.toDrive(
        image=img.clip(grid_region(grid)),
        description=tag,
        folder=folder,
        fileNamePrefix=tag,
        crs=grid["crs"],
        crsTransform=grid["transform"],       # explicit grid — the whole point
        dimensions=f"{grid['width']}x{grid['height']}",
        maxPixels=1e10,
        fileFormat="GeoTIFF",
    )
    task.start()
    return task, tag


# --------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------
def cmd_smoke(cfg, reach_id):
    """
    Item 8 — end-to-end smoke test.

    Shrinks the reach to a ~10 km box at its centre and exports a single year.
    Purpose is to surface authentication, project, quota and Drive problems in
    two minutes rather than at hour five of a full export.
    """
    reach = get_reach(cfg, reach_id)
    init_ee(cfg["project"]["ee_project"])

    w, s, e, n = reach["bbox"]
    clon, clat = (w + e) / 2, (s + n) / 2
    d = 0.045                                   # ~5 km in degrees, so a ~10 km box
    small = [clon - d, clat - d, clon + d, clat + d]

    grid = build_grid(small, reach["crs"], reach["scale"])
    year = reach["split"]["test"][0]

    print(f"\nSMOKE TEST  ·  reach={reach_id}  year={year}")
    print(f"  bbox      {[round(v, 4) for v in small]}")
    print(f"  crs       {grid['crs']}")
    print(f"  grid      {grid['width']} x {grid['height']} px at {reach['scale']} m")
    print(f"  origin    x={grid['transform'][2]:.0f}  y={grid['transform'][5]:.0f}")
    print(f"  expected  ~{grid['width'] * grid['height'] / 1e6:.2f} Mpx, a few hundred KB\n")

    task, tag = export_year(reach, grid, year, cfg["project"]["drive_folder"], smoke=True)

    print(f"Submitted '{tag}'. Polling...\n")
    t0 = time.time()
    while True:
        st = task.status()
        state = st["state"]
        print(f"  [{time.time() - t0:6.0f}s] {state}")
        if state in ("COMPLETED", "FAILED", "CANCELLED"):
            if state == "FAILED":
                print("\nFAILED: " + st.get("error_message", "no message"))
                print(_troubleshoot())
                sys.exit(1)
            break
        time.sleep(15)

    print(f"""
SMOKE TEST PASSED in {time.time() - t0:.0f}s.

Now verify by eye — do not skip this:
  1. Open Google Drive, folder '{cfg['project']['drive_folder']}'
  2. Download {tag}.tif
  3. Open it in QGIS, or run:

       import rasterio, numpy as np
       src = rasterio.open("{tag}.tif")
       a = src.read(1)
       print(src.crs, src.shape, src.transform)
       print("class counts:", dict(zip(*np.unique(a, return_counts=True))))

  You should see classes 0-3, with plenty of 3 (permanent water) if the box
  landed on the channel. All zeros means the box missed the river — recentre
  the bbox in configs/reaches.yaml and re-run.

Next:  python data/gee_export.py --reach {reach_id}
""")


def cmd_export(cfg, reach_id):
    reach = get_reach(cfg, reach_id)
    init_ee(cfg["project"]["ee_project"])

    grid = build_grid(reach["bbox"], reach["crs"], reach["scale"])
    y0, y1 = reach["years"]
    years = list(range(y0, y1 + 1))

    mpx = grid["width"] * grid["height"] / 1e6
    print(f"\nEXPORT  ·  reach={reach_id}  ({reach['name']})")
    print(f"  grid      {grid['width']} x {grid['height']} px  ({mpx:.1f} Mpx)")
    print(f"  years     {y0}-{y1}  ({len(years)} exports)")
    print(f"  est size  ~{mpx * len(years) / 1000:.2f} GB uncompressed, far less as GeoTIFF")
    print(f"  folder    Drive/{cfg['project']['drive_folder']}\n")

    if mpx > 400:
        print("WARNING: grid exceeds 400 Mpx. Tighten the bbox in the config —")
        print("you want the channel belt plus ~3 km, not the whole district.\n")

    meta = {
        "reach_id": reach_id,
        "name": reach["name"],
        "river": reach["river"],
        "bbox_4326": reach["bbox"],
        "crs": grid["crs"],
        "transform": grid["transform"],
        "width": grid["width"],
        "height": grid["height"],
        "scale_m": reach["scale"],
        "years": years,
        "water_classes": reach["water_classes"],
        "source": "JRC/GSW1_4/YearlyHistory",
        "band": "waterClass",
    }
    os.makedirs(cfg["project"]["cache_dir"], exist_ok=True)
    mpath = os.path.join(cfg["project"]["cache_dir"], f"{reach_id}_grid.json")
    with open(mpath, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Wrote grid spec -> {mpath}")
    print("build_stack.py validates every GeoTIFF against this file.\n")

    for year in years:
        _, tag = export_year(reach, grid, year, cfg["project"]["drive_folder"])
        print(f"  queued  {tag}")

    print(f"""
{len(years)} tasks queued. Earth Engine runs a few concurrently; expect
20-60 minutes for the full set.

  Monitor:  https://code.earthengine.google.com/tasks
       or:  python data/gee_export.py --status

When all are COMPLETED:
  1. Download the folder from Drive (or mount it in Colab)
  2. python data/build_stack.py --reach {reach_id}
  3. PLOT ALL {len(years)} YEARS AND LOOK AT THEM  <- do not skip
""")


def cmd_status(cfg):
    init_ee(cfg["project"]["ee_project"])
    tasks = ee.batch.Task.list()[:40]
    counts = {}
    print(f"\n{'STATE':<12} {'DESCRIPTION':<34} NOTE")
    print("-" * 78)
    for t in tasks:
        st = t.status()
        state = st["state"]
        counts[state] = counts.get(state, 0) + 1
        note = st.get("error_message", "")[:24] if state == "FAILED" else ""
        print(f"{state:<12} {st.get('description', '?'):<34} {note}")
    print("-" * 78)
    print("  ".join(f"{k}={v}" for k, v in sorted(counts.items())) or "no tasks")
    print()


def _troubleshoot():
    return """
Common causes, in order of likelihood:

  "not signed up for Earth Engine"     Register at earthengine.google.com and
                                       wait for approval. This is why it is
                                       checklist item 1.

  "project is not registered"          Your GCP project needs the Earth Engine
                                       API enabled. Console -> APIs & Services
                                       -> enable "Google Earth Engine API".

  "Image.clip: Parameter 'input'..."   The year has no GSW image. YearlyHistory
                                       covers 1984-2021 only. Check `years`.

  "Export region is empty"             The bbox is malformed. Order is
                                       [west, south, east, north].

  "User memory limit exceeded"         Grid too large. Tighten the bbox.
"""


def main():
    p = argparse.ArgumentParser(description="Earth Engine export, per reach")
    p.add_argument("--reach", help="Reach ID from configs/reaches.yaml (P1, P2, J1)")
    p.add_argument("--smoke", action="store_true", help="One year, ~10 km box (item 8)")
    p.add_argument("--status", action="store_true", help="Show recent task states")
    p.add_argument("--config", default=CONFIG)
    a = p.parse_args()

    cfg = load_config(a.config)
    if a.status:
        cmd_status(cfg)
    elif a.smoke:
        if not a.reach:
            sys.exit("--smoke needs --reach")
        cmd_smoke(cfg, a.reach)
    elif a.reach:
        cmd_export(cfg, a.reach)
    else:
        p.print_help()


if __name__ == "__main__":
    main()
