#!/usr/bin/env python3
"""
analysis/baselines.py — B0 persistence, and the anchor table.

Day 1, hours 9-12, immediately after metrics.py. Before any model.

    python analysis/baselines.py --reach STUB
    python analysis/baselines.py --reach P1 --split test

Writes results/baselines_{reach}_{split}.csv. Every later result is quoted
against this table, so run it once and keep it.

B0 persistence is the benchmark prior work used — "no morphological change
occurs". In the delta framing it is the constant prediction `stable`.

B1 (DSAS end-point rate) and B2 (ARIMA) are stubbed at the bottom with the
interface they must satisfy. Day 3, hours 16-18. B6 (CA-Markov, following
Ritu et al. 2023) goes there too.
"""

import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from dataset import ReachData, window_indices, STABLE, EROSION      # noqa: E402
import metrics as M                                                  # noqa: E402


def evaluate_over_split(reach: ReachData, predict_fn, split="test", k=4):
    """
    predict_fn(reach, i) -> delta prediction for the transition years[i]->[i+1].
    Evaluates per target year and pooled, so you can see which years are hard.
    """
    idx = window_indices(reach, k, split)
    if not idx:
        raise SystemExit(f"No windows in split '{split}'. Check meta['split'].")

    per_year, P, T = {}, [], []
    for i in idx:
        pred = predict_fn(reach, i)
        true = reach.delta[i]
        w = reach.valid
        per_year[reach.years[i + 1]] = M.evaluate(pred, true, reach.water[i], w)
        m = w > 0
        P.append(pred[m])
        T.append(true[m])

    # Pooled over the split — the number that goes in the paper
    Pc, Tc = np.concatenate(P), np.concatenate(T)
    ero = M.class_scores(Pc, Tc, cls=EROSION)
    pooled = {
        "M1_eroded_area_bias": M.eroded_area_bias(Pc, Tc),
        "M2_erosion_iou": ero["iou"],
        "M2_erosion_recall": ero["recall"],
        "M2_erosion_precision": ero["precision"],
        "M3_change_restricted_csi": M.change_restricted_csi(Pc, Tc),
    }
    # Whole-mask and bankline need 2-D geometry, so average the per-year values
    for key in ("M8_whole_mask_f1", "M8_whole_mask_csi", "M8_whole_mask_accuracy",
                "M5_bankline_mean_m", "M5_bankline_median_m"):
        vals = [v[key] for v in per_year.values() if np.isfinite(v[key])]
        pooled[key] = float(np.mean(vals)) if vals else np.nan
    return pooled, per_year


# ---------------------------------------------------------------------------
# B0 — persistence
# ---------------------------------------------------------------------------
def b0_persistence(reach, i):
    """No morphological change. The benchmark prior work improved on by 5-6%."""
    return np.zeros_like(reach.delta[i])


# ---------------------------------------------------------------------------
# B1 / B2 / B6 — Day 3, hours 16-18
# ---------------------------------------------------------------------------
def b1_dsas_endpoint_rate(reach, i, lookback=5):
    """
    TODO (Day 3). Per cross-section, fit the bank's end-point rate over the
    previous `lookback` years and extrapolate one year forward.

    This is what practitioners actually do, and it is surprisingly hard to beat
    on straight reaches. Expect it to be competitive — that is a finding, not a
    failure, and it belongs in the Results discussion.
    """
    raise NotImplementedError("B1 — Day 3 h16")


def b2_arima(reach, i, lookback=10):
    """TODO (Day 3). ARIMA per cross-section on bank position. Published for
    this exact problem, so it is cheap credibility."""
    raise NotImplementedError("B2 — Day 3 h16")


def b6_ca_markov(reach, i):
    """
    TODO (Day 3). DSAS + CA-Markov, following Ritu, Sarkar & Zonaed (2023),
    Ecological Indicators — a PUBLISHED prediction baseline on the Padma.
    Do not skip this one; benchmarking against a real published method on your
    own river is unusually strong and costs about three hours.
    """
    raise NotImplementedError("B6 — Day 3 h16")


BASELINES = {
    "B0_persistence": b0_persistence,
    # "B1_dsas": b1_dsas_endpoint_rate,
    # "B2_arima": b2_arima,
    # "B6_ca_markov": b6_ca_markov,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reach", default="STUB")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--cache", default=os.path.join(
        os.path.dirname(__file__), "..", "data", "cache"))
    ap.add_argument("--out", default=os.path.join(
        os.path.dirname(__file__), "..", "results"))
    a = ap.parse_args()

    reach = ReachData(a.cache, a.reach)
    print(f"{reach}\nsplit={a.split}  k={a.k}\n")

    rows = {}
    for name, fn in BASELINES.items():
        pooled, per_year = evaluate_over_split(reach, fn, a.split, a.k)
        rows[name] = pooled
        M.print_report(f"{name}  ·  {a.reach}  ·  {a.split}", pooled)
        print("\n  per target year:")
        print(f"    {'year':<7}{'M1 bias':>10}{'ero IoU':>10}{'whole-F1':>11}")
        for y, v in sorted(per_year.items()):
            print(f"    {y:<7}{v['M1_eroded_area_bias']:>10.4f}"
                  f"{v['M2_erosion_iou']:>10.4f}{v['M8_whole_mask_f1']:>11.4f}")

    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, f"baselines_{a.reach}_{a.split}.csv")
    keys = sorted({k for r in rows.values() for k in r})
    with open(path, "w", newline="") as f:
        wcsv = csv.writer(f)
        wcsv.writerow(["baseline"] + keys)
        for name, r in rows.items():
            wcsv.writerow([name] + [r.get(k, "") for k in keys])
    print(f"\nWrote {path}")

    p = rows["B0_persistence"]
    print(f"""
{'='*66}
THE ANCHOR — quote these in the paper and on every slide
{'='*66}
  Persistence whole-mask F1        {p['M8_whole_mask_f1']:.4f}   <- saturated
  Persistence whole-mask accuracy  {p['M8_whole_mask_accuracy']:.4f}   <- saturated
  Persistence M1 eroded-area bias  {p['M1_eroded_area_bias']:.4f}   <- sees NO erosion
  Persistence erosion IoU          {p['M2_erosion_iou']:.4f}

Doing nothing scores ~{p['M8_whole_mask_f1']*100:.0f}% on the metric the field reports, and
exactly zero on the metric that matters. That gap IS contribution C3.

Any model you train from here is measured against this row. If a model beats
persistence on whole-mask F1 but not on M1, it has learned to copy — which is
precisely the failure this project exists to fix.

Next: src/losses.py, then the Day-2 experiments (B3 state framing, A1 delta).
{'='*66}""")


if __name__ == "__main__":
    main()
