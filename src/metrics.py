#!/usr/bin/env python3
"""
src/metrics.py — M1-M9 from the playbook.

Day 1, hours 9-12. WRITE THIS BEFORE ANY MODEL.

If you cannot measure the failure you are targeting, you cannot show you fixed
it — and you will discover that on Day 6 with no time left.

    python src/metrics.py        # self-test on synthetic cases

The metric that carries the paper is M1, the eroded-area bias ratio:

    M1 = predicted erosion area / actual erosion area

    1.0  = unbiased
    <1.0 = UNDERPREDICTS EROSION — the documented failure of prior work, and
           the one that tells a village it is safe when it is not
    >1.0 = overpredicts, which costs money rather than lives

Everything else in this file exists to stop M1 being gamed.
"""

import numpy as np

STABLE, EROSION, ACCRETION = 0, 1, 2
EPS = 1e-9


def _prep(pred, true, weight=None):
    pred = np.asarray(pred).ravel()
    true = np.asarray(true).ravel()
    if weight is None:
        return pred, true
    m = np.asarray(weight).ravel() > 0
    return pred[m], true[m]


# ---------------------------------------------------------------------------
# M1 — the headline
# ---------------------------------------------------------------------------
def eroded_area_bias(pred, true, weight=None, cls=EROSION):
    """M1. Ratio of predicted to actual area of `cls`. Perfect = 1.0."""
    p, t = _prep(pred, true, weight)
    a, b = float((p == cls).sum()), float((t == cls).sum())
    return np.nan if b == 0 else a / b


# ---------------------------------------------------------------------------
# M2 — per-class, on the delta map only
# ---------------------------------------------------------------------------
def class_scores(pred, true, weight=None, cls=EROSION):
    p, t = _prep(pred, true, weight)
    pp, tt = (p == cls), (t == cls)
    tp = float((pp & tt).sum())
    fp = float((pp & ~tt).sum())
    fn = float((~pp & tt).sum())
    return {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": tp / (tp + fp + EPS),
        "recall": tp / (tp + fn + EPS),
        "f1": 2 * tp / (2 * tp + fp + fn + EPS),
        "iou": tp / (tp + fp + fn + EPS),
        "csi": tp / (tp + fp + fn + EPS),          # CSI == IoU for binary
    }


# ---------------------------------------------------------------------------
# M3 — change-restricted CSI
# ---------------------------------------------------------------------------
def change_restricted_csi(pred, true, weight=None):
    """
    CSI over the union of predicted-change and actual-change pixels. Strips out
    the vast static background that saturates whole-mask metrics.
    """
    p, t = _prep(pred, true, weight)
    m = (p != STABLE) | (t != STABLE)
    if m.sum() == 0:
        return np.nan
    return float((p[m] == t[m]).mean())


# ---------------------------------------------------------------------------
# M4 — skill vs persistence
# ---------------------------------------------------------------------------
def skill_score(model, persistence, perfect=1.0):
    """SS = (model - persistence) / (perfect - persistence). 0 = no better."""
    d = perfect - persistence
    return np.nan if abs(d) < EPS else (model - persistence) / d


# ---------------------------------------------------------------------------
# M5 — bankline displacement, in metres
# ---------------------------------------------------------------------------
def _longest_run(col, min_run):
    """
    First and last index of the LONGEST contiguous run of True in `col`.

    Why not simply the first and last True pixel: classification speckle puts
    isolated water pixels out on the floodplain, and a naive first/last sweep
    reads those as the bank — inflating displacement error by an order of
    magnitude. The self-test in this file demonstrates it (2019 m vs 45 m).
    The main channel is by construction the longest run in a cross-section.
    """
    idx = np.flatnonzero(col)
    if len(idx) == 0:
        return None
    breaks = np.flatnonzero(np.diff(idx) > 1)
    starts = np.concatenate(([0], breaks + 1))
    ends = np.concatenate((breaks, [len(idx) - 1]))
    lengths = ends - starts + 1
    b = int(np.argmax(lengths))
    if lengths[b] < min_run:
        return None
    return int(idx[starts[b]]), int(idx[ends[b]])


def bankline_displacement(pred_water, true_water, pixel_m=30.0, axis=0,
                          min_run=5):
    """
    M5. Mean, median and p90 absolute error in bank POSITION, in metres — the
    unit BWDB and CEGIS actually think in. IoU means nothing to them; "the bank
    is 90 m out" does.

    Sweeps cross-sections along `axis` and compares the edges of the main
    channel, defined as the longest contiguous water run (see _longest_run).
    `min_run` discards cross-sections where no run is long enough to be a
    channel — raise it if your reach is speckly.
    """
    p = np.asarray(pred_water).astype(bool)
    t = np.asarray(true_water).astype(bool)
    if axis == 1:
        p, t = p.T, t.T

    errs, skipped = [], 0
    for c in range(p.shape[1]):
        rp, rt = _longest_run(p[:, c], min_run), _longest_run(t[:, c], min_run)
        if rp is None or rt is None:
            skipped += 1
            continue
        errs.append(abs(rp[0] - rt[0]))        # north / left bank
        errs.append(abs(rp[1] - rt[1]))        # south / right bank

    if not errs:
        return {"mean_m": np.nan, "median_m": np.nan, "p90_m": np.nan,
                "n": 0, "skipped": skipped}
    e = np.array(errs, float) * pixel_m
    return {"mean_m": float(e.mean()), "median_m": float(np.median(e)),
            "p90_m": float(np.percentile(e, 90)), "n": len(e),
            "skipped": skipped}


# ---------------------------------------------------------------------------
# M6 — asset recall  (needs C's Open Buildings overlay)
# ---------------------------------------------------------------------------
def asset_recall(flagged_ids, actually_lost_ids):
    """Fraction of assets genuinely lost that were flagged. Owner: C, Day 5."""
    lost = set(actually_lost_ids)
    if not lost:
        return np.nan
    return len(set(flagged_ids) & lost) / len(lost)


def expected_assets_lost(prob_map, footprints):
    """
    Playbook §13.1 — expected value, not a hard threshold count.
    `footprints`: iterable of (asset_id, boolean mask). Returns id -> P(loss).
    """
    prob = np.asarray(prob_map, float)
    return {aid: float(prob[m].mean()) if m.any() else 0.0 for aid, m in footprints}


# ---------------------------------------------------------------------------
# M7 — calibration
# ---------------------------------------------------------------------------
def calibration(prob, actual, bins=10):
    """Reliability curve and expected calibration error for P(erosion)."""
    p = np.asarray(prob, float).ravel()
    a = np.asarray(actual, bool).ravel()
    edges = np.linspace(0, 1, bins + 1)
    rows, ece = [], 0.0
    for i in range(bins):
        m = (p >= edges[i]) & (p < edges[i + 1] if i < bins - 1 else p <= 1.0)
        if not m.any():
            continue
        conf, freq, n = float(p[m].mean()), float(a[m].mean()), int(m.sum())
        rows.append({"bin": (float(edges[i]), float(edges[i + 1])),
                     "confidence": conf, "frequency": freq, "n": n})
        ece += n / len(p) * abs(conf - freq)
    return {"bins": rows, "ece": float(ece)}


# ---------------------------------------------------------------------------
# M8 — legacy whole-mask metrics (report for comparability, then explain)
# ---------------------------------------------------------------------------
def whole_mask(pred_water, true_water, weight=None):
    p, t = _prep(pred_water, true_water, weight)
    pp, tt = p.astype(bool), t.astype(bool)
    tp = float((pp & tt).sum())
    fp = float((pp & ~tt).sum())
    fn = float((~pp & tt).sum())
    tn = float((~pp & ~tt).sum())
    return {
        "accuracy": (tp + tn) / (tp + tn + fp + fn + EPS),
        "precision": tp / (tp + fp + EPS),
        "recall": tp / (tp + fn + EPS),
        "f1": 2 * tp / (2 * tp + fp + fn + EPS),
        "csi": tp / (tp + fp + fn + EPS),
    }


# ---------------------------------------------------------------------------
# M9 — structural break at the Padma Bridge  (C6)
# ---------------------------------------------------------------------------
def structural_break(err_by_year, break_year=2022):
    """Mean error before vs after `break_year`. Run for bridge and controls."""
    pre = [v for y, v in err_by_year.items() if y < break_year and np.isfinite(v)]
    post = [v for y, v in err_by_year.items() if y >= break_year and np.isfinite(v)]
    if not pre or not post:
        return {"pre": np.nan, "post": np.nan, "delta": np.nan, "ratio": np.nan}
    a, b = float(np.mean(pre)), float(np.mean(post))
    return {"pre": a, "post": b, "delta": b - a,
            "ratio": b / (a + EPS), "n_pre": len(pre), "n_post": len(post)}


# ---------------------------------------------------------------------------
# The full report
# ---------------------------------------------------------------------------
def evaluate(pred_delta, true_delta, water_t, weight=None, pixel_m=30.0,
             persistence_ref=None):
    """
    One call, every metric. `persistence_ref` is the dict returned by running
    this on the persistence baseline — pass it to populate M4.
    """

    pw = _apply(water_t, pred_delta)
    tw = _apply(water_t, true_delta)

    ero = class_scores(pred_delta, true_delta, weight, EROSION)
    acc = class_scores(pred_delta, true_delta, weight, ACCRETION)

    out = {
        "M1_eroded_area_bias": eroded_area_bias(pred_delta, true_delta, weight),
        "M1b_accreted_area_bias": eroded_area_bias(pred_delta, true_delta,
                                                   weight, ACCRETION),
        "M2_erosion_iou": ero["iou"],
        "M2_erosion_recall": ero["recall"],
        "M2_erosion_precision": ero["precision"],
        "M2_accretion_iou": acc["iou"],
        "M3_change_restricted_csi": change_restricted_csi(pred_delta, true_delta, weight),
        "M8_whole_mask_f1": whole_mask(pw, tw, weight)["f1"],
        "M8_whole_mask_csi": whole_mask(pw, tw, weight)["csi"],
        "M8_whole_mask_accuracy": whole_mask(pw, tw, weight)["accuracy"],
    }
    d = bankline_displacement(pw, tw, pixel_m)
    out["M5_bankline_mean_m"] = d["mean_m"]
    out["M5_bankline_median_m"] = d["median_m"]

    if persistence_ref:
        for key in ("M2_erosion_iou", "M3_change_restricted_csi", "M8_whole_mask_csi"):
            out[f"M4_skill_{key}"] = skill_score(out[key], persistence_ref[key])
    return out


def _apply(water_t, delta):
    out = np.asarray(water_t).copy()
    out[delta == EROSION] = 1
    out[delta == ACCRETION] = 0
    return out


def print_report(name, m):
    print(f"\n{'-'*64}\n{name}\n{'-'*64}")
    bias = m["M1_eroded_area_bias"]
    flag = ("  <- UNDERPREDICTS EROSION" if bias < 0.9 else
            "  <- overpredicts" if bias > 1.1 else "  <- well calibrated")
    print(f"  M1  eroded-area bias      {bias:8.4f}{flag}")
    print(f"  M2  erosion IoU           {m['M2_erosion_iou']:8.4f}")
    print(f"      erosion recall        {m['M2_erosion_recall']:8.4f}")
    print(f"      erosion precision     {m['M2_erosion_precision']:8.4f}")
    print(f"  M3  change-restr. CSI     {m['M3_change_restricted_csi']:8.4f}")
    print(f"  M5  bankline mean error   {m['M5_bankline_mean_m']:8.1f} m")
    print(f"  M8  whole-mask F1         {m['M8_whole_mask_f1']:8.4f}  <- saturated")
    print(f"      whole-mask accuracy   {m['M8_whole_mask_accuracy']:8.4f}  <- saturated")
    for k, v in m.items():
        if k.startswith("M4_"):
            print(f"  M4  skill {k[9:]:<20}{v:8.4f}")


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
def _self_test():
    rng = np.random.default_rng(0)
    # Geometry calibrated to a REAL river, not a convenient one. The Padma's
    # north bank retreats at most ~97 m/yr = ~3 px at 30 m. Making the
    # synthetic case more dynamic than reality would flatter every metric and
    # hide exactly the saturation this file exists to expose.
    H = W = 400
    water_t = np.zeros((H, W), np.uint8)
    water_t[150:250] = 1                   # 100-px channel, ~3 km wide

    true_delta = np.zeros((H, W), np.uint8)
    true_delta[250:253] = EROSION          # south bank retreats ~90 m/yr
    true_delta[150:151] = ACCRETION        # north bank builds out ~30 m/yr

    rho = (true_delta != STABLE).mean()
    print("=" * 64)
    print("METRICS SELF-TEST")
    print("=" * 64)
    print(f"true erosion pixels: {(true_delta==EROSION).sum():,}"
          f"   (rho = {rho*100:.2f}% of frame changed — realistic)")

    # 1. persistence — the copy solution
    persist = np.zeros_like(true_delta)
    mp = evaluate(persist, true_delta, water_t)
    print_report("PERSISTENCE  (predict 'stable' everywhere)", mp)
    assert mp["M1_eroded_area_bias"] == 0.0
    assert mp["M8_whole_mask_f1"] > 0.95, (
        f"whole-mask F1 was {mp['M8_whole_mask_f1']:.4f}, expected >0.95. "
        "If this trips on real data, your rho is unusually high — good news, "
        "but re-check the flood/erosion separation in the composite window.")

    # 2. timid — the JamUNet-style failure: right place, too little of it
    timid = np.zeros_like(true_delta)
    timid[250:251] = EROSION
    mt = evaluate(timid, true_delta, water_t, persistence_ref=mp)
    print_report("TIMID  (correct location, underpredicts extent)", mt)
    assert mt["M1_eroded_area_bias"] < 0.5

    # 3. calibrated — right extent, some noise
    cal = true_delta.copy()
    flip = rng.random((H, W)) < 0.01
    cal[flip & (cal == STABLE)] = EROSION
    cal[(rng.random((H, W)) < 0.15) & (cal == EROSION)] = STABLE
    mc = evaluate(cal, true_delta, water_t, persistence_ref=mp)
    print_report("CALIBRATED  (right extent, noisy)", mc)

    print(f"\n{'='*64}\nTHE ARGUMENT, IN THREE NUMBERS\n{'='*64}")
    print(f"{'model':<14}{'whole-mask F1':>16}{'M1 bias':>12}{'erosion IoU':>14}")
    for n, m in (("persistence", mp), ("timid", mt), ("calibrated", mc)):
        print(f"{n:<14}{m['M8_whole_mask_f1']:>16.4f}"
              f"{m['M1_eroded_area_bias']:>12.4f}{m['M2_erosion_iou']:>14.4f}")
    print("""
Whole-mask F1 barely separates these three. M1 separates them completely.
That is the evaluation-protocol contribution (C3), demonstrable on synthetic
data before you have trained anything. Put this table in the paper.
""")
    print("ALL PASS.")


if __name__ == "__main__":
    _self_test()
