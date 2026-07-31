#!/usr/bin/env python3
"""
src/losses.py — the scientific core (contribution C2).

Day 1, hour 16.

    python src/losses.py          # self-test; works with or without torch

THE FORMULA, stated once so there is no ambiguity:

    TI_c = TP_c / ( TP_c + ALPHA * FN_c + BETA * FP_c ),   ALPHA + BETA = 1

    >>> ALPHA MULTIPLIES FALSE NEGATIVES. <<<

The literature is genuinely inconsistent about which Greek letter goes on which
error term — some papers put beta on FN. Write this formula explicitly in your
paper and state the convention, or a reviewer cannot tell what you did.

    ALPHA = BETA = 0.5   ->  Dice. Symmetric. The likely cause of the erosion
                             underprediction reported in prior work.
    ALPHA = 0.7          ->  a missed erosion pixel costs 2.33x a false alarm.
                             YOUR DEFAULT.

Focal Tversky (Abraham & Khan, arXiv:1810.07842):

    FTL_c = ( 1 - TI_c ) ** (1 / GAMMA),   GAMMA > 1

With GAMMA > 1 the gradient is amplified where TI < 0.5, focusing training on
the hard, small, rare structures. Erosion patches on a braided river are
geometrically analogous to small lesions: thin, elongated, low-area, and
high-consequence to miss.

WHY THIS IS THE PROJECT: prior work optimises a symmetric objective on a task
where under 3% of pixels change, so the loss is minimised by predicting almost
no change. Raising ALPHA makes a missed erosion pixel explicitly more expensive
than a false alarm, which is also the correct real-world cost asymmetry: a
false negative loses a house, a false positive costs an inspection.
"""

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    nn = type("nn", (), {"Module": object})

STABLE, EROSION, ACCRETION = 0, 1, 2
EPS = 1e-6


# ===========================================================================
# NumPy reference — the ground truth for the maths.
# Tested here so the formula is verified independently of any framework.
# ===========================================================================
def tversky_index_np(prob, target_onehot, alpha=0.7, beta=0.3, weight=None):
    """
    prob, target_onehot : [C, ...] float, prob sums to 1 over C
    weight              : [...] float or None, 0 = ignore pixel
    returns             : [C] Tversky index per class
    """
    p = np.asarray(prob, float)
    g = np.asarray(target_onehot, float)
    if weight is not None:
        w = np.asarray(weight, float)[None]
        p, g = p * w, g * w
    axes = tuple(range(1, p.ndim))
    tp = (p * g).sum(axes)
    fn = ((1 - p) * g).sum(axes)
    fp = (p * (1 - g)).sum(axes)
    return tp / (tp + alpha * fn + beta * fp + EPS)


def focal_tversky_np(prob, target_onehot, alpha=0.7, beta=0.3, gamma=1.333,
                     weight=None, class_weights=None):
    ti = tversky_index_np(prob, target_onehot, alpha, beta, weight)
    ftl = (1.0 - ti) ** (1.0 / gamma)
    if class_weights is not None:
        cw = np.asarray(class_weights, float)
        return float((ftl * cw).sum() / cw.sum())
    return float(ftl.mean())


# ===========================================================================
# Torch implementations
# ===========================================================================
class TverskyLoss(nn.Module):
    """
    Asymmetric region loss. `alpha` weights FALSE NEGATIVES.

    logits : [B, C, H, W]
    target : [B, H, W] int64 in {0..C-1}
    weight : [B, H, W] float, 0 = ignore (use the validity mask)
    """

    def __init__(self, alpha=0.7, beta=None, gamma=1.0,
                 class_weights=None, ignore_class=None):
        super().__init__()
        self.alpha = alpha
        self.beta = (1.0 - alpha) if beta is None else beta
        self.gamma = gamma                       # 1.0 = plain Tversky
        self.class_weights = class_weights
        self.ignore_class = ignore_class

    def forward(self, logits, target, weight=None):
        C = logits.shape[1]
        p = torch.softmax(logits, dim=1)
        g = F.one_hot(target.long(), C).permute(0, 3, 1, 2).float()

        if weight is not None:
            w = weight.unsqueeze(1)
            p, g = p * w, g * w

        dims = (0, 2, 3)                          # sum over batch and space
        tp = (p * g).sum(dims)
        fn = ((1 - p) * g).sum(dims)
        fp = (p * (1 - g)).sum(dims)

        ti = tp / (tp + self.alpha * fn + self.beta * fp + EPS)
        loss = (1.0 - ti) ** (1.0 / self.gamma)

        keep = [c for c in range(C) if c != self.ignore_class]
        loss = loss[keep]

        if self.class_weights is not None:
            cw = torch.as_tensor(self.class_weights, dtype=loss.dtype,
                                 device=loss.device)[keep]
            return (loss * cw).sum() / cw.sum()
        return loss.mean()


class FocalTverskyLoss(TverskyLoss):
    """Tversky with gamma > 1. Default gamma = 4/3, a safe starting point."""

    def __init__(self, alpha=0.7, beta=None, gamma=1.3333, **kw):
        super().__init__(alpha=alpha, beta=beta, gamma=gamma, **kw)


class BoundaryWeightedCE(nn.Module):
    """
    Cross-entropy up-weighted in a band around the target boundary.

    HONEST NAMING: this is NOT Kervadec et al.'s boundary loss, which integrates
    a precomputed level-set distance map. This is a cheaper, GPU-native proxy —
    morphological dilate minus erode via max-pool gives a band, and pixels in
    the band get weight (1 + lam). Describe it as "boundary-weighted
    cross-entropy" in the paper, not as "boundary loss". Claiming the stronger
    method and shipping the weaker one is the kind of thing a careful reviewer
    checks.

    It targets M5 (bankline displacement in metres) specifically: region losses
    treat every misplaced pixel the same regardless of how far off it is.
    """

    def __init__(self, lam=2.0, width=3, class_weights=None):
        super().__init__()
        self.lam = lam
        self.k = 2 * width + 1
        self.class_weights = class_weights

    def _band(self, target, C):
        g = F.one_hot(target.long(), C).permute(0, 3, 1, 2).float()
        pad = self.k // 2
        dil = F.max_pool2d(g, self.k, stride=1, padding=pad)
        ero = -F.max_pool2d(-g, self.k, stride=1, padding=pad)
        return (dil - ero).amax(dim=1).clamp(0, 1)      # [B, H, W]

    def forward(self, logits, target, weight=None):
        C = logits.shape[1]
        cw = (torch.as_tensor(self.class_weights, dtype=logits.dtype,
                              device=logits.device)
              if self.class_weights is not None else None)
        ce = F.cross_entropy(logits, target.long(), weight=cw, reduction="none")
        w = 1.0 + self.lam * self._band(target, C)
        if weight is not None:
            w = w * weight
        return (ce * w).sum() / (w.sum() + EPS)


class ComboLoss(nn.Module):
    """
    The recipe from playbook section 9.4.

      epochs 0..warmup-1 : weighted cross-entropy only
      thereafter         : FocalTversky + lam_boundary * BoundaryWeightedCE

    Tversky-family losses are unstable from a random initialisation — the index
    is near zero for every class, gradients are large and badly conditioned, and
    runs diverge. Let CE establish a coarse solution first, then switch.

    Call `set_epoch(e)` once per epoch from the training loop.
    """

    def __init__(self, alpha=0.7, gamma=1.3333, lam_boundary=0.5,
                 warmup_epochs=3, class_weights=(0.2, 1.0, 1.0),
                 boundary_width=3):
        super().__init__()
        self.warmup_epochs = warmup_epochs
        self.epoch = 0
        self.class_weights = class_weights
        self.ftl = FocalTverskyLoss(alpha=alpha, gamma=gamma,
                                    class_weights=class_weights)
        self.bnd = BoundaryWeightedCE(lam=2.0, width=boundary_width,
                                      class_weights=class_weights)
        self.lam_boundary = lam_boundary

    def set_epoch(self, e):
        self.epoch = e

    def _ramp(self):
        """Ramp the boundary term in over 3 epochs after warm-up ends."""
        if self.epoch < self.warmup_epochs:
            return 0.0
        return self.lam_boundary * min(1.0, (self.epoch - self.warmup_epochs + 1) / 3.0)

    def forward(self, logits, target, weight=None):
        if self.epoch < self.warmup_epochs:
            cw = torch.as_tensor(self.class_weights, dtype=logits.dtype,
                                 device=logits.device)
            ce = F.cross_entropy(logits, target.long(), weight=cw, reduction="none")
            if weight is not None:
                return (ce * weight).sum() / (weight.sum() + EPS)
            return ce.mean()
        lam = self._ramp()
        loss = self.ftl(logits, target, weight)
        if lam > 0:
            loss = loss + lam * self.bnd(logits, target, weight)
        return loss


def build_loss(name, **kw):
    """Factory, so configs/exp_*.yaml can name a loss as a string (ablation A2)."""
    name = name.lower()
    if name in ("ce", "crossentropy"):
        cw = kw.get("class_weights", (0.2, 1.0, 1.0))

        class _CE(nn.Module):
            def __init__(self):
                super().__init__()
                self.cw = cw

            def set_epoch(self, e):
                pass

            def forward(self, logits, target, weight=None):
                w = torch.as_tensor(self.cw, dtype=logits.dtype, device=logits.device)
                ce = F.cross_entropy(logits, target.long(), weight=w, reduction="none")
                if weight is not None:
                    return (ce * weight).sum() / (weight.sum() + EPS)
                return ce.mean()
        return _CE()
    if name == "dice":
        return TverskyLoss(alpha=0.5, beta=0.5, **kw)        # symmetric
    if name == "tversky":
        return TverskyLoss(**kw)
    if name in ("focal_tversky", "ftl"):
        return FocalTverskyLoss(**kw)
    if name == "combo":
        return ComboLoss(**kw)
    raise ValueError(f"unknown loss '{name}'")


# ===========================================================================
# Self-test
# ===========================================================================
def _make_case(H=64, W=64, timid_frac=0.3, seed=0):
    """Target: a 4-px erosion band. Prediction: only part of it — the failure."""
    rng = np.random.default_rng(seed)
    target = np.zeros((H, W), np.int64)
    target[30:34] = EROSION
    target[20:21] = ACCRETION

    n_rows = max(1, int(round(4 * timid_frac)))
    prob = np.full((3, H, W), 0.02)
    prob[STABLE] = 0.96
    prob[EROSION, 30:30 + n_rows] = 0.90
    prob[STABLE, 30:30 + n_rows] = 0.08
    prob += rng.normal(0, 0.005, prob.shape)
    prob = np.clip(prob, 1e-4, None)
    prob /= prob.sum(0, keepdims=True)
    return prob, target


def _onehot(t, C=3):
    return np.stack([(t == c).astype(float) for c in range(C)])


def _self_test():
    print("=" * 70)
    print("LOSSES SELF-TEST")
    print("=" * 70)

    prob, target = _make_case(timid_frac=0.3)
    g = _onehot(target)

    # --- 1. the asymmetry does what it claims ------------------------------
    print("\n[1] Raising ALPHA must penalise the under-predicting model more")
    print("    (prediction covers 30% of the true erosion band)\n")
    print(f"    {'alpha':>7}{'beta':>7}{'erosion TI':>13}{'FTL':>10}   note")
    prev = None
    for a in (0.5, 0.6, 0.7, 0.8, 0.9):
        ti = tversky_index_np(prob, g, alpha=a, beta=1 - a)
        ftl = focal_tversky_np(prob, g, alpha=a, beta=1 - a, gamma=1.3333)
        note = "Dice (symmetric)" if a == 0.5 else ("<- default" if a == 0.7 else "")
        print(f"    {a:>7.1f}{1-a:>7.1f}{ti[EROSION]:>13.4f}{ftl:>10.4f}   {note}")
        if prev is not None:
            assert ti[EROSION] < prev, "erosion TI must FALL as alpha rises"
        prev = ti[EROSION]
    print("\n    PASS — the loss gets strictly worse as alpha rises, which is")
    print("    exactly the pressure that pushes the model to predict MORE erosion.")

    # --- 2. class weighting matters as much as alpha -----------------------
    print("\n[2] Averaging over classes DILUTES the erosion signal")
    print("    Unweighted mean vs erosion-class-only, across coverage levels\n")
    print(f"    {'coverage':>10}{'mean, Dice':>13}{'mean, a=.9':>13}"
          f"{'ero, Dice':>12}{'ero, a=.9':>12}")
    rows = []
    for frac in (0.25, 0.5, 0.75, 1.0):
        p, t = _make_case(timid_frac=frac)
        gg = _onehot(t)
        md = focal_tversky_np(p, gg, alpha=0.5, beta=0.5, gamma=1.0)
        ms = focal_tversky_np(p, gg, alpha=0.9, beta=0.1, gamma=1.0)
        ed = 1.0 - tversky_index_np(p, gg, 0.5, 0.5)[EROSION]
        es = 1.0 - tversky_index_np(p, gg, 0.9, 0.1)[EROSION]
        rows.append((frac, md, ms, ed, es))
        print(f"    {frac:>9.0%}{md:>13.4f}{ms:>13.4f}{ed:>12.4f}{es:>12.4f}")

    sp = lambda i: rows[0][i] - rows[-1][i]                      # noqa: E731
    print(f"\n    spread across coverage:")
    print(f"      unweighted mean, Dice     {sp(1):.4f}")
    print(f"      unweighted mean, a=0.9    {sp(2):.4f}   ({sp(2)/sp(1):.2f}x)")
    print(f"      EROSION CLASS, Dice       {sp(3):.4f}")
    print(f"      EROSION CLASS, a=0.9      {sp(4):.4f}   ({sp(4)/sp(3):.2f}x)")

    assert sp(2) > sp(1), "asymmetric loss must separate coverage levels more"
    assert sp(4) > sp(1) * 1.5, "erosion-only signal should be far stronger"

    print("""
    PASS — and note the size of the effect. Averaging the loss equally over
    three classes buries the erosion signal under the stable class, which is
    ~99% of pixels and easy to fit. THAT is why ComboLoss defaults to
    class_weights=(0.2, 1.0, 1.0): down-weight stable so alpha can actually
    do its job. Alpha alone, with unweighted class averaging, will not fix
    the underprediction. Report both settings in ablation A2.""")

    # --- 3. torch matches numpy -------------------------------------------
    print("\n[3] Torch implementation vs the numpy reference")
    if not HAS_TORCH:
        print("    SKIPPED — torch not installed.")
        print("    The maths above is verified in numpy and the torch code is a")
        print("    direct transcription. Re-run this on Colab to confirm.")
    else:
        p_t = torch.tensor(prob, dtype=torch.float32).unsqueeze(0)
        t_t = torch.tensor(target).unsqueeze(0)
        logits = torch.log(p_t.clamp_min(1e-8))          # softmax(log p) == p
        for a in (0.5, 0.7, 0.9):
            ref = focal_tversky_np(prob, g, alpha=a, beta=1 - a, gamma=1.0)
            got = TverskyLoss(alpha=a, gamma=1.0)(logits, t_t).item()
            print(f"    alpha={a}:  numpy {ref:.6f}   torch {got:.6f}   "
                  f"diff {abs(ref-got):.2e}")
            assert abs(ref - got) < 1e-4, "torch and numpy disagree"
        print("    PASS")

        print("\n[4] Gradients flow and shapes are right")
        lg = torch.randn(2, 3, 64, 64, requires_grad=True)
        tg = torch.randint(0, 3, (2, 64, 64))
        wg = torch.ones(2, 64, 64)
        for nm in ("ce", "dice", "tversky", "focal_tversky", "combo"):
            fn = build_loss(nm)
            if hasattr(fn, "set_epoch"):
                fn.set_epoch(10)
            v = fn(lg, tg, wg)
            v.backward(retain_graph=True)
            assert torch.isfinite(v), f"{nm} produced non-finite loss"
            print(f"    {nm:<16}loss {v.item():8.4f}   grad ok")
        print("    PASS")

    print(f"\n{'='*70}")
    print("""FOR THE PAPER (ablation A2)

Sweep alpha over {0.5, 0.6, 0.7, 0.8, 0.9} and plot M1 eroded-area bias
against alpha, with a horizontal line at 1.0. Mark where Dice (alpha=0.5)
sits. That single figure is the clearest statement of contribution C2 you
can make, and it is figure F3 in the outline.""")
    print("=" * 70)
    print("ALL PASS.")


if __name__ == "__main__":
    _self_test()
