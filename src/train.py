#!/usr/bin/env python3
"""
src/train.py — training loop with the guardrails that decide this project.

Day 1, hour 18.

    # A1, your method — delta framing, asymmetric loss
    python src/train.py --reach STUB --tag A1_delta_a07 --loss combo --alpha 0.7

    # B3, JamUNet-style baseline — state framing, symmetric loss
    python src/train.py --reach STUB --tag B3_state_dice --framing state --loss dice

    # A2, the alpha sweep (playbook figure F3)
    for a in 0.5 0.6 0.7 0.8 0.9; do
      python src/train.py --reach STUB --tag A2_a$a --loss focal_tversky --alpha $a
    done

THREE GUARDRAILS, each enforced in code rather than left to discipline:

  1. Early stopping is on EROSION RECALL or M1 bias. Selecting on accuracy or
     overall F1 hands you the copy model — the exact failure this project
     exists to fix. `--select accuracy` is rejected outright.

  2. The test set is refused unless you pass --final. Playbook section 10.1:
     touch it once, Day 4 hour 16.

  3. Checkpoints and the CSV log are written every epoch. Free Colab will
     disconnect; assume it.
"""

import argparse
import csv
import json
import os
import sys
import time

# NOTE: torch, numpy and the project modules are imported INSIDE main(), after
# argument validation. Two reasons, both practical:
#   · `--help` and a rejected `--select` return instantly instead of waiting
#     ~5 s for torch to load.
#   · The guardrails below are testable on a machine without torch installed.

SELECTABLE = {
    "erosion_recall": ("max", "M2_erosion_recall"),
    "erosion_iou":    ("max", "M2_erosion_iou"),
    "bias_to_one":    ("min", "M1_eroded_area_bias"),   # |bias - 1|
    "change_csi":     ("max", "M3_change_restricted_csi"),
}
BANNED = {"accuracy", "acc", "f1", "whole_mask_f1", "loss"}


def validate(args):
    """
    Guardrail 1, enforced before any heavy import so it fails in milliseconds.

    Selecting checkpoints on accuracy or overall F1 is not a matter of taste
    here — those metrics are ~99% stable pixels, so they select the copy model
    and reproduce the exact failure this project exists to fix.
    """
    if args.select in BANNED:
        sys.exit(f"""
REFUSED: --select {args.select}

Accuracy and overall F1 are dominated by stable pixels (~99% of them). A model
that predicts 'no change' everywhere scores ~0.97 F1 while seeing zero erosion.
Selecting on them hands you the copy model — the exact failure this project
exists to fix.

Use one of: {list(SELECTABLE)}
""")
    if args.select not in SELECTABLE:
        sys.exit(f"--select must be one of {list(SELECTABLE)}, got '{args.select}'")
    if not 0.0 < args.alpha < 1.0:
        sys.exit(f"--alpha must be in (0, 1); got {args.alpha}. "
                 f"It weights FALSE NEGATIVES and alpha + beta = 1.")
    if args.framing == "state" and args.loss not in ("dice", "ce"):
        print(f"  note: --framing state with --loss {args.loss}. The B3 baseline "
              f"is meant to be SYMMETRIC (dice), so the A1 comparison isolates "
              f"the framing. Use --loss dice unless you mean something else.\n")
    return args


def set_seed(s):
    import random
    import numpy as np
    import torch
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def evaluate(model, loader, device, framing):
    """Accumulate predictions over a split and score with metrics.py."""
    import numpy as np
    import torch
    import metrics as M
    from dataset import EROSION

    model.eval()
    P, T, Wt = [], [], []
    with torch.no_grad():
        for x, y, w in loader:
            x = x.to(device, non_blocking=True)
            logits = model(x)
            if framing == "delta":
                pred = logits.argmax(1).cpu().numpy()
            else:
                # State framing predicts water[t+1]. Convert to a delta so BOTH
                # framings are scored by identical metrics — without this the
                # A1 comparison is meaningless.
                nxt = (torch.sigmoid(logits.squeeze(1)) > 0.5).cpu().numpy().astype(np.uint8)
                cur = x[:, -1].cpu().numpy().astype(np.uint8)   # last input year
                pred = np.zeros_like(nxt)
                pred[(cur == 0) & (nxt == 1)] = 1
                pred[(cur == 1) & (nxt == 0)] = 2
            P.append(pred.ravel())
            T.append(y.numpy().ravel())
            Wt.append(w.numpy().ravel())

    p = np.concatenate(P)
    t = np.concatenate(T)
    w = np.concatenate(Wt)
    m = w > 0
    p, t = p[m], t[m]

    ero = M.class_scores(p, t, cls=EROSION)
    return {
        "M1_eroded_area_bias": M.eroded_area_bias(p, t),
        "M2_erosion_iou": ero["iou"],
        "M2_erosion_recall": ero["recall"],
        "M2_erosion_precision": ero["precision"],
        "M3_change_restricted_csi": M.change_restricted_csi(p, t),
        "pixel_accuracy": float((p == t).mean()),
    }


def score(metrics_dict, key):
    mode, field = SELECTABLE[key]
    v = metrics_dict[field]
    if key == "bias_to_one":
        return -abs(v - 1.0)          # higher is better, peak at bias == 1
    return v if mode == "max" else -v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reach", default="STUB")
    ap.add_argument("--tag", required=True, help="experiment name; names all outputs")
    ap.add_argument("--framing", default="delta", choices=["delta", "state"],
                    help="delta = A1 (ours) · state = B3 (JamUNet-style)")
    ap.add_argument("--loss", default="combo",
                    choices=["ce", "dice", "tversky", "focal_tversky", "combo"])
    ap.add_argument("--alpha", type=float, default=0.7, help="weight on FALSE NEGATIVES")
    ap.add_argument("--gamma", type=float, default=1.3333)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--select", default="erosion_recall",
                    help=f"checkpoint criterion: {list(SELECTABLE)}")
    ap.add_argument("--final", action="store_true",
                    help="ALSO evaluate on test. Day 4 h16, once.")
    ap.add_argument("--cache", default=None)
    ap.add_argument("--out", default=None)
    args = validate(ap.parse_args())        # guardrail 1, before heavy imports

    import numpy as np
    import torch
    from torch.utils.data import DataLoader

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from dataset import ReachData, RiverWindows
    from losses import build_loss
    from models.unet import build_model
    import metrics as M

    try:
        from paths import get_cache_dir, get_results_dir, get_ckpt_dir
        cache = args.cache or get_cache_dir()
        results = args.out or get_results_dir()
        ckpt_dir = get_ckpt_dir()
    except ImportError:
        here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cache = args.cache or os.path.join(here, "data", "cache")
        results = args.out or os.path.join(here, "results")
        ckpt_dir = os.path.join(here, "checkpoints")
    for d in (results, ckpt_dir):
        os.makedirs(d, exist_ok=True)

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available()
                          else "cpu")

    reach = ReachData(cache, args.reach)
    train_ds = RiverWindows(reach, k=args.k, split="train", augment=True)
    val_ds = RiverWindows(reach, k=args.k, split="val")

    in_ch = train_ds.in_channels
    out_ch = 3 if args.framing == "delta" else 1
    model = build_model("unet", in_channels=in_ch, out_channels=out_ch).to(device)

    if args.framing == "delta":
        crit = build_loss(args.loss, **({"alpha": args.alpha, "gamma": args.gamma}
                                        if args.loss in ("tversky", "focal_tversky")
                                        else {"alpha": args.alpha, "gamma": args.gamma}
                                        if args.loss == "combo" else {}))
    else:
        crit = torch.nn.BCEWithLogitsLoss()

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                          num_workers=2, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=2)

    print(f"""{'='*70}
{args.tag}
{'='*70}
  reach      {args.reach}   {reach.meta['name']}
  rho        {reach.meta.get('rho_mean_annual_change', float('nan')):.4f}
  framing    {args.framing}      loss {args.loss}   alpha {args.alpha}
  device     {device}
  model      {model.n_params():,} params   in_ch {in_ch}  out_ch {out_ch}
  data       train {len(train_ds):,}  val {len(val_ds):,}
  select on  {args.select}   (accuracy is refused by design)
  seed       {args.seed}
{'='*70}""")

    log_path = os.path.join(results, f"train_{args.tag}.csv")
    fields = ["epoch", "train_loss", "M1_eroded_area_bias", "M2_erosion_recall",
              "M2_erosion_precision", "M2_erosion_iou",
              "M3_change_restricted_csi", "pixel_accuracy", "lr", "secs"]
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(fields)

    best, best_ep, bad = -np.inf, -1, 0
    for ep in range(args.epochs):
        t0 = time.time()
        model.train()
        if hasattr(crit, "set_epoch"):
            crit.set_epoch(ep)

        tot = n = 0
        for x, y, w in train_dl:
            x = x.to(device, non_blocking=True)
            w = w.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            if args.framing == "delta":
                loss = crit(logits, y.to(device), w)
            else:
                # state framing target: water[t+1], reconstructed from delta
                cur = x[:, -1]
                y_d = y.to(device)
                nxt = cur.clone()
                nxt[y_d == 1] = 1.0
                nxt[y_d == 2] = 0.0
                loss = crit(logits.squeeze(1), nxt)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item() * x.size(0)
            n += x.size(0)
        sched.step()

        vm = evaluate(model, val_dl, device, args.framing)
        s = score(vm, args.select)
        dt = time.time() - t0

        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([
                ep, tot / max(n, 1), vm["M1_eroded_area_bias"],
                vm["M2_erosion_recall"], vm["M2_erosion_precision"],
                vm["M2_erosion_iou"], vm["M3_change_restricted_csi"],
                vm["pixel_accuracy"], opt.param_groups[0]["lr"], round(dt, 1)])

        flag = ""
        if s > best:
            best, best_ep, bad = s, ep, 0
            torch.save({"epoch": ep, "model": model.state_dict(),
                        "optimizer": opt.state_dict(), "args": vars(args),
                        "val": vm},
                       os.path.join(ckpt_dir, f"{args.tag}_best.pt"))
            flag = "  *best"
        else:
            bad += 1

        print(f"  ep {ep:>3}  loss {tot/max(n,1):.4f}  "
              f"M1 {vm['M1_eroded_area_bias']:.3f}  "
              f"ero-recall {vm['M2_erosion_recall']:.3f}  "
              f"ero-prec {vm['M2_erosion_precision']:.3f}  "
              f"acc {vm['pixel_accuracy']:.4f}  {dt:.0f}s{flag}")

        if bad >= args.patience:
            print(f"  early stop — no improvement in {args.patience} epochs")
            break

    print(f"\n  best epoch {best_ep}, {args.select} = {abs(best):.4f}")

    ck = torch.load(os.path.join(ckpt_dir, f"{args.tag}_best.pt"),
                    map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])
    summary = {"tag": args.tag, "best_epoch": best_ep, "val": ck["val"],
               "args": vars(args)}

    # ---- guardrail 2 -------------------------------------------------------
    if args.final:
        print("\n  --final given: evaluating on TEST. This should happen once.")
        test_dl = DataLoader(RiverWindows(reach, k=args.k, split="test"),
                             batch_size=args.batch, shuffle=False, num_workers=2)
        tm = evaluate(model, test_dl, device, args.framing)
        summary["test"] = tm
        M.print_report(f"TEST · {args.tag}", {**tm,
                                              "M5_bankline_mean_m": float("nan"),
                                              "M8_whole_mask_f1": float("nan"),
                                              "M8_whole_mask_accuracy": float("nan")})
    else:
        print("\n  test set NOT touched (pass --final on Day 4 h16, once)")

    with open(os.path.join(results, f"summary_{args.tag}.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  wrote {log_path}")
    print(f"  wrote {os.path.join(results, f'summary_{args.tag}.json')}")

    b = ck["val"]["M1_eroded_area_bias"]
    print(f"""
{'-'*70}
  M1 eroded-area bias on validation: {b:.4f}
  {'UNDERPREDICTS erosion — raise --alpha and re-run' if b < 0.9 else
   'overpredicts — lower --alpha' if b > 1.1 else
   'well calibrated'}
{'-'*70}""")


if __name__ == "__main__":
    main()
