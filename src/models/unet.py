#!/usr/bin/env python3
"""
src/models/unet.py — the U-Net. Deliberately unremarkable.

Day 1, hours 16-18.

    python src/models/unet.py       # shape and parameter-count check

ARCHITECTURE IS NOT A CONTRIBUTION OF THIS PROJECT, and treating it as one is
the most common way a student project burns a deadline for nothing. The
contributions are C1 (change-prediction reframing), C2 (asymmetric loss),
C3 (evaluation protocol), C4 (calibration), C5 (asset translation) and
C6 (the Padma Bridge experiment). This file exists to be a fair, competent,
FIXED backbone so those five can be measured without architecture confounding
the comparison.

If a judge asks why you did not use a transformer: your contribution is the
problem framing and the loss, and you held architecture constant deliberately.
That is a better answer than a marginal leaderboard gain.

Design choices, each with a reason:

  4 levels          256x256 tiles. Deeper wastes compute and overfits.
  base 32 channels  Inputs are BINARY MASKS, not RGB. Information content per
                    pixel is roughly a thousandth of a natural image, so a
                    large network is unnecessary and actively harmful here.
  GroupNorm         Small batches on a free-tier T4 make BatchNorm statistics
                    unstable across steps.
  no pretraining    ImageNet features are meaningless for binary water masks.
                    Training from scratch takes 10-20 minutes.
  3 output classes  stable / erosion / accretion  (DATA_CONTRACT.md section 3)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def gn(c):
    """GroupNorm with a group count that always divides the channel count."""
    return nn.GroupNorm(num_groups=min(8, c), num_channels=c)


class DoubleConv(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False), gn(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False), gn(cout), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    """
    in_channels : from RiverWindows.in_channels — do not hard-code it. With
                  k=4 water years plus distance and erosion-history aux it is 6,
                  but it changes the moment B delivers static aux channels.
    out_channels: 3 for the delta task; 1 for the state-framing baseline B3.
    """

    def __init__(self, in_channels=6, out_channels=3, base=32, depth=4,
                 dropout=0.0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depth = depth

        chs = [base * (2 ** i) for i in range(depth)]        # 32 64 128 256

        self.downs = nn.ModuleList()
        c = in_channels
        for ch in chs:
            self.downs.append(DoubleConv(c, ch))
            c = ch
        self.pool = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(chs[-1], chs[-1] * 2)
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

        self.ups = nn.ModuleList()
        self.upconvs = nn.ModuleList()
        c = chs[-1] * 2
        for ch in reversed(chs):
            self.upconvs.append(nn.ConvTranspose2d(c, ch, 2, stride=2))
            self.ups.append(DoubleConv(ch * 2, ch))
            c = ch

        self.head = nn.Conv2d(chs[0], out_channels, 1)
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        skips = []
        for d in self.downs:
            x = d(x)
            skips.append(x)
            x = self.pool(x)

        x = self.drop(self.bottleneck(x))

        for up, conv, skip in zip(self.upconvs, self.ups, reversed(skips)):
            x = up(x)
            if x.shape[-2:] != skip.shape[-2:]:       # odd input sizes
                x = F.interpolate(x, size=skip.shape[-2:], mode="nearest")
            x = conv(torch.cat([skip, x], dim=1))

        return self.head(x)

    def n_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_model(name="unet", **kw):
    if name.lower() == "unet":
        return UNet(**kw)
    if name.lower() in ("convlstm", "simvp", "simvpv2"):
        raise NotImplementedError(
            f"'{name}' comes from OpenSTL — Day 3, hours 5-9, with a HARD "
            f"90-minute stop on setup friction. Do not implement it yourself; "
            f"that is a full day you do not have. See playbook section 8.2.")
    raise ValueError(f"unknown model '{name}'")


if __name__ == "__main__":
    print("=" * 66)
    print("U-NET SHAPE CHECK")
    print("=" * 66)

    for ic, oc, tag in [(6, 3, "delta framing  (A1, your method)"),
                        (4, 1, "state framing  (B3, JamUNet-style)")]:
        m = build_model("unet", in_channels=ic, out_channels=oc)
        x = torch.randn(2, ic, 256, 256)
        with torch.no_grad():
            y = m(x)
        assert y.shape == (2, oc, 256, 256), f"bad output shape {y.shape}"
        print(f"\n  {tag}")
        print(f"    in  {tuple(x.shape)}")
        print(f"    out {tuple(y.shape)}")
        print(f"    params {m.n_params():,}  ({m.n_params()*4/1e6:.1f} MB fp32)")

    print("\n  gradient check")
    m = build_model("unet", in_channels=6, out_channels=3)
    x = torch.randn(2, 6, 256, 256, requires_grad=True)
    m(x).sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    print("    PASS — gradients finite")

    print("\n  odd input size (tiles at the reach edge)")
    with torch.no_grad():
        o = m(torch.randn(1, 6, 250, 194))
    print(f"    250x194 -> {tuple(o.shape[-2:])}  PASS")

    print("\n" + "=" * 66)
    print("""Next: src/train.py.

REMEMBER: early-stop on validation EROSION RECALL or M1 bias — never on
accuracy or overall F1. Both are dominated by stable pixels and will select
the copy model, recreating the exact failure this project exists to fix.""")
    print("=" * 66)
