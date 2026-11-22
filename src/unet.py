"""Plain 2D U-Net whose only free parameter is the base width w.

Encoder widths w, 2w, 4w, 8w, 16w (four 2x poolings). Every width variant uses
the same topology, normalization, loss and training budget, so width is the
only factor that changes between models.

Decoder blocks carry Dropout2d(p). It acts as a regularizer during training
for every width alike and is switched on at inference only for MC-Dropout
scores; the reported prediction and Dice always come from the deterministic
pass.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def _block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
    )


class UNet2D(nn.Module):
    def __init__(self, width: int = 32, in_ch: int = 1, depth: int = 4, dropout_p: float = 0.1):
        super().__init__()
        ch = [width * 2 ** i for i in range(depth + 1)]
        self.enc = nn.ModuleList([_block(in_ch, ch[0])] + [_block(ch[i - 1], ch[i]) for i in range(1, depth + 1)])
        self.pool = nn.MaxPool2d(2)
        self.up = nn.ModuleList([nn.ConvTranspose2d(ch[i], ch[i - 1], 2, stride=2) for i in range(depth, 0, -1)])
        self.drop = nn.ModuleList([nn.Dropout2d(dropout_p) for _ in range(depth)])
        self.dec = nn.ModuleList([_block(2 * ch[i - 1], ch[i - 1]) for i in range(depth, 0, -1)])
        self.head = nn.Conv2d(ch[0], 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for i, e in enumerate(self.enc):
            x = e(x if i == 0 else self.pool(x))
            skips.append(x)
        x = skips.pop()
        for up, drop, dec in zip(self.up, self.drop, self.dec):
            x = up(x)
            x = dec(drop(torch.cat([x, skips.pop()], 1)))
        return self.head(x)

    def set_mc_dropout(self, on: bool) -> None:
        """Dropout on/off without touching BatchNorm (which stays in eval mode)."""
        for d in self.drop:
            d.train(on)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


@torch.no_grad()
def count_macs(model: nn.Module, shape=(1, 1, 256, 256)) -> int:
    """Multiply-accumulates of conv / transposed-conv layers for one input."""
    total = 0

    def hook(m, inp, out):
        nonlocal total
        if isinstance(m, nn.Conv2d):
            k = m.kernel_size[0] * m.kernel_size[1] * m.in_channels // m.groups
            total += out.numel() * k
        elif isinstance(m, nn.ConvTranspose2d):
            k = m.kernel_size[0] * m.kernel_size[1] * m.out_channels // m.groups
            total += inp[0].numel() * k

    hs = [m.register_forward_hook(hook) for m in model.modules()
          if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d))]
    was = model.training
    model.eval()
    model(torch.zeros(shape, device=next(model.parameters()).device))
    model.train(was)
    for h in hs:
        h.remove()
    return int(total)
