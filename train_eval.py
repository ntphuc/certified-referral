"""Train one width/seed U-Net on BRISC2025, then export per-case reliability data.

Outputs in <out>/w{width}_s{seed}/:
  best.pt            best-validation-Dice checkpoint
  train_log.csv      per-epoch loss / val Dice
  per_case_val.csv   per-image Dice + uncertainty scores on the validation split
  per_case_test.csv  same on the official test split
  summary.json       params, MACs, temperature, mean Dice, pixel ECE/Brier

The prediction (and hence Dice) is always the deterministic pass at 0.5;
TTA and MC-Dropout only produce uncertainty scores, so every score is ranked
against the same masks. Scores are oriented so that higher = less trustworthy.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.data import ArrayDataset, load_split
from src.unet import UNet2D, count_macs, count_params

LN2 = math.log(2.0)
EPS = 1e-6


def norm(x: torch.Tensor) -> torch.Tensor:
    return (x - 0.5) / 0.25


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def loss_fn(logits, y):
    bce = F.binary_cross_entropy_with_logits(logits, y)
    p = torch.sigmoid(logits)
    inter = (p * y).sum((1, 2, 3))
    dice = 1 - (2 * inter + 1) / (p.sum((1, 2, 3)) + y.sum((1, 2, 3)) + 1)
    return 0.5 * bce + 0.5 * dice.mean()


def dice_hard(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Per-image Dice of boolean masks (N,1,H,W); both empty -> 1."""
    a, b = a.flatten(1).float(), b.flatten(1).float()
    inter = (a * b).sum(1)
    s = a.sum(1) + b.sum(1)
    return torch.where(s > 0, 2 * inter / s.clamp_min(1), torch.ones_like(s))


def entropy(p: torch.Tensor) -> torch.Tensor:
    p = p.clamp(EPS, 1 - EPS)
    return -(p * p.log() + (1 - p) * (1 - p).log())


def region_mean(v: torch.Tensor, r: torch.Tensor, empty_value: float) -> torch.Tensor:
    num = (v * r).flatten(1).sum(1)
    den = r.flatten(1).sum(1)
    return torch.where(den > 0, num / den.clamp_min(1), torch.full_like(num, empty_value))


@torch.no_grad()
def predict_logits(model, x):
    model.eval()
    model.set_mc_dropout(False)
    return model(norm(x)).float()


@torch.no_grad()
def collect_logits(model, loader, device, max_pixels=4_000_000, seed=0):
    """Random subsample of (logit, label) pixels for temperature fitting."""
    g = torch.Generator().manual_seed(seed)
    zs, ys = [], []
    per_batch = None
    for x, y in loader:
        z = predict_logits(model, x.to(device)).cpu().flatten()
        yy = y.flatten()
        if per_batch is None:
            per_batch = max(1, max_pixels // max(1, len(loader)))
        idx = torch.randint(0, z.numel(), (min(per_batch, z.numel()),), generator=g)
        zs.append(z[idx]); ys.append(yy[idx])
    return torch.cat(zs), torch.cat(ys)


def fit_temperature(z: torch.Tensor, y: torch.Tensor) -> float:
    temps = torch.exp(torch.linspace(math.log(0.25), math.log(4.0), 81))
    nll = torch.stack([F.binary_cross_entropy_with_logits(z / t, y) for t in temps])
    return float(temps[int(nll.argmin())])


class ECEAccumulator:
    def __init__(self, bins=15):
        self.bins = bins
        self.cnt = np.zeros(bins); self.conf = np.zeros(bins); self.acc = np.zeros(bins)
        self.brier = 0.0; self.n = 0

    def add(self, p: torch.Tensor, y: torch.Tensor):
        p = p.flatten().double(); y = y.flatten().double()
        b = (p * self.bins).long().clamp_max(self.bins - 1)
        self.cnt += torch.bincount(b, minlength=self.bins).cpu().numpy()
        self.conf += torch.bincount(b, weights=p, minlength=self.bins).cpu().numpy()
        self.acc += torch.bincount(b, weights=y, minlength=self.bins).cpu().numpy()
        self.brier += float(((p - y) ** 2).sum()); self.n += p.numel()

    def result(self):
        m = self.cnt > 0
        ece = float((np.abs(self.conf[m] - self.acc[m])).sum() / self.n)
        return ece, self.brier / self.n


@torch.no_grad()
def evaluate_split(model, x, y, meta, temp, device, batch, mc_passes, dil_px):
    loader = DataLoader(ArrayDataset(x, y), batch_size=batch, shuffle=False, num_workers=2)
    ece_raw, ece_ts = ECEAccumulator(), ECEAccumulator()
    rows, i0 = [], 0
    k = 2 * dil_px + 1
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        z = predict_logits(model, xb)
        p_raw = torch.sigmoid(z)
        p = torch.sigmoid(z / temp)
        pred = z > 0
        gt = yb > 0.5
        ece_raw.add(p_raw, yb); ece_ts.add(p, yb)

        dice = dice_hard(pred, gt)
        inter = (pred & gt).flatten(1).sum(1).float()
        union = (pred | gt).flatten(1).sum(1).float()
        iou = torch.where(union > 0, inter / union.clamp_min(1), torch.ones_like(union))

        # Region of interest: dilated (prediction OR p>0.25). Empty region means the
        # model sees no lesion at all; scores then take their maximum so such cases
        # are referred (every BRISC segmentation image contains a tumor).
        seed_r = (pred | (p > 0.25)).float()
        reg = F.max_pool2d(seed_r, k, stride=1, padding=dil_px) > 0
        ent = entropy(p)

        s_ent_img = ent.flatten(1).mean(1)
        s_ent_fg = region_mean(ent, reg.float(), LN2)
        # Estimated Dice from calibrated probabilities (expected TP/FP/FN).
        pf = p.flatten(1); hard = pred.flatten(1).float()
        tp = (hard * pf).sum(1); fp = hard.sum(1) - tp; fn = ((1 - hard) * pf).sum(1)
        dhat = torch.where(hard.sum(1) + fn > 0.5, 2 * tp / (2 * tp + fp + fn).clamp_min(EPS),
                           torch.zeros_like(tp))
        s_dhat = 1 - dhat

        # Test-time augmentation: flips; disagreement with the deterministic mask.
        tta_d = []
        for dims in ((3,), (2,), (2, 3)):
            zt = predict_logits(model, xb.flip(dims)).flip(dims)
            tta_d.append(dice_hard(zt > 0, pred))
        s_tta = 1 - torch.stack(tta_d).mean(0)

        # MC-Dropout: decoder dropout on, BatchNorm in eval mode.
        model.eval(); model.set_mc_dropout(True)
        ps, mc_d = [], []
        for _ in range(mc_passes):
            zm = model(norm(xb)).float()
            ps.append(torch.sigmoid(zm / temp)); mc_d.append(dice_hard(zm > 0, pred))
        model.set_mc_dropout(False)
        ps = torch.stack(ps)
        mi = entropy(ps.mean(0)) - entropy(ps).mean(0)
        s_mc_mi = region_mean(mi, reg.float(), LN2)
        s_mc_dis = 1 - torch.stack(mc_d).mean(0)

        cols = dict(dice=dice, iou=iou, gt_px=gt.flatten(1).sum(1).float(),
                    pred_px=pred.flatten(1).sum(1).float(), dhat_est=dhat,
                    ent_img=s_ent_img, ent_fg=s_ent_fg, dhat=s_dhat, tta_dis=s_tta,
                    mc_mi=s_mc_mi, mc_dis=s_mc_dis)
        cols = {c: v.cpu().numpy() for c, v in cols.items()}
        for j in range(len(xb)):
            r = {"case_id": meta["case_id"][i0 + j], "tumor": meta["tumor_label"][i0 + j],
                 "plane": meta["plane_label"][i0 + j]}
            r.update({c: float(v[j]) for c, v in cols.items()})
            rows.append(r)
        i0 += len(xb)
    e_raw, b_raw = ece_raw.result()
    e_ts, b_ts = ece_ts.result()
    return rows, {"ece_raw": e_raw, "brier_raw": b_raw, "ece_ts": e_ts, "brier_ts": b_ts}


def write_csv(path: Path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--dataset", default="brisc", choices=["brisc", "kvasir"])
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--cache-dir", default="cache")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--mc-passes", type=int, default=10)
    ap.add_argument("--dil-px", type=int, default=5)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-hours", type=float, default=10.0)
    ap.add_argument("--limit", type=int, default=0, help="debug: cap samples per split")
    args = ap.parse_args()

    t_start = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    run = Path(args.out) / f"w{args.width}_s{args.seed}"
    run.mkdir(parents=True, exist_ok=True)
    if (run / "summary.json").exists():
        print(f"[skip] {run} already finished"); return

    data = {s: load_split(args.manifest, s, args.data_root, args.size, args.cache_dir,
                          dataset=args.dataset)
            for s in ("train", "val", "test")}
    if args.limit:
        data = {s: (x[:args.limit], y[:args.limit], {k: v[:args.limit] for k, v in m.items()})
                for s, (x, y, m) in data.items()}
    print({s: len(v[0]) for s, v in data.items()})

    set_seed(args.seed)
    in_ch = 1 if data["train"][0].ndim == 3 else 3
    model = UNet2D(args.width, in_ch=in_ch, dropout_p=args.dropout).to(device)
    n_params, macs = count_params(model), count_macs(model, (1, in_ch, args.size, args.size))
    print(f"width={args.width} params={n_params} MACs={macs}")

    g = torch.Generator().manual_seed(args.seed)
    tr_loader = DataLoader(ArrayDataset(data["train"][0], data["train"][1], augment=True),
                           batch_size=args.batch, shuffle=True, drop_last=True, generator=g,
                           num_workers=args.workers, persistent_workers=args.workers > 0,
                           worker_init_fn=lambda i: (random.seed(args.seed * 100 + i),
                                                     np.random.seed(args.seed * 100 + i)))
    va_loader = DataLoader(ArrayDataset(data["val"][0], data["val"][1]), batch_size=args.batch,
                           shuffle=False, num_workers=2)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, epochs=args.epochs,
                                                steps_per_epoch=len(tr_loader), pct_start=0.05)
    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best, best_ep, log = -1.0, -1, []
    for ep in range(args.epochs):
        model.train(); tl, nb = 0.0, 0
        for xb, yb in tr_loader:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=use_amp):
                loss = loss_fn(model(norm(xb)), yb)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
            tl += float(loss); nb += 1
        dices = []
        with torch.no_grad():
            for xb, yb in va_loader:
                z = predict_logits(model, xb.to(device))
                dices.append(dice_hard(z > 0, yb.to(device) > 0.5).cpu())
        vd = float(torch.cat(dices).mean())
        log.append({"epoch": ep, "train_loss": tl / max(nb, 1), "val_dice": vd,
                    "minutes": (time.time() - t_start) / 60})
        print(f"ep {ep:03d} loss {tl / max(nb, 1):.4f} val_dice {vd:.4f}", flush=True)
        if vd > best + 1e-4:
            best, best_ep = vd, ep
            torch.save(model.state_dict(), run / "best.pt")
        if ep - best_ep >= args.patience:
            print("[early stop]"); break
        if (time.time() - t_start) / 3600 > args.max_hours:
            print("[time limit]"); break
    write_csv(run / "train_log.csv", log)

    model.load_state_dict(torch.load(run / "best.pt", map_location=device))
    z, yv = collect_logits(model, va_loader, device, seed=args.seed)
    temp = fit_temperature(z, yv)
    print(f"temperature={temp:.3f}")

    summary = {"dataset": args.dataset, "in_ch": in_ch, "width": args.width, "seed": args.seed, "params": n_params, "macs": macs,
               "best_epoch": best_ep, "best_val_dice": best, "temperature": temp,
               "epochs_run": len(log), "train_minutes": log[-1]["minutes"] if log else 0}
    for split in ("val", "test"):
        x, y, meta = data[split]
        rows, cal = evaluate_split(model, x, y, meta, temp, device, args.batch,
                                   args.mc_passes, args.dil_px)
        write_csv(run / f"per_case_{split}.csv", rows)
        summary[f"{split}_dice_mean"] = float(np.mean([r["dice"] for r in rows]))
        summary.update({f"{split}_{k}": v for k, v in cal.items()})
    summary["total_minutes"] = (time.time() - t_start) / 60
    (run / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
