"""Deep-ensemble disagreement scores from the three seeds of each width (K=3).

Inference only: loads outputs*/w{w}_s{seed}/best.pt for the three seeds, predicts the
deterministic mask of each member on val and test, and writes per member
  per_case_{split}_ens.csv  with columns case_id, ens_pw, ens_dis
    ens_pw  = 1 - mean pairwise Dice over all K(K-1)/2 member pairs (same for every member)
    ens_dis = 1 - mean Dice between this member's mask and the other members' masks
Each member keeps its own mask and Dice, so failures are defined exactly as before.
Added after the main results (reviewer request).

python ensemble_scores.py --dataset brisc --runs outputs
"""
from __future__ import annotations

import argparse
import csv
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data import ArrayDataset, load_split
from src.unet import UNet2D
from train_eval import dice_hard, predict_logits


@torch.no_grad()
def masks(model, x, y, device, batch=32):
    out = []
    for xb, _ in DataLoader(ArrayDataset(x, y), batch_size=batch, shuffle=False, num_workers=2):
        out.append((predict_logits(model, xb.to(device)) > 0).cpu())
    return torch.cat(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="brisc", choices=["brisc", "kvasir"])
    ap.add_argument("--runs", default="outputs")
    ap.add_argument("--manifest", default="data/brisc2025_manifest.csv")
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--cache-dir", default="cache")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--widths", type=int, nargs="+", default=[4, 8, 16, 32, 64])
    ap.add_argument("--seeds", type=int, nargs="+", default=[2026, 2027, 2028])
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = {s: load_split(a.manifest, s, a.data_root, a.size, a.cache_dir, dataset=a.dataset)
            for s in ("val", "test")}
    in_ch = 1 if data["val"][0].ndim == 3 else 3
    for w in a.widths:
        runs = [Path(a.runs) / f"w{w}_s{s}" for s in a.seeds]
        pred = {split: [] for split in data}
        for r in runs:
            drop = json.loads((r / "summary.json").read_text()).get("dropout", 0.1)
            m = UNet2D(w, in_ch=in_ch, dropout_p=drop).to(device)
            m.load_state_dict(torch.load(r / "best.pt", map_location=device))
            for split, (x, y, _) in data.items():
                pred[split].append(masks(m, x, y, device))
        for split, (_, _, meta) in data.items():
            P = pred[split]; K = len(P)
            pair = {(i, j): dice_hard(P[i], P[j]).numpy() for i, j in combinations(range(K), 2)}
            ens_pw = 1 - np.mean(list(pair.values()), axis=0)
            for k, r in enumerate(runs):
                own = [pair[tuple(sorted((k, o)))] for o in range(K) if o != k]
                ens_dis = 1 - np.mean(own, axis=0)
                # Sanity check: row order must match the per-case file written by train_eval.
                ref = [row["case_id"] for row in csv.DictReader(open(r / f"per_case_{split}.csv"))]
                assert ref == list(meta["case_id"]), f"case order mismatch in {r}"
                with open(r / f"per_case_{split}_ens.csv", "w", newline="") as f:
                    wr = csv.writer(f); wr.writerow(["case_id", "ens_pw", "ens_dis"])
                    for cid, u, v in zip(meta["case_id"], ens_pw, ens_dis):
                        wr.writerow([cid, float(u), float(v)])
        print(f"done w{w}", flush=True)


if __name__ == "__main__":
    main()
