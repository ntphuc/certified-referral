"""Sensitivity of certified coverage to delta, grid density and calibration size,
plus failure breakdown by lesion size / tumor type. Headline setting tau=0.5, alpha=0.05.
Added after the main results (reviewer request). Needs numpy only.

python sensitivity.py > results/sensitivity.txt
"""
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "certref"))
from certify import make_grid, repeated_split_eval  # noqa: E402

TAU, ALPHA = 0.5, 0.05
SCORES = ["dhat", "mc_mi", "ent_fg", "oracle"]
DATASETS = [("BRISC2025", "outputs", 0.2), ("Kvasir-SEG", "outputs_kvasir", 0.3)]
VARIANTS = [("base: delta=0.1, 50 pts, cal=0.5", dict(delta=0.1, n_grid=50, cal=0.5)),
            ("delta=0.05", dict(delta=0.05, n_grid=50, cal=0.5)),
            ("grid 25 pts", dict(delta=0.1, n_grid=25, cal=0.5)),
            ("grid 100 pts", dict(delta=0.1, n_grid=100, cal=0.5)),
            ("cal=0.7 (larger)", dict(delta=0.1, n_grid=50, cal=0.7)),
            ("cal=0.3 (smaller)", dict(delta=0.1, n_grid=50, cal=0.3))]


def load(path):
    rows = list(csv.DictReader(open(path)))
    out = {k: np.array([float(r[k]) for r in rows]) for k in
           ["dice", "gt_px", "dhat", "mc_mi", "ent_fg"]}
    out["oracle"] = 1.0 - out["dice"]
    out["tumor"] = np.array([r["tumor"] for r in rows])
    return out


def main(splits=200):
    for ds, root, minc in DATASETS:
        print(f"\n=== {ds}  (tau={TAU}, alpha={ALPHA}; mean over 3 seeds; {splits} splits)")
        print("variant                          w   " + "  ".join(f"{s:>14}" for s in SCORES)
              + "   max pop-viol")
        for name, v in VARIANTS:
            for w in (4, 16, 64):
                cov = {s: [] for s in SCORES}; pv = []
                for seed in (2026, 2027, 2028):
                    d = Path(root) / f"w{w}_s{seed}"
                    val, te = load(d / "per_case_val.csv"), load(d / "per_case_test.csv")
                    for s in SCORES:
                        g = make_grid(val[s], n_grid=v["n_grid"], min_coverage=minc)
                        r = repeated_split_eval(te[s], te["dice"], TAU, ALPHA, v["delta"], g,
                                                n_splits=splits, cal_frac=v["cal"], seed=seed)
                        cov[s].append(r["coverage_mean"]); pv.append(r["pop_violation_rate"])
                print(f"{name:32s} {w:2d}  " + "  ".join(f"{np.mean(cov[s]):14.2f}" for s in SCORES)
                      + f"   {max(pv):.3f}")

        print(f"\n--- {ds}: failures (Dice<{TAU}) by group, w=64 and w=4 pooled over seeds")
        for w in (4, 64):
            te = [load(Path(root) / f"w{w}_s{s}" / "per_case_test.csv") for s in (2026, 2027, 2028)]
            dice = np.concatenate([t["dice"] for t in te]); gt = np.concatenate([t["gt_px"] for t in te])
            tum = np.concatenate([t["tumor"] for t in te])
            fail = dice < TAU
            cuts = np.quantile(gt, [1 / 3, 2 / 3])
            grp = np.digitize(gt, cuts)
            parts = [f"size tercile {k}: fail {fail[grp == k].mean():.3f} (n={int((grp == k).sum())})"
                     for k in range(3)]
            print(f"w={w}: " + "; ".join(parts) + f"; share of failures in smallest tercile "
                  f"{(fail & (grp == 0)).sum() / max(fail.sum(), 1):.2f}")
            for t in sorted(set(tum)):
                m = tum == t
                print(f"   {t:>12s}: fail {fail[m].mean():.3f} (n={int(m.sum())})")


if __name__ == "__main__":
    main()
