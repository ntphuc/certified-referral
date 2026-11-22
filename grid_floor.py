"""Does the grid floor (first threshold at 20% / 30% validation coverage) cost coverage?

Compares the headline floor with the lowest floor at which the first threshold can still
accept the 45 cases needed for alpha=0.05 (430 * 0.11 and 175 * 0.26 calibration cases),
and reports how often certification stops at the first grid point (zero coverage).
tau=0.5, alpha=0.05, delta=0.1, 200 splits. Added after the main results (reviewer request).

python grid_floor.py > results/grid_floor.txt
python grid_floor.py high > results/grid_floor_high.txt   (floors 0.4 / 0.5)
python grid_floor.py high ens / python grid_floor.py ens     (ensemble scores)
"""
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "certref"))
from certify import make_grid, repeated_split_eval  # noqa: E402

SCORES = ["ent_img", "ent_fg", "dhat", "tta_dis", "mc_mi", "mc_dis", "oracle"]
DATASETS = [("BRISC2025", "outputs", 0.2, 0.11), ("Kvasir-SEG", "outputs_kvasir", 0.3, 0.26)]
if len(sys.argv) > 1 and sys.argv[1] == "high":  # higher floors, same report layout
    DATASETS = [("BRISC2025", "outputs", 0.2, 0.4), ("Kvasir-SEG", "outputs_kvasir", 0.3, 0.5)]


if "ens" in sys.argv:  # ensemble scores only (ensemble_scores.py must have run)
    SCORES = ["ens_pw", "ens_dis", "oracle"]


def load(path):
    rows = list(csv.DictReader(open(path)))
    ens = Path(str(path).replace(".csv", "_ens.csv"))
    if ens.exists():
        for r, e in zip(rows, csv.DictReader(open(ens))):
            assert r["case_id"] == e["case_id"]; r.update(e)
    out = {k: np.array([float(r[k]) for r in rows]) for k in SCORES[:-1] + ["dice"]}
    out["oracle"] = 1.0 - out["dice"]
    return out


def main(splits=200):
    for ds, root, base, low in DATASETS:
        print(f"\n=== {ds}: floor {base} (headline) vs {low} (lowest testable)")
        print("w   score     cov@base  cov@low   diff   stop@1st base  stop@1st low")
        diffs = []
        for w in (4, 8, 16, 32, 64):
            for s in SCORES:
                c = {base: [], low: []}; z = {base: [], low: []}
                for seed in (2026, 2027, 2028):
                    d = Path(root) / f"w{w}_s{seed}"
                    val, te = load(d / "per_case_val.csv"), load(d / "per_case_test.csv")
                    for f in (base, low):
                        r = repeated_split_eval(te[s], te["dice"], 0.5, 0.05, 0.1,
                                                make_grid(val[s], 50, f), n_splits=splits, seed=seed)
                        c[f].append(r["coverage_mean"]); z[f].append(float((r["coverage"] == 0).mean()))
                cb, cl = np.mean(c[base]), np.mean(c[low]); diffs.append((cl - cb, w, s))
                print(f"{w:2d}  {s:8s}  {cb:8.3f}  {cl:8.3f}  {cl - cb:+.3f}  {np.mean(z[base]):13.3f}"
                      f"  {np.mean(z[low]):12.3f}")
        d = np.array([x[0] for x in diffs])
        print(f"{ds}: mean diff {d.mean():+.3f}, max gain {d.max():+.3f} ({max(diffs)[1:]}),"
              f" max loss {d.min():+.3f} ({min(diffs)[1:]})")


if __name__ == "__main__":
    main()
