"""Certified-referral analysis over all finished width/seed runs.

Protocol (fixed before looking at test results):
  * failure:       Dice < tau, tau in {0.5, 0.7}
  * target:        selective risk alpha in {0.05, 0.10}, confidence 1 - delta, delta = 0.1
  * grid:          score quantiles on the VALIDATION split (never the test split)
  * calibration:   random halves of the official TEST split, 500 repetitions;
                   threshold certified on one half, evaluated on the other
  * decomposition: certified coverage = C_oracle(alpha) * eta

Writes results/ (CSV + Markdown + figures).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from certref.certify import aurc, make_grid, oracle_coverage, repeated_split_eval

def md(df, **kw):
    try:
        return df.to_markdown(**kw)
    except ImportError:  # tabulate not installed
        return "```\n" + df.to_string(index=kw.get("index", True)) + "\n```"


SCORES = ["ent_img", "ent_fg", "dhat", "tta_dis", "mc_mi", "mc_dis"]
# "oracle" = 1 - true Dice: the best possible ranking, run through the same
# certification, so eta_oracle is the attainable ceiling at this sample size.
ORACLE = "oracle"
ENS = ["ens_pw", "ens_dis"]  # from ensemble_scores.py (K=3 seeds), merged when present


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="outputs")
    ap.add_argument("--out", default="results")
    ap.add_argument("--taus", type=float, nargs="+", default=[0.5, 0.7])
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.05, 0.10])
    ap.add_argument("--delta", type=float, default=0.1)
    ap.add_argument("--splits", type=int, default=500)
    ap.add_argument("--min-coverage", type=float, default=0.2)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    rows, base = [], []
    for sj in sorted(Path(args.runs).glob("w*_s*/summary.json")):
        s = json.loads(sj.read_text())
        val = pd.read_csv(sj.parent / "per_case_val.csv")
        te = pd.read_csv(sj.parent / "per_case_test.csv")
        val[ORACLE] = 1.0 - val["dice"]; te[ORACLE] = 1.0 - te["dice"]
        extra = []
        if (sj.parent / "per_case_test_ens.csv").exists():
            for name, df in (("val", val), ("test", te)):
                e = pd.read_csv(sj.parent / f"per_case_{name}_ens.csv")
                assert list(e["case_id"]) == list(df["case_id"]), "ensemble row order"
                for c in ENS:
                    df[c] = e[c].values
            extra = ENS
        base.append({k: s[k] for k in ("width", "seed", "params", "macs", "temperature",
                                       "val_dice_mean", "test_dice_mean", "test_ece_raw",
                                       "test_ece_ts", "test_brier_ts")})
        for tau in args.taus:
            fails = (te["dice"].values < tau).astype(float)
            for score in SCORES + extra + [ORACLE]:
                a, ea = aurc(te[score].values, fails)
                grid = make_grid(val[score].values, n_grid=50, min_coverage=args.min_coverage)
                for alpha in args.alphas:
                    r = repeated_split_eval(te[score].values, te["dice"].values, tau, alpha,
                                            args.delta, grid, n_splits=args.splits,
                                            seed=s["seed"])
                    rows.append({"width": s["width"], "seed": s["seed"], "params": s["params"],
                                 "tau": tau, "alpha": alpha, "score": score,
                                 "fail_rate": r["fail_rate"],
                                 "oracle_cov": oracle_coverage(r["fail_rate"], alpha),
                                 "cert_cov": r["coverage_mean"], "eta": r["eta_mean"],
                                 "violation_rate": r["violation_rate"],
                                 "pop_violation_rate": r["pop_violation_rate"],
                                 "naive_cov": r["naive_coverage"],
                                 "naive_violation_rate": r["naive_violation_rate"],
                                 "naive_pop_violation_rate": r["naive_pop_violation_rate"],
                                 "risk_mean": float(np.mean(r["risk"])),
                                 "risk_q90": float(np.quantile(r["risk"], 0.9)),
                                 "aurc": a, "eaurc": ea})
        print(f"done {sj.parent.name}")
    if not rows:
        raise SystemExit(f"no finished runs under {args.runs}")

    b = pd.DataFrame(base); r = pd.DataFrame(rows)
    b.to_csv(out / "per_run_basic.csv", index=False)
    r.to_csv(out / "per_run_certified.csv", index=False)

    agg_b = b.groupby("width").agg(params=("params", "first"), macs=("macs", "first"),
                                   n_seeds=("seed", "count"),
                                   dice=("test_dice_mean", "mean"), dice_sd=("test_dice_mean", "std"),
                                   ece_raw=("test_ece_raw", "mean"), ece_ts=("test_ece_ts", "mean"))
    agg_r = (r.groupby(["tau", "alpha", "score", "width"])
              .agg(fail_rate=("fail_rate", "mean"), oracle_cov=("oracle_cov", "mean"),
                   cert_cov=("cert_cov", "mean"), cert_cov_sd=("cert_cov", "std"),
                   eta=("eta", "mean"), eta_sd=("eta", "std"),
                   violation=("violation_rate", "mean"),
                   pop_violation=("pop_violation_rate", "mean"),
                   naive_cov=("naive_cov", "mean"), naive_violation=("naive_violation_rate", "mean"),
                   naive_pop_violation=("naive_pop_violation_rate", "mean"), risk_mean=("risk_mean", "mean"),
                   risk_q90=("risk_q90", "mean"), eaurc=("eaurc", "mean"))
              .reset_index())
    agg_b.to_csv(out / "summary_basic.csv"); agg_r.to_csv(out / "summary_certified.csv", index=False)

    with open(out / "RESULTS.md", "w") as f:
        f.write("# Certified referral results (test split, repeated halves)\n\n")
        f.write(f"delta = {args.delta}; {args.splits} calibration/test splits per run; "
                "violation = seed-averaged fraction of splits whose realized test-half selective risk > alpha; "
                "it includes test-half sampling noise, so it can exceed delta even when the population risk is controlled; pop_violation = fraction of splits whose selective risk on the whole test set (the finite population the calibration half is drawn from) exceeds alpha.\n\n## Segmentation and calibration\n\n")
        f.write(md(agg_b, floatfmt=".4f") + "\n\n")
        for (tau, alpha), g in agg_r.groupby(["tau", "alpha"]):
            f.write(f"## tau = {tau}, alpha = {alpha}\n\n")
            f.write(md(g.drop(columns=["tau", "alpha"]), index=False, floatfmt=".3f") + "\n\n")
    plot(agg_b, agg_r, out)
    print(f"wrote {out}/RESULTS.md")


def plot(agg_b, agg_r, out: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib missing; skipping figures"); return
    params = agg_b["params"]
    for (tau, alpha), g in agg_r.groupby(["tau", "alpha"]):
        fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
        first = g[g.score == g.score.iloc[0]].set_index("width")
        ax[0].plot(params.loc[first.index], first["oracle_cov"], "k--", label="oracle")
        for score, gs in g.groupby("score"):
            gs = gs.set_index("width")
            ax[0].errorbar(params.loc[gs.index], gs["cert_cov"], yerr=gs["cert_cov_sd"],
                           marker="o", capsize=2, label=score)
            ax[1].plot(params.loc[gs.index], gs["eta"], marker="o", label=score)
        for a in ax:
            a.set_xscale("log"); a.set_xlabel("parameters"); a.grid(alpha=0.3)
        ax[0].set_ylabel(f"certified coverage (alpha={alpha})"); ax[1].set_ylabel("ranking efficiency eta")
        ax[0].legend(fontsize=7); fig.suptitle(f"tau={tau}")
        fig.tight_layout(); fig.savefig(out / f"coverage_tau{tau}_alpha{alpha}.pdf"); plt.close(fig)


if __name__ == "__main__":
    main()
