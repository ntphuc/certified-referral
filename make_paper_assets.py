"""Paper table + figure straight from results*/ CSVs (no hand-typed numbers).

python make_paper_assets.py --tau 0.5 --alpha 0.05 --paper ../paper
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

DATASETS = [("BRISC2025", "results"), ("Kvasir-SEG", "results_kvasir")]
LABEL = {"ent_img": "entropy (image)", "ent_fg": "entropy (region)", "dhat": r"$1-\hat D$",
         "tta_dis": "TTA disagreement", "mc_mi": "MC mutual info.", "mc_dis": "MC disagreement", "oracle": r"oracle score $1-D$",
         "ens_pw": "ensemble pairwise Dice", "ens_dis": "ensemble disagreement"}
TABLE_SCORES = ["dhat", "mc_mi", "ent_fg"]
SHORT = {"dhat": r"$1-\hat D$", "mc_mi": "MC-MI", "ent_fg": "region ent."}


def load(root: Path, name: str, tau: float, alpha: float):
    d = root / name
    if not (d / "summary_basic.csv").exists():
        return None, None
    b = pd.read_csv(d / "summary_basic.csv")
    c = pd.read_csv(d / "summary_certified.csv")
    c = c[(c.tau == tau) & (c.alpha == alpha)]
    return b, c


def fmt(x, nd=2):
    return "--" if pd.isna(x) else f"{x:.{nd}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--paper", default="../paper")
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--alpha", type=float, default=0.05)
    a = ap.parse_args()
    root, paper = Path(a.root), Path(a.paper)
    (paper / "figures").mkdir(parents=True, exist_ok=True)

    lines = [r"\begin{tabular}{@{}llrcccccc@{}}", r"\toprule",
             r" & $w$ & Params & Dice & $p$ & $C^\star$ & $\eta^\star$ & "
             + " & ".join(SHORT[s] for s in TABLE_SCORES) + r" \\",
             r"\midrule"]
    lines[0] = r"\begin{tabular}{@{}llrcccc" + "c" * (len(TABLE_SCORES) + 1) + "@{}}"
    panels = []
    for ds, name in DATASETS:
        b, c = load(root, name, a.tau, a.alpha)
        if b is None:
            continue
        panels.append((ds, b, c))
        for i, r in b.sort_values("width").iterrows():
            cw = c[c.width == r.width].set_index("score")
            first = cw.iloc[0] if len(cw) else None
            cells = [ds if r.width == b.width.min() else "", str(int(r.width)),
                     f"{r.params / 1e6:.2f}M",
                     f"{r.dice:.3f}" + (rf"$\pm${r.dice_sd:.3f}" if not pd.isna(r.dice_sd) else ""),
                     fmt(first.fail_rate if first is not None else float("nan"), 3),
                     fmt(first.oracle_cov if first is not None else float("nan")),
                     fmt(cw.eta.get("oracle"))]
            for s in TABLE_SCORES:
                cells.append(f"{fmt(cw.cert_cov.get(s))} / {fmt(cw.eta.get(s))}" if s in cw.index else "--")
            lines.append(" & ".join(cells) + r" \\")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    (paper / "table_main.tex").write_text("\n".join(lines) + "\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 7, "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(1, len(panels), figsize=(4.8, 1.6), sharey=True, squeeze=False)
    for ax, (ds, b, c) in zip(axes[0], panels):
        params = b.set_index("width")["params"]
        any_s = c[c.score == c.score.iloc[0]].set_index("width").sort_index()
        ax.plot(params.loc[any_s.index], any_s.oracle_cov, "k--", lw=1, label=r"oracle $C^\star$")
        for s, g in c.groupby("score"):
            g = g.set_index("width").sort_index()
            if s == "ens_dis":  # nearly identical to ens_pw; keep the figure readable
                continue
            if s == "oracle":
                ax.plot(params.loc[g.index], g.cert_cov, ":", color="0.4", lw=1.2, label=LABEL[s])
                continue
            ax.errorbar(params.loc[g.index], g.cert_cov, yerr=g.cert_cov_sd, marker="o", ms=2.5,
                        lw=0.9, capsize=1.5, label=LABEL[s])
        ax.set_xscale("log"); ax.set_title(ds); ax.set_xlabel("parameters"); ax.grid(alpha=0.3)
    axes[0][0].set_ylabel(rf"certified coverage ($\alpha$={a.alpha})")
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=5, fontsize=5, frameon=False)
    fig.tight_layout(pad=0.3, rect=(0, 0, 1, 0.78))
    fig.savefig(paper / "figures" / "coverage.pdf")
    print(f"wrote {paper/'table_main.tex'} and {paper/'figures/coverage.pdf'}")


if __name__ == "__main__":
    main()
