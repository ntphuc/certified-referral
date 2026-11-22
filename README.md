# Certified referral for thin segmentation networks

Code and results for the paper *"Certified referral exposes the hidden reliability cost of thin
segmentation networks"* (N. T. Phuc, K. V. Nguyen, H. M. Tan; submitted to ESANN 2027).

A segmentation model signs off on an image when its uncertainty score is below a threshold and
refers the rest to a clinician. We certify the threshold with Learn-then-Test (fixed-sequence
exact binomial tests) so that, with probability at least 1 - delta, at most a fraction alpha of
the accepted images are failures (Dice < tau). Certified coverage is split into an accuracy
bound C* = min(1, (1 - p)/(1 - alpha)) and an efficiency eta = C_hat / C*. We apply this to
2D U-Nets of base width 4 to 64 (0.12M to 31M parameters) on BRISC2025 and Kvasir-SEG.

## Contents

| Path | What it is |
|---|---|
| `certref/certify.py` | Grid, LTT certification, oracle bound, naive baseline, repeated-split evaluation |
| `certref/test_certify.py`, `certref/sim_iid.py` | Synthetic checks; i.i.d. validity simulation |
| `train_eval.py` | Train one width/seed U-Net, export per-image Dice and uncertainty scores |
| `ensemble_scores.py` | K=3 deep-ensemble disagreement from the three seeds of each width |
| `analyze.py` | Pooled tables and figures from all runs |
| `grid_floor.py`, `sensitivity.py` | Grid-floor, delta, grid-density and calibration-size analyses |
| `make_paper_assets.py` | Table 1 and Figure 1 of the paper |
| `outputs*/w{w}_s{seed}/` | Per-image CSVs, training logs and summaries for every run (no checkpoints) |
| `results*/` | Summary CSVs, `RESULTS.md`, analysis outputs |

All numbers in the paper come from `results*/` and `outputs*/`; the analyses can be rerun
without a GPU from the per-image CSVs.

## Reproduce the analysis (CPU)

```bash
pip install numpy scipy pandas matplotlib
python certref/test_certify.py
python analyze.py --runs outputs --out results --min-coverage 0.2
python analyze.py --runs outputs_kvasir --out results_kvasir --min-coverage 0.3
python make_paper_assets.py --tau 0.5 --alpha 0.05 --paper paper_assets
python grid_floor.py > results/grid_floor.txt
python sensitivity.py > results/sensitivity.txt
```

## Retrain (GPU)

Requires PyTorch, opencv-python, pandas and matplotlib.

- BRISC2025: see Fateh et al., arXiv:2506.14318, for the official release; extract it and set
  `DATA_ROOT` to the folder containing `segmentation_task/`.
- Kvasir-SEG: download from the official Simula page (Jha et al., MMM 2020); set `DATA_ROOT` to
  the folder containing `images/` and `masks/`.

```bash
DATA_ROOT=/path/to/brisc2025 IDX=0 bash RUN.sh                 # one width/seed (IDX 0..14)
DATASET=kvasir DATA_ROOT=/path/to/kvasir-seg IDX=0 bash RUN.sh
```

The `*.slurm` files are the array jobs we used; their `--account`, `--partition` and `--gres`
lines are specific to our cluster and must be edited for yours.

## Protocol

Fixed in advance: tau in {0.5, 0.7}, alpha in {0.05, 0.10}, delta = 0.1, a 50-point grid of
validation-score quantiles starting at 20% (BRISC2025) or 30% (Kvasir-SEG) coverage, and 500
random halvings of the test set (certify on one half, measure on the other).

Added after the main results had been seen, and labelled as such in the paper: the headline
setting (tau = 0.5, alpha = 0.05), validity on the full test set, the i.i.d. simulation, the
naive baseline, the oracle ceiling eta*, the ensemble scores, and the grid-floor and
sensitivity analyses.
