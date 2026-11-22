"""Certified image-level referral for segmentation (Learn-then-Test, selective risk).

Setting
-------
Each case i has a scalar uncertainty score s_i (higher = less trustworthy) and a
failure label f_i = 1{Dice_i < tau}. The deployed rule accepts a case iff
s_i <= lam; rejected cases are referred to a human.

Selective risk:   R(lam) = P(f = 1 | s <= lam)
Coverage:         C(lam) = P(s <= lam)

We pick lam_hat on a calibration set so that, over the draw of that set,
    P( R(lam_hat) <= alpha ) >= 1 - delta.

Procedure (Angelopoulos et al., "Learn then Test", fixed-sequence testing):
candidate thresholds are ordered from most to least conservative. For each lam
the null H(lam): R(lam) > alpha is tested with the exact binomial p-value
    p(lam) = P( Bin(n_acc(lam), alpha) <= k(lam) ),
which is valid because, conditional on the number of accepted i.i.d. cases,
the failure count among them is Binomial(n_acc, R(lam)). Testing stops at the
first non-rejection; lam_hat is the last rejected threshold. Fixed-sequence
testing controls the family-wise error rate at delta without a correction.

Decomposition
-------------
With a perfect ranking (accept by true Dice) and population failure rate p,
the largest coverage with selective risk <= alpha is
    C_oracle(alpha) = min(1, (1 - p) / (1 - alpha)).
Certified coverage of a real score splits into an accuracy part (C_oracle,
set by the Dice distribution alone) and a ranking part
    eta = C_cert / C_oracle  in [0, 1],
so a smaller network can lose coverage because it fails more often (lower
C_oracle) or because its uncertainty ranks failures worse (lower eta).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import binom


@dataclass
class CertResult:
    lam: float            # accept iff score <= lam; -inf means accept nothing
    n_accepted: int
    n_failures: int
    p_value: float        # p-value of the last rejected null (nan if none)


def binom_pvalue(k: np.ndarray, n: np.ndarray, alpha: float) -> np.ndarray:
    """Exact p-value for H0: risk > alpha, given k failures in n accepted cases."""
    n = np.asarray(n)
    k = np.asarray(k)
    p = binom.cdf(k, n, alpha)
    return np.where(n > 0, p, 1.0)


def make_grid(reference_scores: np.ndarray, n_grid: int = 50,
              min_coverage: float = 0.1) -> np.ndarray:
    """Threshold grid from scores that are NOT the calibration set.

    The guarantee requires the grid to be fixed before calibration labels are
    seen; build it from training-set or other held-out scores. Thresholds are
    score quantiles at coverage levels min_coverage..1, ascending, so the
    fixed sequence starts at a conservative but testable point.
    """
    q = np.linspace(min_coverage, 1.0, n_grid)
    grid = np.quantile(np.asarray(reference_scores, dtype=float), q)
    grid[-1] = np.inf  # last candidate accepts everything
    return np.unique(grid)


def certify_threshold(scores: np.ndarray, fails: np.ndarray, alpha: float,
                      delta: float, grid: np.ndarray) -> CertResult:
    """Learn-then-Test threshold with P(selective risk <= alpha) >= 1 - delta.

    `grid` must be ascending (most conservative first) and independent of
    the calibration data (see make_grid).
    """
    scores = np.asarray(scores, dtype=float)
    fails = np.asarray(fails, dtype=int)
    grid = np.sort(np.asarray(grid, dtype=float))
    acc = scores[None, :] <= grid[:, None]
    n_acc = acc.sum(1)
    k_acc = (acc & (fails[None, :] == 1)).sum(1)
    pvals = binom_pvalue(k_acc, n_acc, alpha)
    rejected = pvals <= delta
    if not rejected[0]:
        return CertResult(-np.inf, 0, 0, float("nan"))
    # Fixed-sequence: stop at the first non-rejection.
    stop = int(np.argmin(rejected)) if not rejected.all() else len(rejected)
    j = stop - 1
    return CertResult(float(grid[j]), int(n_acc[j]), int(k_acc[j]), float(pvals[j]))


def oracle_coverage(fail_rate: float, alpha: float) -> float:
    """Largest coverage at selective risk <= alpha under a perfect ranking."""
    if fail_rate <= alpha:
        return 1.0
    return (1.0 - fail_rate) / (1.0 - alpha)


def selective_stats(scores: np.ndarray, fails: np.ndarray, lam: float):
    acc = np.asarray(scores) <= lam
    cov = float(acc.mean())
    risk = float(np.asarray(fails)[acc].mean()) if acc.any() else 0.0
    return cov, risk


def aurc(scores: np.ndarray, losses: np.ndarray) -> tuple[float, float]:
    """Area under risk-coverage curve and its excess over the oracle (E-AURC)."""
    losses = np.asarray(losses, dtype=float)
    n = len(losses)
    risk = np.cumsum(losses[np.argsort(scores, kind="mergesort")]) / np.arange(1, n + 1)
    best = np.cumsum(np.sort(losses)) / np.arange(1, n + 1)
    return float(risk.mean()), float(risk.mean() - best.mean())


def naive_threshold(scores, fails, alpha: float, grid: np.ndarray) -> float:
    """Largest grid threshold with empirical selective risk <= alpha (no guarantee)."""
    scores = np.asarray(scores, dtype=float); fails = np.asarray(fails)
    best = -np.inf
    for lam in grid:
        acc = scores <= lam
        if acc.any() and fails[acc].mean() <= alpha:
            best = lam
    return best


def repeated_split_eval(scores, dice, tau: float, alpha: float, delta: float,
                        grid: np.ndarray, n_splits: int = 500,
                        cal_frac: float = 0.5, seed: int = 0):
    """Random calibration/test splits: realized selective risk, coverage, eta.

    `grid` comes from make_grid on scores outside (scores, dice), e.g. train set.

    Returns a dict of per-split arrays plus summary numbers. `violation_rate` is
    the fraction of splits whose test selective risk exceeds alpha; it should be
    about <= delta (up to test-set sampling noise).
    """
    scores = np.asarray(scores, dtype=float)
    fails = (np.asarray(dice) < tau).astype(int)
    rng = np.random.default_rng(seed)
    n = len(scores)
    n_cal = int(round(cal_frac * n))
    cov, risk, eta, cov_or, pop_risk = (np.empty(n_splits) for _ in range(5))
    nv_cov, nv_risk, nv_pop = (np.empty(n_splits) for _ in range(3))
    for b in range(n_splits):
        perm = rng.permutation(n)
        cal, te = perm[:n_cal], perm[n_cal:]
        res = certify_threshold(scores[cal], fails[cal], alpha, delta, grid)
        cov[b], risk[b] = selective_stats(scores[te], fails[te], res.lam)
        cov_or[b] = oracle_coverage(fails[te].mean(), alpha)
        # Risk on the whole evaluation set, treated as the finite population the
        # calibration half was drawn from without replacement (binomial test is
        # conservative for that design); free of test-half sampling noise.
        pop_risk[b] = selective_stats(scores, fails, res.lam)[1]
        # Naive baseline: largest grid threshold whose empirical calibration
        # risk is <= alpha (no finite-sample correction).
        lam_nv = naive_threshold(scores[cal], fails[cal], alpha, grid)
        nv_cov[b], nv_risk[b] = selective_stats(scores[te], fails[te], lam_nv)
        nv_pop[b] = selective_stats(scores, fails, lam_nv)[1]
        eta[b] = cov[b] / cov_or[b] if cov_or[b] > 0 else np.nan
    return {
        "coverage": cov, "risk": risk, "eta": eta, "oracle_coverage": cov_or,
        "violation_rate": float((risk > alpha).mean()),
        "pop_risk": pop_risk,
        "pop_violation_rate": float((pop_risk > alpha).mean()),
        "naive_coverage": float(nv_cov.mean()),
        "naive_violation_rate": float((nv_risk > alpha).mean()),
        "naive_pop_violation_rate": float((nv_pop > alpha).mean()),
        "fail_rate": float(fails.mean()),
        "coverage_mean": float(cov.mean()),
        "eta_mean": float(np.nanmean(eta)) if np.isfinite(eta).any() else float("nan"),
    }
