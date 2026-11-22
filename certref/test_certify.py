"""Sanity checks on synthetic data: validity of the guarantee and the decomposition."""
import numpy as np

from certify import (aurc, certify_threshold, make_grid, oracle_coverage,
                     repeated_split_eval)


def synth(n, fail_rate, ranking_quality, rng):
    """Dice in [0,1]; score = noisy (1 - Dice). ranking_quality in [0,1]."""
    fails = rng.random(n) < fail_rate
    dice = np.where(fails, rng.uniform(0.0, 0.5, n), rng.uniform(0.5, 1.0, n))
    noise = rng.normal(0, 1, n)
    score = ranking_quality * (1 - dice) * 4 + (1 - ranking_quality) * noise
    return score, dice


def test_validity():
    rng = np.random.default_rng(1)
    for q in (0.2, 0.6, 1.0):
        s_ref, _ = synth(2000, 0.15, q, rng)          # stands in for train-set scores
        s, d = synth(2000, 0.15, q, rng)
        out = repeated_split_eval(s, d, tau=0.5, alpha=0.05, delta=0.1,
                                  grid=make_grid(s_ref), n_splits=300)
        # Test-set noise makes realized violations fluctuate; allow slack.
        assert out["violation_rate"] <= 0.15, (q, out["violation_rate"])
        print(f"q={q}: cov={out['coverage_mean']:.3f} eta={out['eta_mean']:.3f} "
              f"viol={out['violation_rate']:.3f}")


def test_perfect_ranking_near_oracle():
    rng = np.random.default_rng(2)
    s_ref, _ = synth(20000, 0.2, 1.0, rng)
    s, d = synth(20000, 0.2, 1.0, rng)
    out = repeated_split_eval(s, d, tau=0.5, alpha=0.05, delta=0.1,
                              grid=make_grid(s_ref, n_grid=200), n_splits=50)
    assert out["eta_mean"] > 0.9, out["eta_mean"]


def test_edge_cases():
    # First grid point accepts 100 cases: (0.95)^100 < 0.1, so it is testable.
    grid = make_grid(np.arange(1000.0))
    r = certify_threshold(np.arange(1000.0), np.ones(1000), 0.05, 0.1, grid)
    assert r.n_accepted == 0                       # everything fails
    r = certify_threshold(np.arange(1000.0), np.zeros(1000), 0.05, 0.1, grid)
    assert r.n_accepted == 1000                    # nothing fails
    # Too few accepted at the first grid point (20 < 45): no power, accept nothing.
    r = certify_threshold(np.arange(200.0), np.zeros(200), 0.05, 0.1,
                          make_grid(np.arange(200.0)))
    assert r.n_accepted == 0
    assert oracle_coverage(0.03, 0.05) == 1.0
    _, e = aurc(np.arange(10.0), np.r_[np.zeros(5), np.ones(5)])
    assert abs(e) < 1e-12


if __name__ == "__main__":
    test_edge_cases()
    test_perfect_ranking_near_oracle()
    test_validity()
    print("all tests passed")
