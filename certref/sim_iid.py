"""I.i.d. validity check: calibration sets drawn from a large synthetic population.

Unlike the repeated-halving protocol, the true selective risk of each certified
threshold is computed on a separate population sample of 200k cases, so the
violation rate estimates P(R(lambda_hat) > alpha) directly.

python certref/sim_iid.py
"""
import numpy as np

from certify import certify_threshold, make_grid, naive_threshold, selective_stats
from test_certify import synth


def main(reps=2000, seed=0):
    rng = np.random.default_rng(seed)
    alpha, delta = 0.05, 0.1
    print("p     q    n_cal  viol_LTT  viol_naive  cov_LTT  cov_naive")
    for p in (0.086, 0.15):
        for q in (0.2, 0.6, 1.0):
            pop_s, pop_d = synth(200_000, p, q, rng)
            pop_f = (pop_d < 0.5).astype(int)
            ref_s, _ = synth(2000, p, q, rng)              # stands in for validation scores
            grid = make_grid(ref_s, n_grid=50, min_coverage=0.2)
            for n_cal in (175, 430):
                v_l = v_n = c_l = c_n = 0.0
                for _ in range(reps):
                    s, d = synth(n_cal, p, q, rng)
                    f = (d < 0.5).astype(int)
                    lam = certify_threshold(s, f, alpha, delta, grid).lam
                    lam_nv = naive_threshold(s, f, alpha, grid)
                    cl, rl = selective_stats(pop_s, pop_f, lam)
                    cn, rn = selective_stats(pop_s, pop_f, lam_nv)
                    v_l += rl > alpha; v_n += rn > alpha; c_l += cl; c_n += cn
                print(f"{p:.3f} {q:.1f} {n_cal:5d}  {v_l/reps:8.3f}  {v_n/reps:10.3f}"
                      f"  {c_l/reps:7.3f}  {c_n/reps:9.3f}")


if __name__ == "__main__":
    main()
