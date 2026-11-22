#!/usr/bin/env python3
"""
Screens all 15 pairwise interactions among Model_Ly1's 6 van
Genuchten-Mualem parameters (theta_r, theta_s, alpha, n, Ks, l) using
REAL HYDRUS-1D runs -- same model/mechanics as linking_test_ly1.py (reused
directly: select_model_ly1(), write_params(), run_hydrus(),
cumulative_outflow()), just generalized from one fixed triple
(theta_s-n-Ks at a few Ks) to every parameter pair, each swept over its
own 5-level grid while the other 4 parameters sit at Model_Ly1's own
SELECTOR.IN defaults (BASELINE below).

For each pair (P1, P2): a 5x5 grid = 25 HYDRUS runs, holding the other 4
parameters fixed. 15 pairs x 25 = 375 runs total (~10-15 min at ~1.5-2s/
run). Output is cumulative bottom-node outflow Q (cm over 1216 days,
same quantity linking_test_ly1.py used for its DQ criterion) -- an
interaction between P1 and P2 shows up as Q(P1, P2) NOT being separable
into f(P1) + g(P2), i.e. the grid's contour lines bend instead of running
parallel to the axes.

Resumable: writes each (pair, v1, v2) result as soon as it's computed, and
skips combinations already present in the output CSV if re-run.

Run:
    cd pest_setup
    python interaction_test_ly1.py
"""

import os
import csv
import itertools
import numpy as np

import linking_test_ly1 as base

RESULTS_DIR = os.path.join(base.PEST_DIR, 'results', 'interaction_test_model_ly1')
os.makedirs(RESULTS_DIR, exist_ok=True)
OUT_CSV = os.path.join(RESULTS_DIR, 'pairwise_grid.csv')

# Model_Ly1's own SELECTOR.IN defaults (same reference point linking_test_ly1.py
# used) -- held fixed for whichever 4 parameters aren't the current pair.
BASELINE = {
    'thr1': 0.007, 'ths1': 0.275, 'alf1': 0.35,
    'vgn1': 1.41, 'ks1': 6040.20, 'l1': 0.5,
}

# 5-level grids per parameter, inset from LY1_BOUNDS (SWP_setup_parallel.py)
# to avoid the non-convergent edges probed by hand before committing to the
# full 375-run screen: l=-8 alone failed to converge (l=-4 was fine); n=3
# and n=4 (at baseline alpha/Ks) failed (n<=~2.0 was reliable). alf1 is
# log-spaced (it's one of PEST++'s LOG_PARAMS, sampled log-uniformly
# elsewhere in this repo) since it spans two decades.
GRIDS = {
    'thr1': np.array([0.005, 0.030, 0.055, 0.080, 0.099]),
    'ths1': np.array([0.15, 0.21, 0.27, 0.33, 0.38]),
    'alf1': np.array([0.006, 0.020, 0.060, 0.180, 0.450]),
    'vgn1': np.array([1.05, 1.30, 1.55, 1.80, 2.05]),
    'ks1':  np.array([4500.0, 20000.0, 40000.0, 60000.0, 78000.0]),
    'l1':   np.array([-2.0, -0.25, 1.5, 3.25, 5.0]),  # same l range as the Monte Carlo sweeps (Sect. 3.6)
}

PARAM_KWARG = {'thr1': 'theta_r', 'ths1': 'theta_s', 'alf1': 'alpha',
               'vgn1': 'n', 'ks1': 'ks', 'l1': 'l'}


def load_done():
    done = set()
    if os.path.exists(OUT_CSV):
        with open(OUT_CSV) as f:
            for row in csv.DictReader(f):
                done.add((row['p1'], row['p2'], row['v1'], row['v2']))
    return done


def main():
    base.select_model_ly1()
    done = load_done()
    write_header = not os.path.exists(OUT_CSV)

    pairs = list(itertools.combinations(GRIDS.keys(), 2))
    total = len(pairs) * 25
    i = 0
    with open(OUT_CSV, 'a', newline='') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(['p1', 'p2', 'v1', 'v2', 'q_sim', 't_end'])

        for p1, p2 in pairs:
            for v1 in GRIDS[p1]:
                for v2 in GRIDS[p2]:
                    i += 1
                    key = (p1, p2, f"{v1:.6g}", f"{v2:.6g}")
                    if key in done:
                        continue
                    kwargs = {PARAM_KWARG[p]: BASELINE[p] for p in GRIDS}
                    kwargs[PARAM_KWARG[p1]] = float(v1)
                    kwargs[PARAM_KWARG[p2]] = float(v2)
                    t_end, q = base.evaluate(kwargs.pop('theta_s'), kwargs.pop('n'),
                                              kwargs.pop('ks'), theta_r=kwargs['theta_r'],
                                              alpha=kwargs['alpha'], l=kwargs['l'])
                    status = 'FAIL' if q is None else f"Q={q:.3f}"
                    print(f"[{i}/{total}] {p1}={v1:.4g} {p2}={v2:.4g} -> {status}")
                    writer.writerow([p1, p2, f"{v1:.6g}", f"{v2:.6g}",
                                      '' if q is None else q, '' if t_end is None else t_end])
                    f.flush()

    print(f"Done -> {OUT_CSV}")


if __name__ == '__main__':
    main()
