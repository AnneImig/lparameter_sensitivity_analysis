"""Morris trajectories for model_ly1, generated with SALib (no model runs).

    python design.py

r=2000 trajectories, p=10 levels, the step pestpp-sen also uses (delta =
p / (2 (p - 1)) = 5/9 of the range). alpha and Ks are stepped in log10, as in
the Ly1 run. Each trajectory is 7 rows: a grid start, then one parameter moved
per row. Writes design/morris_points.csv (real_name + parameters, in run order)
and design/morris_design.json (bounds and settings, read by analyze.py).
"""
import json
import os

import numpy as np
import pandas as pd
from SALib.sample import morris as morris_sample

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'results', 'ly1_morris_p10')
OUT = os.path.join(ROOT, 'design')
R, P, SEED = 2000, 10, 42

# models.PARAM_SHORT order. Rerun of the Ly1 Morris (cluster 333, ly1_morris_swp)
# on the same footing as the Ly2 rerun (ly2_morris_swp2): SALib trajectories, and
# theta_s from 0.15, not 0.1, so theta_s stays at least 0.05 above theta_r's upper
# bound (31% of cluster 333's runs at theta_s = 0.1 failed). Every other range is
# Ly1's own, as in cluster 333: alpha from 0.005, Ks 4,000-80,000, n capped at 2.3.
BOUNDS = {
    'thr1': [0.001, 0.1],
    'ths1': [0.15, 0.4],
    'alf1': [0.005, 0.5],
    'vgn1': [1.05, 2.3],
    'ks1': [4000.0, 80000.0],
    'l1': [-2.0, 5.0],
}
LOG = ['alf1', 'ks1']
names = list(BOUNDS)
problem = {'num_vars': len(names), 'names': names,
           'bounds': [list(np.log10(BOUNDS[n])) if n in LOG else BOUNDS[n] for n in names]}

X = morris_sample.sample(problem, R, num_levels=P, seed=SEED)
df = pd.DataFrame(X, columns=names)
for n in LOG:
    df[n] = 10 ** df[n]
df.index.name = 'real_name'

# Check the trajectory structure before anything is run on it.
block = len(names) + 1
assert len(df) == R * block
U = np.array([(X[:, j] - lo) / (hi - lo) for j, (lo, hi) in enumerate(problem['bounds'])]).T
for t in range(R):
    for s in range(1, block):
        moved = np.flatnonzero(np.abs(U[t * block + s] - U[t * block + s - 1]) > 1e-9)
        assert len(moved) == 1, (t, s, moved)
        assert abs(abs(U[t * block + s, moved[0]] - U[t * block + s - 1, moved[0]]) - P / (2 * (P - 1))) < 1e-9
    assert sorted(np.flatnonzero(np.abs(np.diff(U[t * block:(t + 1) * block], axis=0)) > 1e-9) % len(names)) \
        == list(range(len(names)))

df.to_csv(os.path.join(OUT, 'morris_points.csv'))
with open(os.path.join(OUT, 'morris_design.json'), 'w') as f:
    json.dump({'model': 'model_ly1', 'r': R, 'p': P, 'seed': SEED, 'delta': P / (2 * (P - 1)),
               'bounds': BOUNDS, 'log_params': LOG, 'n_runs': len(df), 'block': block,
               'sampler': 'SALib.sample.morris.sample (no trajectory optimisation)'}, f, indent=1)
print(f"{len(df)} runs = {R} trajectories x {block}; each step moves one parameter by delta=5/9")
print(df.describe().loc[['min', 'max']].to_string())
for n in names:
    print(n, 'levels:', np.unique(df[n].round(8)).size)
