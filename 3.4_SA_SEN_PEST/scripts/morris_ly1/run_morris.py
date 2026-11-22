"""Run the Ly1 Morris trajectories (design/morris_points.csv) through pestpp-swp,
keeping every run's daily outflow and water-balance totals.

    python run_morris.py [--dry-run] [--local] [--workers N] [--chunk N] [--first N --root DIR]
"""
import argparse
import json
import os

import pandas as pd

from hydrus_sa import condor_config, runner
from hydrus_sa.stages import morris, sweep

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'results', 'ly1_morris_p10')
POINTS = os.path.join(ROOT, 'design/morris_points.csv')
STAGE = 'morris_swp'

ap = argparse.ArgumentParser()
ap.add_argument('--dry-run', action='store_true')
ap.add_argument('--local', action='store_true')
ap.add_argument('--workers', type=int, default=70)
ap.add_argument('--chunk', type=int, default=2000)
ap.add_argument('--first', type=int, default=None,
                help='only the first N runs (a test; keep it a multiple of 7)')
ap.add_argument('--root', default=ROOT)
args = ap.parse_args()


def sampler(bounds, n_layers):
    df = pd.read_csv(POINTS, index_col='real_name')
    return df if args.first is None else df.iloc[:args.first]


DESIGN = os.path.join(ROOT, 'design/morris_design.json')
FULL = dict(zip(['thr1', 'ths1', 'alf1', 'vgn1', 'ks1', 'l1'],
                ['theta_r', 'theta_s', 'alpha', 'n', 'k_s', 'l']))


def prepare(built):
    # Put the design's bounds in the .pst and the manifest, so the record shows
    # what was sampled rather than the registry's ranges (Ly2 cluster 334's
    # manifest listed n [1, 4] for a design on [1.05, 2.3]).
    with open(DESIGN) as f:
        bounds = json.load(f)['bounds']
    morris.override_bounds(built, {FULL[k]: tuple(v) for k, v in bounds.items()})
    morris.fail_on_missing(built)
    morris.enable_extra_observations(built, water_balance=True)


mode = 'local' if args.local else 'condor'
if mode == 'condor':
    condor_config.configure(dry_run=args.dry_run)
try:
    sweep.run_matrix('model_ly1', sampler, stage=STAGE, workers=args.workers, mode=mode,
                     root=args.root, chunk=args.chunk, prepare=prepare,
                     sample_meta={'method': 'morris trajectories (SALib, r=2000, p=10) from '
                                            + POINTS},
                     abort_on_failures=False)
except runner.DryRunStop as e:
    print(e)
