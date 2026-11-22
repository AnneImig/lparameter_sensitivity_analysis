"""Pair each l = 0.5 run with its l-free twin and compare the fit.

    python compare_l.py model_ly1|model_ly2

The fixed-l sweep was drawn with the same seed and size as the l-free 1M sweep,
so row i of each has the same theta_r, theta_s, alpha, n and Ks; only l differs.
Writes, into the fixed-l stage folder:
  paired.csv               one row per run: KGE both ways, l in the free run, flags
  compare_summary.json     outcomes, best fits, paired KGE change
  figures/delta_kge_by_l.png  KGE change from fixing l, by the l the free run had
Runs that take in more water than rained (infiltration > 1.01 x rain, as in the
Morris analysis) are flagged from the water-balance totals; the l-free sweep has
no totals, so its flag comes from results/l_check for its best 2,000 runs only.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, '/Users/scratch/portega/SA_6VGN_parameters')
from hydrus_sa import observations, plots  # noqa: E402

MODEL = sys.argv[1]
LY = MODEL.split('_')[1]
REPO = '/Users/scratch/portega/SA_6VGN_parameters'
FIXED = os.path.join(REPO, f'results/l05_1M/results/{MODEL}/sweep')
FREE = os.path.join(REPO, f'results/{LY}_1M/results/{MODEL}/sweep')
TOPCHECK = os.path.join(REPO, f'results/l_check/results/{MODEL}/topcheck/top_runs_checked.csv')
TOL = 1.01


def rain_litres(model_dir):
    lines = open(os.path.join(model_dir, 'ATMOSPH.IN')).read().splitlines()
    start = next(i for i, l in enumerate(lines) if l.split()[:1] == ['tAtm']) + 1
    total = 0.0
    for l in lines[start:]:
        f = l.split()
        if not f or not f[0][0].isdigit():
            break
        total += float(f[1])
    return total * observations.SWI_LYSIMETER_AREA_CM2 / 1000.0


cols = ['input_run_id', 'failed', 'sentinel', 'kge_q']
fx = pd.read_csv(os.path.join(FIXED, 'metrics_partial.csv'), usecols=cols).set_index('input_run_id')
fr = pd.read_csv(os.path.join(FREE, 'metrics_partial.csv'), usecols=cols).set_index('input_run_id')
l_free = pd.read_csv(os.path.join(FREE, 'sweep_in.csv'), index_col='real_name', usecols=['real_name', 'l1'])['l1']
wb = pd.read_csv(os.path.join(FIXED, 'sweep_out.csv'), usecols=['input_run_id', 'wb_infil']).set_index('input_run_id')
rain = rain_litres(os.path.join(FIXED, 'raw/template/tmp_copy'))

d = pd.DataFrame({
    'l_free': l_free,
    'ok_free': (fr.failed == 0) & (fr.sentinel == 0),
    'kge_free': fr.kge_q,
    'ok_fixed': (fx.failed == 0) & (fx.sentinel == 0),
    'kge_fixed': fx.kge_q,
    'infil_ratio_fixed': wb.wb_infil / rain,
})
d['nonphysical_fixed'] = d.ok_fixed & (d.infil_ratio_fixed > TOL)
d.loc[~d.ok_free, 'kge_free'] = np.nan
d.loc[~d.ok_fixed, 'kge_fixed'] = np.nan
d['delta_kge'] = d.kge_fixed - d.kge_free
d.index.name = 'input_run_id'
d.to_csv(os.path.join(FIXED, 'paired.csv'))

phys_fixed = d[d.ok_fixed & ~d.nonphysical_fixed]
both = d[d.ok_free & d.ok_fixed]
summary = {
    'model': MODEL, 'rain_L': rain, 'runs': len(d),
    'lost_free': float(1 - d.ok_free.mean()), 'lost_fixed': float(1 - d.ok_fixed.mean()),
    'nonphysical_fixed': int(d.nonphysical_fixed.sum()),
    'best_kge_free_all': float(d.kge_free.max()),
    'best_kge_fixed_all': float(d.kge_fixed.max()),
    'best_kge_fixed_physical': float(phys_fixed.kge_fixed.max()),
    'best_fixed_run': int(phys_fixed.kge_fixed.idxmax()),
    'kge_above_0_5': {'free': int((d.kge_free > 0.5).sum()), 'fixed_physical': int((phys_fixed.kge_fixed > 0.5).sum())},
    'paired_complete_both': len(both),
    'paired_median_delta_kge': float(both.delta_kge.median()),
    'paired_share_better_fixed': float((both.delta_kge > 0).mean()),
}
if os.path.exists(TOPCHECK):
    tc = pd.read_csv(TOPCHECK)
    ok = tc[(tc.failed_flag == 0) & ~tc.nonphysical]
    summary['best_kge_free_physical_top2000'] = float(ok.kge_q.max())
    summary['top2000_nonphysical_free'] = int(tc.nonphysical.sum())

edges = np.linspace(-2, 5, 15)
both = both.assign(lbin=pd.cut(both.l_free, edges, include_lowest=True))
by = both.groupby('lbin', observed=True).delta_kge
stats = pd.DataFrame({'centre': [iv.mid for iv in by.median().index], 'median': by.median().to_numpy(),
                      'q25': by.quantile(0.25).to_numpy(), 'q75': by.quantile(0.75).to_numpy()})
best = d.assign(lbin=pd.cut(d.l_free, edges, include_lowest=True)).groupby('lbin', observed=True)
stats['best_free'] = best.kge_free.max().to_numpy()
stats['best_fixed_twin'] = best.kge_fixed.max().to_numpy()
summary['delta_kge_by_l_free'] = stats.round(4).to_dict(orient='records')
with open(os.path.join(FIXED, 'compare_summary.json'), 'w') as f:
    json.dump(summary, f, indent=1)
print(json.dumps({k: v for k, v in summary.items() if k != 'delta_kge_by_l_free'}, indent=1))
print(stats.round(3).to_string(index=False))

plt = plots._style()
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
a1.fill_between(stats.centre, stats.q25, stats.q75, color='#2a78d6', alpha=0.25, linewidth=0)
a1.plot(stats.centre, stats['median'], color='#2a78d6', linewidth=2)
a1.axhline(0, color=plots.INK_SECONDARY, linewidth=1)
a1.axvline(0.5, color='#eb6834', linewidth=1.4, linestyle=(0, (5, 3)))
a1.set_xlabel('l in the free run'); a1.set_ylabel('KGE(l = 0.5) − KGE(l free)')
a1.set_title('Paired change in KGE, median and interquartile band', fontsize=11)
a2.plot(stats.centre, stats.best_free, color='#2a78d6', linewidth=2, label='best, l free')
a2.plot(stats.centre, stats.best_fixed_twin, color='#eb6834', linewidth=2, label='best twin, l = 0.5')
a2.set_xlabel('l in the free run'); a2.set_ylabel('best KGE in bin')
a2.set_title('Best fit by l, and the same parameter sets at l = 0.5', fontsize=11)
a2.legend(frameon=False, fontsize=9)
for ax in (a1, a2):
    plots._tidy(ax)
fig.suptitle(f'{LY.upper()}: what fixing l = 0.5 costs, run for run', color=plots.INK, fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.93])
os.makedirs(os.path.join(FIXED, 'figures'), exist_ok=True)
print('wrote', plots._save(fig, os.path.join(FIXED, 'figures', 'delta_kge_by_l.png')))
