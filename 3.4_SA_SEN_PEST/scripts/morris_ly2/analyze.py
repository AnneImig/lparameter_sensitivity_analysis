"""Morris on model_ly2 from SALib trajectories run through pestpp-swp,
with and without the runs that take in more water than rained.

    python analyze.py <stage_dir> [R]

The 14,000 points are design/morris_points.csv (SALib, r=2000, p=10; ranges in
design/morris_design.json): blocks of 7 runs, each after the first moving one parameter. Elementary
effects are recomputed here from every run's daily outflow, so any run can be
left out. A run is non-physical when its total infiltration (sum(Infil) in
T_LEVEL.OUT) exceeds the record's rain by more than 1%: the surface, held at
hCritA, fed the profile water that never fell. Only the elementary effects that
touch such a run are dropped, never the whole trajectory.

Objectives, each a sum of squared errors against measured outflow unless noted:
  q     daily Q (L/d)               peaks and timing
  lq    daily log10 Q               recessions and baseflow
  mvol  monthly volume (L)          water balance, not timing
  cum   cumulative volume (L)       long-term balance
  kge   KGE on daily Q              fit, higher is better

EE = (y_after - y_before) / delta, delta the step in [0, 1] units of the
transformed (log10 for alpha and Ks) parameter range.

Writes <stage_dir>/analysis/ (CSV) and analysis/figures/.
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# needs the hydrus_sa package of the cluster repository (not included here) and sweep_out.csv
# (95 MB, not included); the elementary effects it writes are in ../../results/.
REPO = '/Users/scratch/portega/SA_6VGN_parameters'
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, 'legacy'))
from hydrus_sa import metrics, observations, plots  # noqa: E402

stage = sys.argv[1]
R = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
OUT = os.path.join(stage, 'analysis')
FIG = os.path.join(OUT, 'figures')
os.makedirs(FIG, exist_ok=True)

DESIGN = os.path.join(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'results', 'ly2_morris_p10'), 'design', 'morris_design.json')
MODEL = 'model_ly2'
PARAMS = ['thr1', 'ths1', 'alf1', 'vgn1', 'ks1', 'l1']        # models.PARAM_SHORT order
LOG = {'alf1', 'ks1'}
COLOUR = dict(zip(PARAMS, ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300']))
OBJ = {'q': 'daily Q', 'lq': 'daily log10 Q', 'mvol': 'monthly volume',
       'cum': 'cumulative volume', 'kge': 'KGE (daily Q)'}
TOL = 1.01              # infiltration may exceed rain by 1% before a run is flagged
BLOCK = len(PARAMS) + 1

# ── inputs ──────────────────────────────────────────────────────────────────
model_dir = os.path.join(stage, 'raw/template/tmp_copy')
q_obs = [o for o in observations.build_observations(model_dir) if o['group'] == 'q']
q_names = [o['name'] for o in q_obs]
q_meas = np.array([o['value'] for o in q_obs])
dates = observations._read_compare_dates(observations._find_compare_file(model_dir))
months = pd.Series([d.strftime('%Y_%m') for d in dates]).to_numpy()
month_keys = list(dict.fromkeys(months))
M = np.array([[m == k for m in months] for k in month_keys], dtype=float)  # months x days

rain_l = 0.0
with open(os.path.join(model_dir, 'ATMOSPH.IN')) as f:
    lines = f.read().splitlines()
start = next(i for i, l in enumerate(lines) if l.split()[:1] == ['tAtm']) + 1
for l in lines[start:]:
    f_ = l.split()
    if not f_ or not f_[0][0].isdigit():
        break
    rain_l += float(f_[1])
rain_l *= observations.SWI_LYSIMETER_AREA_CM2 / 1000.0

import json  # noqa: E402
design = json.load(open(DESIGN))
bounds = pd.DataFrame(design['bounds'], index=['parlbnd', 'parubnd']).T
par = pd.read_csv(os.path.join(stage, 'sweep_in.csv'), index_col=0)
par.index = par.index.astype(int)
wb_cols = [n for n, _, _ in observations.WB_TOTALS]
out = pd.read_csv(os.path.join(stage, 'sweep_out.csv'),
                  usecols=['input_run_id', 'failed_flag'] + q_names + wb_cols)
out = out.set_index('input_run_id').sort_index()
assert list(out.index) == list(range(len(out))), "run ids are not 0..N-1"
par = par.loc[out.index, PARAMS]

failed = out['failed_flag'].to_numpy() != 0
Q = out[q_names].to_numpy(float)
Q[failed] = np.nan
infil = out['wb_infil'].to_numpy(float)
nonphys = ~failed & (infil > TOL * rain_l)
overdrain_old = ~failed & (out['wb_drain'].to_numpy(float) > 3750.0)

# ── one value per run and objective ─────────────────────────────────────────
log_floor = observations.LOG_Q_FLOOR
y = {
    'q': np.sum((Q - q_meas) ** 2, axis=1),
    'lq': np.sum((np.log10(np.maximum(Q, log_floor)) - np.log10(np.maximum(q_meas, log_floor))) ** 2, axis=1),
    'mvol': np.sum((Q @ M.T - M @ q_meas) ** 2, axis=1),
    'cum': np.sum((np.cumsum(Q @ M.T, axis=1) - np.cumsum(M @ q_meas)) ** 2, axis=1),
}
kge = np.full(len(Q), np.nan)
ms = pd.Series(q_meas)
for i in np.flatnonzero(~failed):
    kge[i] = metrics.kge(ms, pd.Series(Q[i]))
y['kge'] = kge
for k in y:
    y[k] = np.where(failed, np.nan, y[k])

runs = pd.DataFrame({'failed': failed.astype(int), 'nonphysical': nonphys.astype(int),
                     'rain_L': rain_l, **{c: out[c].to_numpy() for c in wb_cols},
                     **{f'y_{k}': v for k, v in y.items()}}, index=out.index)
runs.index.name = 'run_id'
runs.to_csv(os.path.join(OUT, 'runs.csv'))

# ── elementary effects ──────────────────────────────────────────────────────
X = par.to_numpy(float).copy()
lo = bounds.loc[PARAMS, 'parlbnd'].to_numpy(float).copy()
hi = bounds.loc[PARAMS, 'parubnd'].to_numpy(float).copy()
for j, p in enumerate(PARAMS):
    if p in LOG:
        X[:, j], lo[j], hi[j] = np.log10(X[:, j]), np.log10(lo[j]), np.log10(hi[j])
U = (X - lo) / (hi - lo)

n_traj = len(U) // BLOCK
rows = []
for t in range(n_traj):
    for s in range(1, BLOCK):
        a, b = t * BLOCK + s - 1, t * BLOCK + s
        d = U[b] - U[a]
        moved = np.flatnonzero(np.abs(d) > 1e-9)
        if len(moved) != 1:
            raise RuntimeError(f"runs {a}->{b} move {len(moved)} parameters")
        j = moved[0]
        row = {'traj': t, 'run_a': a, 'run_b': b, 'parameter': PARAMS[j], 'delta': d[j],
               'failed': bool(failed[a] or failed[b]),
               'nonphysical': bool(nonphys[a] or nonphys[b])}
        for k, v in y.items():
            row[k] = (v[b] - v[a]) / d[j]
        rows.append(row)
ee = pd.DataFrame(rows)
ee.to_csv(os.path.join(OUT, 'elementary_effects.csv'), index=False)

rng = np.random.default_rng(7)


def stats(sub, obj):
    x = sub[obj].dropna().to_numpy(float)
    boot = np.abs(x)[rng.integers(0, len(x), (2000, len(x)))].mean(1)
    return {'mu': x.mean(), 'mu_star': np.abs(x).mean(), 'sigma': x.std(ddof=1),
            'n_ee': len(x), 'mu_star_lo': np.percentile(boot, 2.5),
            'mu_star_hi': np.percentile(boot, 97.5)}


VARIANTS = {'all': ee[~ee['failed']], 'physical': ee[~ee['failed'] & ~ee['nonphysical']]}
rows = []
for var, sub in VARIANTS.items():
    for obj in OBJ:
        for p in PARAMS:
            rows.append({'variant': var, 'objective': obj, 'parameter': p,
                         **stats(sub[sub['parameter'] == p], obj)})
t = pd.DataFrame(rows)
mx = t.groupby(['variant', 'objective'])['mu_star'].transform('max')
for c in ('mu_star', 'mu_star_lo', 'mu_star_hi'):
    t[c + '_rel'] = t[c] / mx
t['ratio'] = t['sigma'] / t['mu_star']
t['rank'] = t.groupby(['variant', 'objective'])['mu_star'].rank(ascending=False).astype(int)
t.to_csv(os.path.join(OUT, 'morris_by_objective.csv'), index=False)

# ── report ──────────────────────────────────────────────────────────────────
print(f"rain {rain_l:,.0f} L; {len(out)} runs, {failed.sum()} failed, "
      f"{nonphys.sum()} non-physical (infiltration > {TOL:.2f} x rain); "
      f"{overdrain_old.sum()} by the old outflow > 3,750 L rule")
print(f"elementary effects: {len(ee)}; touching a failed run {ee['failed'].sum()}, "
      f"a non-physical run {(ee['nonphysical'] & ~ee['failed']).sum()} more")
print("EEs kept per parameter (physical):",
      VARIANTS['physical'].groupby('parameter').size().loc[PARAMS].to_dict())

for var in VARIANTS:
    ranks = t[t['variant'] == var].pivot(index='parameter', columns='objective',
                                         values='rank').loc[PARAMS, list(OBJ)]
    print(f"\nrank per objective ({var}):")
    print(ranks.to_string())

rho = {}
for var in VARIANTS:
    mu = t[t['variant'] == var].pivot(index='parameter', columns='objective',
                                      values='mu_star').loc[PARAMS, list(OBJ)]
    rho[var] = pd.DataFrame([[spearmanr(mu[a], mu[b])[0] for b in OBJ] for a in OBJ],
                            index=list(OBJ), columns=list(OBJ))
    rho[var].to_csv(os.path.join(OUT, f'objective_rank_correlation_{var}.csv'))
print("\nSpearman between objectives (physical):")
print(rho['physical'].round(2).to_string())

# ── figures ─────────────────────────────────────────────────────────────────
plt = plots._style()
phys = t[t['variant'] == 'physical']

# 1. mu*-sigma, the stock Morris figure, once per objective.
for obj, label in OBJ.items():
    csv = os.path.join(OUT, f'morris_{obj}.csv')
    s = phys[phys['objective'] == obj][['parameter', 'mu', 'mu_star', 'sigma', 'n_ee']]
    s.rename(columns={'parameter': 'parameter_name'}).to_csv(csv, index=False)
    plots.morris(csv, os.path.join(FIG, f'morris_{obj}.png'),
                 title=f'Morris screening, {label} - {MODEL}, r={R}, non-physical runs out')

# 2. mu* per objective, same layout as the objectives run.
fig, axes = plt.subplots(1, len(OBJ), figsize=(16, 3.6), sharey=True)
yy = np.arange(len(PARAMS))[::-1]
for ax, (obj, label) in zip(axes, OBJ.items()):
    s = phys[phys['objective'] == obj].set_index('parameter').loc[PARAMS]
    ax.barh(yy, s['mu_star_rel'], color=[COLOUR[p] for p in PARAMS], height=0.62)
    ax.errorbar(s['mu_star_rel'], yy,
                xerr=[s['mu_star_rel'] - s['mu_star_lo_rel'], s['mu_star_hi_rel'] - s['mu_star_rel']],
                fmt='none', ecolor=plots.INK_MUTED, elinewidth=1, capsize=2)
    for y_, (p, r) in zip(yy, s['rank'].items()):
        ax.text(min(s.loc[p, 'mu_star_hi_rel'], 1.25) + 0.03, y_, f'#{r}', va='center',
                fontsize=8, color=plots.INK_SECONDARY)
    ax.set_xlim(0, 1.45)
    ax.set_title(label)
    ax.set_xlabel('mu* / largest')
    ax.grid(axis='y', visible=False)
    plots._tidy(ax)
axes[0].set_yticks(yy)
axes[0].set_yticklabels(PARAMS)
fig.suptitle(f'Morris mu* under five objectives - {MODEL}, r={R}, non-physical runs out; '
             'bootstrap 95% intervals, rank', color=plots.INK, fontsize=11, x=0.01, ha='left')
fig.tight_layout(rect=[0, 0, 1, 0.92])
plots._save(fig, os.path.join(FIG, 'objectives_mu_star.png'))

# 3. New: mu* with every run against mu* with the non-physical runs out.
fig, axes = plt.subplots(1, len(OBJ), figsize=(16, 3.8), sharey=True)
h = 0.36
for ax, (obj, label) in zip(axes, OBJ.items()):
    for k, (var, alpha, name) in enumerate((('all', 0.35, 'all runs'),
                                            ('physical', 1.0, 'non-physical out'))):
        s = t[(t['variant'] == var) & (t['objective'] == obj)].set_index('parameter').loc[PARAMS]
        ax.barh(yy + (h / 2 if k == 0 else -h / 2), s['mu_star_rel'], height=h,
                color=[COLOUR[p] for p in PARAMS], alpha=alpha, label=name)
        for y_, (p, r) in zip(yy, s['rank'].items()):
            ax.text(min(s.loc[p, 'mu_star_rel'], 1.2) + 0.03, y_ + (h / 2 if k == 0 else -h / 2),
                    f'#{r}', va='center', fontsize=7, color=plots.INK_SECONDARY)
    ax.set_xlim(0, 1.35)
    ax.set_title(label)
    ax.set_xlabel('mu* / largest (each variant)')
    ax.grid(axis='y', visible=False)
    plots._tidy(ax)
axes[0].set_yticks(yy)
axes[0].set_yticklabels(PARAMS)
handles, labels = axes[-1].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper right', ncol=2, fontsize=8, frameon=False)
fig.suptitle(f'Morris mu*: all runs (pale, upper bar) vs non-physical runs out (solid, lower bar) - '
             f'{MODEL}, r={R}', color=plots.INK, fontsize=11, x=0.01, ha='left')
fig.tight_layout(rect=[0, 0, 1, 0.92])
plots._save(fig, os.path.join(FIG, 'before_after_filter.png'))

# 4. r-convergence figures: see ../fig4_morris_convergence.py
print(f"\nfigures in {FIG}")
