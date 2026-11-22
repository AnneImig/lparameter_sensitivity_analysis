"""Figure 5: pairwise interactions (difference of differences) for Ly1 and Ly2.

    python fig5_ddq.py

Reads ../results/interaction_test_model_ly{1,2}/pairwise_grid.csv, written by
interaction_test_ly1.py and interaction_test_ly2.py (5 x 5 grid per pair, the
other four parameters at each model's SELECTOR.IN default; q_sim = cumulative outflow
in cm over 1,216 days). Corner metric:

    ddQ = [Q(A_hi,B_hi) - Q(A_lo,B_hi)] - [Q(A_hi,B_lo) - Q(A_lo,B_lo)]

Differences from that script, to match the manuscript caption:
  - runs that did not finish OR whose cumulative outflow exceeds precipitation are dropped;
  - ddQ is given relative to the mean cumulative outflow of its four corner runs, in %;
  - colour = |ddQ| %, shared log scale; numbers = signed %; lower triangle only;
  - l grid -2 to 5; when a corner run is missing, the widest complete rectangle is used
    (shrinking inward one grid step at a time) and the cell is marked with *.
Writes fig5_ddq.png, ddq_table.csv and excluded_runs.csv to ../results.
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm

PART = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # 3.5_Interaction/
REPO = os.path.dirname(PART)                                        # repository root
OUT = os.path.join(PART, 'results')
PARAMS = ['thr1', 'ths1', 'alf1', 'vgn1', 'ks1', 'l1']
LABELS = {'thr1': r'$\theta_r$', 'ths1': r'$\theta_s$', 'alf1': r'$\alpha$',
          'vgn1': r'$n$', 'ks1': r'$K_s$', 'l1': r'$l$'}


def rain_cm(model_folder):
    lines = open(os.path.join(REPO, 'HYDRUS_models', model_folder, 'ATMOSPH.IN')).read().splitlines()
    start = next(i for i, l in enumerate(lines) if l.split()[:1] == ['tAtm']) + 1
    total = 0.0
    for l in lines[start:]:
        f = l.split()
        if not f or not f[0][0].isdigit():
            break
        total += float(f[1])
    return total


def load(ly, folder):
    g = pd.read_csv(os.path.join(OUT, f'interaction_test_model_{ly}', 'pairwise_grid.csv'))
    rain = rain_cm(folder)
    failed = g.q_sim.isna()
    over = ~failed & (g.q_sim > rain)
    bad = g[failed | over].assign(reason=np.where(failed[failed | over], 'did not finish', 'outflow > precipitation'))
    g.loc[failed | over, 'q_sim'] = np.nan
    return g, bad, rain


def corners(g, p1, p2):
    """Widest rectangle whose four corners all have a valid q_sim; flag if shrunk."""
    s = g[(g.p1 == p1) & (g.p2 == p2)]
    q = s.set_index(['v1', 'v2'])['q_sim']
    v1s, v2s = sorted(s.v1.unique()), sorted(s.v2.unique())
    n1, n2 = len(v1s), len(v2s)
    options = []
    for a in range(n1):
        for b in range(a + 1, n1):
            for c in range(n2):
                for d in range(c + 1, n2):
                    options.append(((b - a) + (d - c), a, b, c, d))
    for _, a, b, c, d in sorted(options, reverse=True):
        vals = [q.get((v1s[i], v2s[j])) for i, j in ((a, c), (b, c), (a, d), (b, d))]
        if all(v is not None and np.isfinite(v) for v in vals):
            shrunk = not (a == 0 and b == n1 - 1 and c == 0 and d == n2 - 1)
            return (v1s[a], v1s[b], v2s[c], v2s[d], *vals, shrunk)
    return None


def table(g):
    rows = []
    for i, p1 in enumerate(PARAMS):
        for p2 in PARAMS[i + 1:]:
            r = corners(g, p1, p2)
            if r is None:
                rows.append({'p1': p1, 'p2': p2})
                continue
            lo1, hi1, lo2, hi2, q_ll, q_hl, q_lh, q_hh, shrunk = r
            ddq = (q_hh - q_lh) - (q_hl - q_ll)
            mean = np.mean([q_ll, q_hl, q_lh, q_hh])
            rows.append({'p1': p1, 'p2': p2, 'ddQ_cm': ddq, 'mean_corner_Q_cm': mean,
                         'ddQ_pct': 100 * ddq / mean, 'reduced_range': shrunk,
                         f'p1_lo': lo1, 'p1_hi': hi1, 'p2_lo': lo2, 'p2_hi': hi2})
    return pd.DataFrame(rows)


tabs, excluded = {}, []
for ly, folder in (('ly1', 'Model_Ly1'), ('ly2', 'Model_Ly2')):
    g, bad, rain = load(ly, folder)
    tabs[ly] = table(g)
    excluded.append(bad.assign(lysimeter=ly))
    print(f'{ly}: {len(g)} runs, rain {rain:.1f} cm; excluded {len(bad)} '
          f'({(bad.reason == "did not finish").sum()} did not finish, '
          f'{(bad.reason == "outflow > precipitation").sum()} outflow > precipitation)')
pd.concat([t.assign(lysimeter=ly) for ly, t in tabs.items()]).to_csv(os.path.join(OUT, 'ddq_table.csv'), index=False)
pd.concat(excluded).to_csv(os.path.join(OUT, 'excluded_runs.csv'), index=False)

absvals = np.concatenate([t.ddQ_pct.abs().dropna().to_numpy() for t in tabs.values()])
norm = LogNorm(vmin=max(absvals[absvals > 0].min(), 1e-3), vmax=absvals.max())
k = len(PARAMS)
fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.2))
for ax, (ly, t), tag in zip(axes, tabs.items(), ('(a) Ly1', '(b) Ly2')):
    mat = np.full((k, k), np.nan)
    txt = {}
    for _, r in t.iterrows():
        i, j = PARAMS.index(r.p2), PARAMS.index(r.p1)   # lower triangle: row > column
        if pd.notna(r.get('ddQ_pct')):
            mat[i, j] = abs(r.ddQ_pct)
            txt[(i, j)] = f"{r.ddQ_pct:+.1f}{'*' if r.reduced_range else ''}"
        else:
            txt[(i, j)] = 'n/a'
    im = ax.imshow(np.ma.masked_invalid(mat), cmap='inferno', norm=norm)
    for (i, j), s in txt.items():
        v = mat[i, j]
        light = np.isfinite(v) and norm(v) > 0.6
        ax.text(j, i, s, ha='center', va='center', fontsize=8.5, color='black' if light else 'white')
    ax.set_xticks(range(k), [LABELS[p] for p in PARAMS])
    ax.set_yticks(range(k), [LABELS[p] for p in PARAMS])
    ax.set_xlim(-0.5, k - 1.5)
    ax.set_ylim(k - 0.5, 0.5)
    ax.set_title(tag, loc='left')
    for s in ax.spines.values():
        s.set_visible(False)
cb = fig.colorbar(im, ax=axes, fraction=0.035, pad=0.03,
             label=r'$|\Delta\Delta Q|$ relative to mean corner outflow (%, log scale)')
cb.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:g}'))
cb.ax.yaxis.set_minor_formatter(plt.NullFormatter())
fig.savefig(os.path.join(OUT, 'fig5_ddq.png'), dpi=200, bbox_inches='tight')
print('wrote', os.path.join(OUT, 'fig5_ddq.png'))
for ly, t in tabs.items():
    print(ly); print(t[['p1', 'p2', 'ddQ_pct', 'reduced_range']].round(2).to_string(index=False))
