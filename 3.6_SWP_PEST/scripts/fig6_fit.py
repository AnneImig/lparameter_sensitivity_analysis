#!/usr/bin/env python3
"""Figure 6 of the manuscript (section 3.6): identifiability of l from the two 1,000,000-run sweeps.

(a) Ly1, (b) Ly2: best KGE of all usable runs in each n-l cell, with the concavity condition of
    Peters et al. (2011), on one colour scale shared by both maps.
(c) Share of each l class among the best 0.1 % of usable runs, divided by its share among all usable
    runs (1 = outflow carries no information on l; losses are removed from both).
"""
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import pandas as pd

from sweeps import RES, LY1, LY2


NE, LE = np.linspace(1, 4, 16), np.linspace(-2, 5, 15)
SPAN = 0.3                                   # colour range of figS_l_planes.py below the best run
SCALE = {'kge': (0.4, 0.84), 'r2': (0.3, 0.75)}   # shared colour range of the two maps per measure
LABEL = {'kge': 'best KGE in cell', 'r2': 'best $R^2$ in cell'}


def load(sweep_in, scores):
    """Parameter sets with KGE and R2 of usable runs (NaN for failed ones)."""
    p = pd.read_csv(sweep_in, index_col=0).reset_index(drop=True)
    s = pd.read_csv(scores)
    d = pd.concat([s[['kge', 'r2', 'failed_flag', 'n_failed_obs']].reset_index(drop=True), p], axis=1)
    d['ok'] = (d.failed_flag == 0) & (d.n_failed_obs == 0) & d.kge.notna()
    if 'nonphysical' in s:                   # infiltration > 1.01 x rain, from the top-2000 water-balance check
        d['ok'] &= ~s.nonphysical.to_numpy()
    d.loc[~d.ok, ['kge', 'r2']] = np.nan
    return d


def load_both():
    return (load(os.path.join(LY1, 'sweep_in.csv.gz'), os.path.join(LY1, 'sweep_scores_5pct.csv.gz')),
            load(os.path.join(LY2, 'sweep_in.csv.gz'), os.path.join(LY2, 'sweep_scores.csv.gz')))
N_SUB = 50                                   # used by figS_l_planes.py


def cell_best(d, col='kge'):
    """Best value of `col` (kge or r2) of all usable runs per n-l cell."""
    u = d[d.ok]
    i = np.clip(np.digitize(u.vgn1, NE) - 1, 0, len(NE) - 2)
    j = np.clip(np.digitize(u.l1, LE) - 1, 0, len(LE) - 2)
    best = np.full((len(NE) - 1, len(LE) - 1), np.nan)
    np.fmax.at(best, (i, j), u[col].to_numpy())
    return best


def fit_map(ax, d, title, col='kge'):
    best = cell_best(d, col)
    vmin, vmax = SCALE[col]
    im = ax.pcolormesh(NE, LE, best.T, cmap='viridis', vmin=vmin, vmax=vmax, shading='flat')
    n = np.linspace(1.001, 4, 400)
    ax.plot(n, 1 - 2 / (1 - 1 / n), color='#ff4fa0', lw=2)
    ax.text(3.92, -1.55, 'Peters et al. (2011)', ha='right', va='bottom', fontsize=8, color='#d6006f',
            bbox=dict(fc='white', ec='none', alpha=0.85, pad=1.5))
    ax.set(xlim=(1, 4), ylim=(-2, 5), xlabel='$n$ [-]', ylabel='$l$ [-]')
    ax.set_title(title, loc='left', fontsize=12)
    return im


def main():
    l1, l2 = load_both()
    fig = plt.figure(figsize=(15.5, 4.3), dpi=300)
    gs = fig.add_gridspec(1, 5, width_ratios=[1, 1, 0.05, 0.38, 1.05], wspace=0.22)
    ax1, ax2, cax, ax3 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1]), fig.add_subplot(gs[2]), fig.add_subplot(gs[4])
    fit_map(ax1, l1, '(a)')
    im = fit_map(ax2, l2, '(b)')
    ax2.set_ylabel('')
    cb = fig.colorbar(im, cax=cax, extend='both')
    cb.set_label(LABEL['kge'])

    bins = np.arange(-2, 5.01, 0.5)
    mid = 0.5 * (bins[:-1] + bins[1:])
    for d, ls, mk, mfc, name in ((l1, '-', 'o', 'black', 'Ly1'), (l2, '--', 'D', 'white', 'Ly2')):
        u = d[d.ok]
        top = u[u.kge >= u.kge.quantile(0.999)]
        hu, _ = np.histogram(u.l1, bins)
        ht, _ = np.histogram(top.l1, bins)
        ax3.plot(mid, (ht / len(top)) / (hu / len(u)), ls=ls, marker=mk, color='black', mfc=mfc, ms=5,
                 label=f'{name}: best 0.1 % (n = {len(top)}, KGE ≥ {top.kge.min():.2f})')
    ax3.axhline(1, color='0.4', lw=1)
    ax3.text(4.9, 1.06, 'no preference', ha='right', va='bottom', fontsize=8, color='0.4')
    ax3.axvline(0.5, color='red', ls='--', lw=1.2)
    ax3.text(0.6, 0.98, '$l$ = 0.5', color='red', fontsize=9, va='top', transform=ax3.get_xaxis_transform())
    ax3.set(xlim=(-2.2, 5), ylim=(0, None), xlabel='$l$ [-]',
            ylabel='relative frequency among best fits')
    ax3.set_title('(c)', loc='left', fontsize=12)
    ax3.legend(fontsize=7.5, loc='center right', framealpha=0.9)

    out = os.path.join(RES, 'fig6_fit.png')
    fig.savefig(out, bbox_inches='tight')
    print(f'wrote {out}')
    for name, d in (('Ly1', l1), ('Ly2', l2)):
        b = cell_best(d); print(name, 'empty cells', int(np.isnan(b).sum()), 'of', b.size)


if __name__ == '__main__':
    main()
