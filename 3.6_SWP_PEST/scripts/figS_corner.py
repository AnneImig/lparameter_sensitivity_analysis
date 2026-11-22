#!/usr/bin/env python3
"""SI corner plots of the two 1M sweeps (Figs. S4, S5): where KGE conditions the parameters and their combinations.

Diagonal: density of each parameter among all sampled sets (prior, dashed), all usable runs (grey), the best 1 %
(blue) and the best 0.1 % (dark red) by KGE. Below the diagonal: all usable runs as a grey 2D histogram, the best
1 % and 0.1 % as points, and the Spearman correlation within the best 1 %. alpha and Ks on log10 axes, as sampled.
Usable as in Fig. 6 (fig6_fit.load_both): complete, not flagged by PEST++, not non-physical.
"""
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from scipy.stats import spearmanr

from sweeps import RES
from fig6_fit import load_both

PARAMS = ['thr1', 'ths1', 'alf1', 'vgn1', 'ks1', 'l1']
LOG = {'alf1', 'ks1'}
LAB = {'thr1': r'$\theta_r$ [–]', 'ths1': r'$\theta_s$ [–]', 'alf1': r'log$_{10}$ $\alpha$ [cm$^{-1}$]',
       'vgn1': '$n$ [–]', 'ks1': r'log$_{10}$ $K_s$ [cm d$^{-1}$]', 'l1': '$l$ [–]'}
BLUE, RED, GREY = '#1f5aa6', '#a3001b', '0.55'
NB = 40


def tr(d, p):
    return np.log10(d[p].to_numpy()) if p in LOG else d[p].to_numpy()


def corner(d, name, out):
    u = d[d.ok]
    t1 = u[u.kge >= u.kge.quantile(0.99)]
    t01 = u[u.kge >= u.kge.quantile(0.999)]
    k = len(PARAMS)
    fig, axes = plt.subplots(k, k, figsize=(17 / 2.54 * 1.25, 17 / 2.54 * 1.25), dpi=300)
    lim = {p: (np.nanmin(tr(d, p)), np.nanmax(tr(d, p))) for p in PARAMS}
    for i, py in enumerate(PARAMS):
        for j, px in enumerate(PARAMS):
            ax = axes[i, j]
            if j > i:
                ax.set_axis_off(); continue
            if i == j:
                bins = np.linspace(*lim[px], NB + 1)
                ax.hist(tr(d, px), bins, density=True, histtype='step', color='0.3', ls='--', lw=0.8)
                ax.hist(tr(u, px), bins, density=True, color=GREY, alpha=0.45)
                ax.hist(tr(t1, px), bins, density=True, histtype='step', color=BLUE, lw=1.2)
                ax.hist(tr(t01, px), bins, density=True, histtype='step', color=RED, lw=1.2)
                ax.set_yticks([])
            else:
                ax.hist2d(tr(u, px), tr(u, py), bins=NB, range=[lim[px], lim[py]], cmap='Greys',
                          norm=LogNorm(), alpha=0.5, rasterized=True)
                ax.scatter(tr(t1, px), tr(t1, py), s=0.4, color=BLUE, alpha=0.35, lw=0, rasterized=True)
                ax.scatter(tr(t01, px), tr(t01, py), s=0.8, color=RED, alpha=0.8, lw=0, rasterized=True)
                rho = spearmanr(tr(t1, px), tr(t1, py))[0]
                ax.text(0.04, 0.96, f'{rho:+.2f}', transform=ax.transAxes, ha='left', va='top', fontsize=6,
                        color='black' if abs(rho) >= 0.3 else '0.45', fontweight='bold' if abs(rho) >= 0.3 else 'normal',
                        bbox=dict(fc='white', ec='none', alpha=0.7, pad=0.6))
                ax.set_ylim(lim[py])
            ax.set_xlim(lim[px])
            ax.tick_params(labelsize=6, length=2)
            if i == k - 1:
                ax.set_xlabel(LAB[px], fontsize=7)
            else:
                ax.set_xticklabels([])
            if j == 0 and i > 0:
                ax.set_ylabel(LAB[py], fontsize=7)
            elif j > 0 or i == 0:
                ax.set_yticklabels([])
    h = [plt.Line2D([], [], color='0.3', ls='--', lw=0.8, label='prior (all sampled)'),
         plt.Rectangle((0, 0), 1, 1, color=GREY, alpha=0.45, label=f'usable ({len(u):,})'),
         plt.Line2D([], [], color=BLUE, lw=1.5, label=f'best 1 % (KGE ≥ {t1.kge.min():.3f})'),
         plt.Line2D([], [], color=RED, lw=1.5, label=f'best 0.1 % (KGE ≥ {t01.kge.min():.3f})')]
    fig.legend(handles=h, loc='upper right', bbox_to_anchor=(0.97, 0.95), fontsize=8, frameon=False,
               title=f'{name}: 1 000 000 realizations\nnumbers: Spearman ρ in the best 1 %', title_fontsize=8)
    fig.subplots_adjust(wspace=0.08, hspace=0.08)
    fig.savefig(out, bbox_inches='tight')
    print('wrote', out)
    c = t1[PARAMS].assign(alf1=np.log10(t1.alf1), ks1=np.log10(t1.ks1)).corr('spearman')
    pairs = sorted(((abs(c.loc[a, b]), a, b, c.loc[a, b]) for x, a in enumerate(PARAMS) for b in PARAMS[x + 1:]), reverse=True)
    print(f'  {name} strongest pairs in best 1 %: ' + ', '.join(f'{a}-{b} {r:+.2f}' for _, a, b, r in pairs[:5]))
    for p in PARAMS:
        q = lambda s: np.subtract(*np.percentile(tr(s, p), [95, 5]))
        print(f'  {name} {p}: 5-95 % width best 1 % / usable = {q(t1) / q(u):.2f}, best 0.1 % / usable = {q(t01) / q(u):.2f}')


def main():
    for (name, d), fig in zip(zip(('Ly1', 'Ly2'), load_both()), ('S4', 'S5')):
        corner(d, name, os.path.join(RES, f'fig{fig}_corner_{name.lower()}.png'))


if __name__ == '__main__':
    main()
