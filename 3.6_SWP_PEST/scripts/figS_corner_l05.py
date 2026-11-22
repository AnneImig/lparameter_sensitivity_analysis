#!/usr/bin/env python3
"""SI corner plots comparing the l-free and the l = 0.5 sweeps of each lysimeter (same 1,000,000 parameter sets).

Five shared parameters (theta_r, theta_s, alpha, n, Ks). Grey: usable runs of the l-free sweep; blue: best 1 % by
KGE with l free; orange: best 1 % by KGE with l = 0.5 (each sweep ranked on its own). Where orange departs from blue,
the other parameters shift to compensate for the fixed l. Numbers: Spearman rho within the best 1 % (blue / orange).
Usable as in Fig. 6 and compare_l05.py: complete, not flagged, not non-physical.
"""
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm
from scipy.stats import spearmanr

from compare_l05 import load as load_paired
from sweeps import RES

PARAMS = ['thr1', 'ths1', 'alf1', 'vgn1', 'ks1']
LOG = {'alf1', 'ks1'}
LAB = {'thr1': r'$\theta_r$ [–]', 'ths1': r'$\theta_s$ [–]', 'alf1': r'log$_{10}$ $\alpha$ [cm$^{-1}$]',
       'vgn1': '$n$ [–]', 'ks1': r'log$_{10}$ $K_s$ [cm d$^{-1}$]'}
FREE, FIXED, GREY = '#1f5aa6', '#e07b00', '0.55'
SWEEP_IN = {'Ly1': 'ly1_1M', 'Ly2': 'ly2_1M'}
NB = 40


def tr(d, p):
    return np.log10(d[p].to_numpy()) if p in LOG else d[p].to_numpy()


def data(ly):
    par = pd.read_csv(os.path.join(RES, SWEEP_IN[ly], 'sweep_in.csv.gz'), index_col=0).reset_index(drop=True)
    pr = load_paired(ly).reset_index(drop=True)
    return pd.concat([par[PARAMS], pr[['ok_free', 'kge_free', 'ok_fixed', 'kge_fixed']]], axis=1)


def corner(ly, out):
    d = data(ly)
    uf, ux = d[d.ok_free], d[d.ok_fixed]
    bf = uf[uf.kge_free >= uf.kge_free.quantile(0.99)]
    bx = ux[ux.kge_fixed >= ux.kge_fixed.quantile(0.99)]
    k = len(PARAMS)
    fig, axes = plt.subplots(k, k, figsize=(8.4, 8.4), dpi=300)
    lim = {p: (np.nanmin(tr(d, p)), np.nanmax(tr(d, p))) for p in PARAMS}
    for i, py in enumerate(PARAMS):
        for j, px in enumerate(PARAMS):
            ax = axes[i, j]
            if j > i:
                ax.set_axis_off(); continue
            if i == j:
                bins = np.linspace(*lim[px], NB + 1)
                ax.hist(tr(uf, px), bins, density=True, color=GREY, alpha=0.45)
                ax.hist(tr(bf, px), bins, density=True, histtype='step', color=FREE, lw=1.3)
                ax.hist(tr(bx, px), bins, density=True, histtype='step', color=FIXED, lw=1.3)
                ax.set_yticks([])
            else:
                ax.hist2d(tr(uf, px), tr(uf, py), bins=NB, range=[lim[px], lim[py]], cmap='Greys', norm=LogNorm(),
                          alpha=0.5, rasterized=True)
                ax.scatter(tr(bf, px), tr(bf, py), s=0.4, color=FREE, alpha=0.35, lw=0, rasterized=True)
                ax.scatter(tr(bx, px), tr(bx, py), s=0.4, color=FIXED, alpha=0.35, lw=0, rasterized=True)
                rf = spearmanr(tr(bf, px), tr(bf, py))[0]; rx = spearmanr(tr(bx, px), tr(bx, py))[0]
                ax.text(0.04, 0.96, f'{rf:+.2f}', transform=ax.transAxes, ha='left', va='top', fontsize=6, color=FREE,
                        bbox=dict(fc='white', ec='none', alpha=0.7, pad=0.5))
                ax.text(0.04, 0.80, f'{rx:+.2f}', transform=ax.transAxes, ha='left', va='top', fontsize=6, color=FIXED,
                        bbox=dict(fc='white', ec='none', alpha=0.7, pad=0.5))
                ax.set_ylim(lim[py])
            ax.set_xlim(lim[px])
            ax.tick_params(labelsize=6, length=2)
            if i == k - 1:
                ax.set_xlabel(LAB[px], fontsize=7)
            else:
                ax.set_xticklabels([])
            if j == 0 and i > 0:
                ax.set_ylabel(LAB[py], fontsize=7)
            else:
                ax.set_yticklabels([])
    h = [plt.Rectangle((0, 0), 1, 1, color=GREY, alpha=0.45, label=f'usable, $l$ free ({len(uf):,})'),
         plt.Line2D([], [], color=FREE, lw=1.5, label=f'best 1 %, $l$ free (KGE ≥ {bf.kge_free.min():.3f})'),
         plt.Line2D([], [], color=FIXED, lw=1.5, label=f'best 1 %, $l$ = 0.5 (KGE ≥ {bx.kge_fixed.min():.3f})')]
    fig.legend(handles=h, loc='upper right', bbox_to_anchor=(0.97, 0.95), fontsize=8, frameon=False,
               title=f'{ly}: same 1 000 000 parameter sets,\n$l$ free and $l$ = 0.5\nnumbers: Spearman ρ in the best 1 %',
               title_fontsize=8)
    fig.subplots_adjust(wspace=0.08, hspace=0.08)
    fig.savefig(out, bbox_inches='tight')
    print('wrote', out)
    for p in PARAMS:
        q = lambda s: np.percentile(tr(s, p), [5, 50, 95])
        f5, f50, f95 = q(bf); x5, x50, x95 = q(bx)
        print(f'  {ly} {p}: best 1 % l free median {f50:.3g} [{f5:.3g}, {f95:.3g}]   l = 0.5 median {x50:.3g} [{x5:.3g}, {x95:.3g}]')
    both = bf.index.intersection(bx.index)
    print(f'  {ly}: best 1 % sets shared by both sweeps: {len(both)} of {len(bf)} ({len(both) / len(bf):.0%})')


def main():
    for ly, fig in (('Ly1', 'S6'), ('Ly2', 'S7')):
        corner(ly, os.path.join(RES, f'fig{fig}_corner_l05_{ly.lower()}.png'))


if __name__ == '__main__':
    main()
