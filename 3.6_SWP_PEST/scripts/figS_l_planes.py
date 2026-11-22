#!/usr/bin/env python3
"""SI figure: best reachable KGE in the plane of l and each other parameter, for both sweeps.

Same measure as Fig. 6a, b (fig6_fit.py): best KGE of all usable runs per cell, one colour scale per
lysimeter spanning SPAN below its best run. alpha and Ks are binned logarithmically, as they were sampled.
"""
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from sweeps import RES
from fig6_fit import LE, SPAN, load_both   # Ly2: final sweep (ly2_1M), non-physical runs removed

PARTNERS = [('thr1', r'$\theta_r$ [-]', False), ('ths1', r'$\theta_s$ [-]', False),
            ('alf1', r'$\alpha$ [cm$^{-1}$]', True), ('vgn1', r'$n$ [-]', False),
            ('ks1', r'$K_s$ [cm d$^{-1}$]', True)]
NBIN = 15


def plane(d, x, log, seed=0):
    v = np.log10(d[x]) if log else d[x]
    xe = np.linspace(v.min(), v.max() + 1e-9, NBIN + 1)
    i = np.clip(np.digitize(v, xe) - 1, 0, NBIN - 1)
    j = np.clip(np.digitize(d.l1, LE) - 1, 0, len(LE) - 2)
    best = np.full((NBIN, len(LE) - 1), np.nan)
    ok = d.ok.to_numpy()
    np.fmax.at(best, (i[ok], j[ok]), d.kge.to_numpy()[ok])
    return (10 ** xe if log else xe), best


def main():
    sweeps = list(zip(('Ly1', 'Ly2'), load_both()))
    fig, axes = plt.subplots(2, 5, figsize=(15, 6.4), dpi=300, sharey=True,
                             gridspec_kw=dict(hspace=0.45, wspace=0.12))
    for r, (name, d) in enumerate(sweeps):
        top = np.nanmax(d.kge)
        for c, (x, label, log) in enumerate(PARTNERS):
            ax = axes[r, c]
            xe, best = plane(d, x, log)
            im = ax.pcolormesh(xe, LE, best.T, cmap='viridis', vmin=top - SPAN, vmax=top, shading='flat')
            if log: ax.set_xscale('log')
            ax.set(xlim=(xe[0], xe[-1]), ylim=(-2, 5), xlabel=label)
            ax.axhline(0.5, color='red', ls='--', lw=0.9)
            ax.set_title(f'({"abcdefghij"[5 * r + c]}) {name}', fontsize=10, loc='left')
        cb = fig.colorbar(im, ax=axes[r, :], shrink=0.9, extend='min', pad=0.015)
        cb.set_label('best KGE in cell')
        axes[r, 0].set_ylabel('$l$ [-]')
        axes[r, 4].text(0.97, 0.97, f'best KGE {top:.3f}', transform=axes[r, 4].transAxes,
                        ha='right', va='top', fontsize=8, color='white')
    out = os.path.join(RES, 'figS2_l_planes.png')
    fig.savefig(out, bbox_inches='tight')
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
