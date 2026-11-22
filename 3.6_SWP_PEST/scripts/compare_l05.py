#!/usr/bin/env python3
"""What fixing l = 0.5 costs: the l-free 1M sweeps against the same parameter sets run with l = 0.5.

Pablo Ortega's paired sweeps (results/l05_1M/, run_chain.sh) drew 1,000,000 parameter sets with the
seed and bounds of the l-free sweep of each lysimeter and ran them with l fixed at 0.5, so row i of
both sweeps has the same theta_r, theta_s, alpha, n and Ks and differs only in l. paired.csv holds
both KGE values per row.

Usable as in Fig. 6: all 1,216 days complete and not flagged by PEST++. Non-physical runs
(infiltration > 1.01 x rain) are removed: on the l = 0.5 side from each run's water balance, on the
l-free side from the rerun of the best 2,000 runs (results/ly2_1M/; none in Ly1).

Writes results/figS3_l05_paired.png (Fig. S3) and prints the numbers quoted in Sect. 3.6.
"""
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(os.path.dirname(HERE), 'results')
EDGES = np.arange(-2, 5.01, 0.5)
MID = 0.5 * (EDGES[:-1] + EDGES[1:])
STYLE = {'Ly1': dict(ls='-', marker='o', mfc='black'), 'Ly2': dict(ls='--', marker='D', mfc='white')}


def load(ly):
    d = pd.read_csv(os.path.join(RES, 'l05_1M', ly.lower(), 'paired.csv.gz'), index_col=0)
    d['ok_fixed'] &= ~d.nonphysical_fixed
    if ly == 'Ly2':
        s = pd.read_csv(os.path.join(RES, 'ly2_1M', 'sweep_scores.csv.gz'))
        d['ok_free'] &= ~s.nonphysical.to_numpy()
    d.loc[~d.ok_free, 'kge_free'] = np.nan
    d.loc[~d.ok_fixed, 'kge_fixed'] = np.nan
    d['delta'] = d.kge_fixed - d.kge_free
    d['cls'] = np.clip(np.digitize(d.l_free, EDGES) - 1, 0, len(MID) - 1)
    return d


def report(ly, d):
    n = len(d)
    free, fixed = d[d.ok_free], d[d.ok_fixed]
    both = d[d.ok_free & d.ok_fixed]
    print(f'\n===== {ly}')
    print(f'usable: l free {len(free) / n:.1%}, l = 0.5 {len(fixed) / n:.1%} '
          f'(non-physical removed at l = 0.5: {int(d.nonphysical_fixed.sum()):,})')
    bf, bx = free.kge_free.idxmax(), fixed.kge_fixed.idxmax()
    print(f'best KGE: l free {free.kge_free.max():.3f} (l = {d.l_free[bf]:.2f}); '
          f'l = 0.5 {fixed.kge_fixed.max():.3f}; difference {free.kge_free.max() - fixed.kge_fixed.max():.3f}')
    print(f'best run of the l-free sweep at l = 0.5: KGE {d.kge_fixed[bf]:.3f}')
    print(f'complete in both: {len(both):,}; median paired change {both.delta.median():+.4f}; '
          f'l = 0.5 better in {(both.delta > 0).mean():.1%}')
    for q in (0.99, 0.999):
        cut = free.kge_free.quantile(q)
        top = d[d.ok_free & (d.kge_free >= cut)]
        tw = top[top.ok_fixed]
        print(f'best {1 - q:.1%} of the l-free sweep (n = {len(top)}, KGE >= {cut:.3f}): '
              f'at l = 0.5 {len(tw) / len(top):.0%} complete, median KGE {tw.kge_fixed.median():.3f}, '
              f'median change {tw.delta.median():+.3f} [IQR {tw.delta.quantile(.25):+.3f}, {tw.delta.quantile(.75):+.3f}]; '
              f'runs reaching this KGE with l = 0.5: {int((fixed.kge_fixed >= cut).sum())}')
    g = d.groupby('cls')
    tab = pd.DataFrame({'l': MID, 'best_free': g.kge_free.max().to_numpy(), 'best_twin_at_0.5': g.kge_fixed.max().to_numpy()})
    print(tab.round(3).to_string(index=False))
    return tab


def main():
    data = {ly: load(ly) for ly in ('Ly1', 'Ly2')}
    tabs = {ly: report(ly, d) for ly, d in data.items()}

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4.3), dpi=300)
    for ly, d in data.items():
        st = STYLE[ly]; t = tabs[ly]
        a1.plot(t.l, t.best_free, color='black', ms=5, label=f'{ly}: $l$ free', **st)
        a1.plot(t.l, t['best_twin_at_0.5'], color='#c0392b', ms=5, label=f'{ly}: same sets, $l$ = 0.5',
                **{**st, 'mfc': '#c0392b' if st['mfc'] == 'black' else 'white'})
        free = d[d.ok_free]
        top = d[d.ok_free & d.ok_fixed & (d.kge_free >= free.kge_free.quantile(0.99))]
        g = top.groupby('cls').delta
        q = pd.DataFrame({'m': g.median(), 'lo': g.quantile(.25), 'hi': g.quantile(.75)}).reindex(range(len(MID)))
        a2.fill_between(MID, q.lo, q.hi, color='0.5' if ly == 'Ly1' else '#c0392b', alpha=0.2, lw=0)
        a2.plot(MID, q.m, color='black', ms=5, label=f'{ly}: best 1 % of the $l$-free sweep', **st)
    for ax in (a1, a2):
        ax.axvline(0.5, color='red', ls=':', lw=1.2)
        ax.set_xlabel('$l$ of the $l$-free run [-]')
        ax.set_xlim(-2.2, 5)
    a1.set_ylabel('best KGE in class of $l$')
    a1.set_title('(a)', loc='left', fontsize=12)
    a1.legend(fontsize=7.5, loc='center right', ncol=2, framealpha=0.9)
    a2.axhline(0, color='0.4', lw=1)
    a2.set_ylabel('KGE($l$ = 0.5) − KGE($l$ free)')
    a2.set_title('(b)', loc='left', fontsize=12)
    a2.legend(fontsize=7.5, loc='lower right', framealpha=0.9)
    fig.tight_layout()
    out = os.path.join(RES, 'figS3_l05_paired.png')
    fig.savefig(out, bbox_inches='tight')
    print(f'\nwrote {out}')


if __name__ == '__main__':
    main()
