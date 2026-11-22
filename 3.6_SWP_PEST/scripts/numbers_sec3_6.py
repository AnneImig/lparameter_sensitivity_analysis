#!/usr/bin/env python3
"""Numbers of Sect. 3.6 and Tables S5-S6 from the two l-free 1,000,000-run sweeps.

    python numbers_sec3_6.py      -> ../results/numbers_sec3_6.log

Usable as in Fig. 6 (fig6_fit.load): not flagged by PEST++, all 1,216 days present and, for Ly2, not
non-physical (infiltration > 1.01 x precipitation, checked for the best 2,000 runs). The numbers on fixing
l = 0.5 are written by compare_l05.py (figS3_l05_paired.log).
"""
import os
import sys

import numpy as np
from scipy.stats import spearmanr

from fig6_fit import load_both
from sweeps import RES, best_per_class

EDGES = np.arange(-2, 5.01, 0.5)          # 14 classes of l, 0.5 wide
PARAMS = ['ths1', 'vgn1', 'thr1', 'ks1', 'l1', 'alf1']


def lost(d, mask):
    m = mask.to_numpy()
    return f'{100 * (~d.ok[m]).mean():.1f}% (n={m.sum():,})'


def report(name, d):
    u = d[d.ok]
    print(f'\n===== {name}: N {len(d):,}; usable {len(u):,} ({100 * len(u) / len(d):.1f}%), lost {len(d) - len(u):,}; '
          f'best KGE {u.kge.max():.3f}; median {u.kge.median():.3f}; KGE > 0.8: {(u.kge > 0.8).sum()}')
    failed = ((d.failed_flag != 0) | (d.n_failed_obs != 0)).sum()
    print(f'  Table S5: failed (crashed or stopped before day 1216) {failed:,} ({100 * failed / len(d):.1f}%); '
          f'non-physical {len(d) - len(u) - failed}')
    print('  lost n>2 & l<-1:', lost(d, (d.vgn1 > 2) & (d.l1 < -1)), '| n<2:', lost(d, d.vgn1 < 2),
          '| n>2:', lost(d, d.vgn1 > 2))
    print('  lost n>2 & alpha>=0.05:', lost(d, (d.vgn1 > 2) & (d.alf1 >= 0.05)),
          '| n>2 & alpha<0.05:', lost(d, (d.vgn1 > 2) & (d.alf1 < 0.05)))
    m = 1 - 1 / d.vgn1
    conc = d.l1 <= 1 - 2 / m
    print(f'  Peters concavity violated: {100 * conc.mean():.1f}% of runs, lost among them '
          f'{100 * (~d.ok[conc]).mean():.1f}%; share of all losses that satisfy it '
          f'{100 * (~d.ok & ~conc).sum() / (~d.ok).sum():.1f}%')
    print('  Spearman with KGE (usable):',
          '  '.join(f'{p} {spearmanr(u[p], u.kge)[0]:+.2f}' for p in PARAMS))
    best, _, floor = best_per_class(d, EDGES)
    print(f'  best KGE per l class (equal n={floor:,}):', ' '.join(f'{b:.3f}' for b in best))
    print(f'  best with l >= 0: {u[u.l1 >= 0].kge.max():.3f}; best with 0.3 <= l <= 0.7: '
          f'{u[(u.l1 >= 0.3) & (u.l1 <= 0.7)].kge.max():.3f}')
    share_all = np.histogram(u.l1, EDGES)[0] / len(u)
    print('  Table S6: subset, n, KGE threshold, l median (IQR), share l < -1')
    rows = [('Prior', d, None), ('Usable', u, None)]
    for f in (0.10, 0.01, 0.001, 0.0001):
        k = int(np.ceil(f * len(u)))
        rows.append((f'Best {100 * f:g} %', u.nlargest(k, 'kge'), None))
    for lab, s, _ in rows:
        thr = f'{s.kge.min():.3f}' if lab.startswith('Best') else '–'
        q = s.l1.quantile([0.5, 0.25, 0.75]).to_numpy()
        print(f'    {lab:<12}{len(s):>10,}  {thr:>6}  {q[0]:.2f} ({q[1]:.2f} to {q[2]:.2f})  '
              f'{100 * (s.l1 < -1).mean():.1f}%')
    for f in (0.01, 0.001):
        b = u.nlargest(int(np.ceil(f * len(u))), 'kge')
        rel = np.histogram(b.l1, EDGES)[0] / len(b) / share_all
        print(f'  best {100 * f:g}%: n={len(b)} KGE>={b.kge.min():.3f}; l<0 {100 * (b.l1 < 0).mean():.1f}% '
              f'vs usable {100 * (u.l1 < 0).mean():.1f}%; Spearman n-l {spearmanr(b.vgn1, b.l1)[0]:+.2f}, '
              f'alpha-l {spearmanr(b.alf1, b.l1)[0]:+.2f}')
        print('    relative frequency per l class:', ' '.join(f'{r:.2f}' for r in rel))


if __name__ == '__main__':
    out = os.path.join(RES, 'numbers_sec3_6.log')
    with open(out, 'w') as fh:
        sys.stdout = fh
        for name, d in zip(('Ly1', 'Ly2'), load_both()):
            report(name, d)
    sys.stdout = sys.__stdout__
    print(open(out).read())
