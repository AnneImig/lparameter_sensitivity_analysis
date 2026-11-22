"""Figure 4 (Sect. 3.4) and Figs. S: Morris r-convergence (mu* and sigma/mu* against r), one per objective.

    python fig4_morris_convergence.py

Reads the 10-level runs' elementary_effects.csv (failed and non-physical EEs out,
same as analyze.py). Writes ../results/fig4_morris_convergence.png (KGE = Fig. 4, the measure of the sweeps) and
figS_morris_convergence_<obj>.png for the other objectives (Ly1 over Ly2) and, in each
lysimeter's p10 figures folder, r_convergence_<obj>.png (that lysimeter only).
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

RES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
OUT = RES
PARAMS = ['thr1', 'ths1', 'alf1', 'vgn1', 'ks1', 'l1']
LABEL = {'thr1': r'$\theta_r$', 'ths1': r'$\theta_s$', 'alf1': r'$\alpha$',
         'vgn1': r'$n$', 'ks1': r'$K_s$', 'l1': r'$l$'}
# Okabe-Ito, colour-blind safe, one fixed colour per parameter
COLOUR = {'thr1': '#E69F00', 'ths1': '#0072B2', 'alf1': '#009E73',
          'vgn1': '#D55E00', 'ks1': '#56B4E9', 'l1': '#CC79A7'}
RS = [10, 20, 50, 100, 200, 500, 1000, 1500, 2000]
OBJ = {'q': 'daily Q', 'kge': 'KGE', 'lq': 'daily log10 Q', 'mvol': 'monthly volume', 'cum': 'cumulative volume'}
# mu* = mean |change in the objective| per full parameter range; objectives are
# sums of squared errors against measured outflow (analyze.py), KGE unitless
YLABEL = {'q': r'$\mu^*$, daily outflow error  [(L d$^{-1}$)$^2$]',
          'kge': r'$\mu^*$, KGE  [–]',
          'lq': r'$\mu^*$, log$_{10}$ daily outflow error  [–]',
          'mvol': r'$\mu^*$, monthly outflow error  [L$^2$]',
          'cum': r'$\mu^*$, cumulative outflow error  [L$^2$]'}
INK, MUTED, RULE = '#1d2421', '#6b7570', '#e4e7e5'

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Helvetica Neue', 'Arial', 'DejaVu Sans'],
    'font.size': 10, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': MUTED,
    'ytick.color': MUTED, 'axes.linewidth': 0.8, 'mathtext.fontset': 'stixsans',
    'savefig.dpi': 200, 'savefig.bbox': 'tight', 'savefig.facecolor': 'white',
})


def table(ly, obj):
    ee = pd.read_csv(f'{RES}/{ly}_morris_p10/results/model_{ly}/morris_swp/analysis/elementary_effects.csv')
    ee = ee[~ee.failed & ~ee.nonphysical].dropna(subset=[obj]).sort_values('traj')
    rows = []
    for p in PARAMS:
        s = ee[ee.parameter == p]
        for r in RS:
            x = s.loc[s.traj < r, obj].to_numpy(float)
            row = {'param': p, 'r': r, 'mu': np.abs(x).mean(), 'ratio': x.std(ddof=1) / np.abs(x).mean()}
            rows.append(row)
    return pd.DataFrame(rows)


def log_r(ax):
    ax.set_xscale('log')
    ax.xaxis.set_major_locator(FixedLocator([10, 20, 50, 100, 200, 500, 1000, 2000]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{int(v):,}'))
    ax.set_xlim(8, 2600)
    ax.set_xlabel('Number of trajectories, r')


def clean(ax, grid='y'):
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis=grid, color=RULE, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=3)


def end_labels(ax, d, col, fmt=None, xpad=1.08):
    """Label each line at its right end; nudge labels apart so none overlap."""
    last = d[d.r == RS[-1]].set_index('param')[col]
    lo, hi = ax.get_ylim()
    log = ax.get_yscale() == 'log'
    tr = (lambda v: np.log10(v)) if log else (lambda v: v)
    inv = (lambda v: 10 ** v) if log else (lambda v: v)
    gap = (tr(hi) - tr(lo)) * 0.055
    order = last.sort_values().index
    orig = np.array([tr(last[p]) for p in order])
    ys = orig.copy()
    for _ in range(20):  # push apart, then re-centre on the line ends
        for i in range(1, len(ys)):
            ys[i] = max(ys[i], ys[i - 1] + gap)
        ys += (orig - ys).mean()
    for i in range(1, len(ys)):
        ys[i] = max(ys[i], ys[i - 1] + gap)
    for p, y in zip(order, ys):
        ax.text(RS[-1] * xpad, inv(y), LABEL[p], color=COLOUR[p], va='center', fontsize=11, fontweight='bold')


def ratio_panel(ax, d, lw=1.8, marker=None):
    ax.axhspan(0, 1, color='#f3f5f4', zorder=0)
    ax.axhline(1, color=MUTED, lw=0.8, ls=(0, (4, 3)))
    for p in PARAMS:
        s = d[d.param == p]
        ax.plot(s.r, s.ratio, color=COLOUR[p], lw=lw, marker=marker, ms=3.5)
    ax.set_ylabel(r'$\sigma / \mu^*$')
    ax.set_ylim(0, max(1.5, d.ratio.max() * 1.08))


def figure(lys, obj):
    fig, axes = plt.subplots(len(lys), 2, figsize=(10, 3.5 * len(lys)), sharex=True, squeeze=False)
    letters = iter('abcd')
    for row, ly in zip(axes, lys):
        d = table(ly, obj)
        k = int(np.log10(d.mu.max()) // 3 * 3) if d.mu.max() >= 1e5 else 0
        d['mu'] /= 10 ** k
        a1, a2 = row
        for p in PARAMS:
            s = d[d.param == p]
            a1.plot(s.r, s.mu, color=COLOUR[p], lw=1.8, marker='o', ms=3)
        a1.set_ylim(0, d.mu.max() * 1.08)
        a1.set_ylabel(YLABEL[obj].replace('[', rf'[$10^{{{k}}}$ ') if k else YLABEL[obj])
        a1.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:,.0f}' if v >= 10 else f'{v:.2g}'))
        ratio_panel(a2, d, marker='o')
        for ax in row:
            log_r(ax); clean(ax)
        end_labels(a1, d, 'mu'); end_labels(a2, d, 'ratio')
        name = ly.capitalize()
        a1.set_title(f'({next(letters)}) {name}: influence', loc='left', fontsize=11, color=INK)
        a2.set_title(f'({next(letters)}) {name}: non-linearity or interaction', loc='left', fontsize=11, color=INK)
    for ax in axes[:-1].ravel():
        ax.set_xlabel('')
    fig.tight_layout(w_pad=4, h_pad=2)
    return fig


for obj in OBJ:
    fig = figure(('ly1', 'ly2'), obj)
    name = 'fig4_morris_convergence.png' if obj == 'kge' else f'figS_morris_convergence_{obj}.png'
    fig.savefig(os.path.join(OUT, name))
    plt.close(fig)
    for ly in ('ly1', 'ly2'):
        fig = figure((ly,), obj)
        fig.savefig(f'{RES}/{ly}_morris_p10/results/model_{ly}/morris_swp/analysis/figures/r_convergence_{obj}.png')
        plt.close(fig)
print('done')
