# -*- coding: utf-8 -*-
"""
Figure 1 of the manuscript (Sect. 3.1): COMSOL numerical experiments for sand, silt and clay.

(a) Effective saturation profiles for l from -4 to 4 at 100 mm/yr (data/Output_tables/*_100_Seff_lambda_*.csv).
(b) Hydraulic gradient against l at 100 and 500 mm/yr (data/lambda-gradH.xlsx),
    shown for l from -4 to 4.

"""
import os

import matplotlib as mpl
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), 'data')
TABLES = os.path.join(DATA, 'Output_tables')
FIGS = os.path.join(os.path.dirname(HERE), 'results')
XLSX = os.path.join(DATA, 'lambda-gradH.xlsx')

SOILS = ['Sand', 'Clay', 'Silt']
LS = {'Sand': '-', 'Clay': ':', 'Silt': '-.'}
L_VALS = [-4, -2, -1, -0.5, 0, 0.5, 1, 2, 4]
L_MAX = 4
FLUX_COL = {100: '#1f1f1f', 500: '#9a9a9a'}
FLUX_MK = {100: 'o', 500: 's'}
CMAP = LinearSegmentedColormap.from_list('div_blue_red', ['#2a78d6', '#c3c2b7', '#e34948'])
CM = 1 / 2.54

mpl.rcParams.update({'figure.dpi': 300, 'font.size': 8, 'axes.labelsize': 8, 'legend.fontsize': 7,
                     'legend.title_fontsize': 7, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
                     'axes.edgecolor': '#555555', 'grid.color': '#e1e0d9', 'mathtext.default': 'it',
                     'savefig.facecolor': 'white'})


def load_blocks():
    """(flux in mm/yr, table of l and hydraulic gradient per soil) for each flux block of lambda-gradH.xlsx.

    The sheet's 'VZ total mass discharge {int1}' header is a mislabel from the COMSOL export: the values
    are (top(dl.H) - bot(dl.H)) / L, the hydraulic gradient, as in Output_tables/*_Q_gradH_lambda_*.csv.
    """
    raw = pd.read_excel(XLSX, sheet_name='Sheet1', header=None)
    rows = list(raw.index[raw[0] == 'lambda'])
    blocks = []
    for i, h in enumerate(rows):
        end = rows[i + 1] if i + 1 < len(rows) else len(raw)
        b = raw.loc[h + 1:end - 1]
        b = b[pd.to_numeric(b[0], errors='coerce').notna()]
        if b.empty:
            continue
        blocks.append((pd.to_numeric(b[1]).iloc[0],
                       pd.DataFrame({'lambda': pd.to_numeric(b[0]), 'gradH_sand': pd.to_numeric(b[2]),
                                     'gradH_silt': pd.to_numeric(b[3]), 'gradH_clay': pd.to_numeric(b[4])})))
    return blocks


def panel_label(ax, letter):
    """Panel letter as in Figs. 2 and 5: bold (a), (b) at the top left of the panel."""
    ax.text(0.0, 1.03, f'({letter})', transform=ax.transAxes, fontsize=10.5, fontweight='bold', va='bottom', ha='left')


def lcol(l):
    return CMAP((l - min(L_VALS)) / (max(L_VALS) - min(L_VALS)))


def panel_a(ax):
    for soil in SOILS:
        for l in L_VALS:
            d = pd.read_csv(os.path.join(TABLES, f'{soil}_100_Seff_lambda_{l:g}.csv'), skiprows=8, header=None)
            ax.plot(d.iloc[:, 1], d.iloc[:, 0] - 1, lw=1.3, ls=LS[soil], color=lcol(l), zorder=3)
    ax.set(xlim=(0, 1), ylim=(-1, 0), xlabel=r'Effective saturation $S_e$ [–]', ylabel='Depth [m]')
    ax.grid(True, lw=0.5); ax.set_axisbelow(True)
    h = [mlines.Line2D([], [], color=lcol(l), lw=1.5, label=f'{l:g}') for l in L_VALS]
    ax.legend(handles=h, title='$l$ [–]', ncols=3, loc='upper center', bbox_to_anchor=(0.56, 0.93),
              framealpha=0.95, columnspacing=0.6, handlelength=1.1, handletextpad=0.4, borderpad=0.4)
    panel_label(ax, 'a')


def panel_b(ax):
    for q, df in load_blocks():
        df = df[df['lambda'] <= L_MAX]
        for soil in SOILS:
            ax.plot(df['lambda'], df[f'gradH_{soil.lower()}'], ls=LS[soil], color=FLUX_COL[int(q)],
                    marker=FLUX_MK[int(q)], ms=3, lw=1.3, zorder=3)
    ax.axvline(0.5, color='red', ls='--', lw=1)
    ax.text(0.6, 1.25, '$l$ = 0.5', color='red', fontsize=7, va='top')
    ax.set_yscale('log')
    ax.set(xlim=(-4.3, 4.3), ylim=(1e-5, 1.5), xlabel='$l$ [–]', ylabel='Hydraulic gradient [–]')
    ax.set_xticks(range(-4, 5))
    ax.grid(True, which='major', lw=0.5); ax.set_axisbelow(True)
    soil_h = [mlines.Line2D([], [], color='black', ls=LS[s], lw=1.3, label=s) for s in SOILS]
    flux_h = [mlines.Line2D([], [], color=FLUX_COL[q], marker=FLUX_MK[q], ms=3, lw=1.3, label=f'{q} mm yr$^{{-1}}$')
              for q in (100, 500)]
    ax.legend(handles=soil_h + flux_h, ncols=2, loc='lower right', framealpha=0.95, columnspacing=0.8,
              handlelength=1.8, handletextpad=0.4, borderpad=0.4, fontsize=6.5)
    panel_label(ax, 'b')


def main():
    fig, (a, b) = plt.subplots(1, 2, figsize=(17 * CM, 7.5 * CM), gridspec_kw={'wspace': 0.32})
    panel_a(a)
    panel_b(b)
    os.makedirs(FIGS, exist_ok=True)
    out = os.path.join(FIGS, 'Figure1_Seff_gradH.png')
    fig.savefig(out, dpi=300, bbox_inches='tight')
    print('wrote', out)


if __name__ == '__main__':
    main()
