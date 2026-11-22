#!/usr/bin/env python3
"""Figure 7 (Sect. 3.7): decision chart for treating the pore-connectivity parameter l.

Rebuilt from the chart in the manuscript (Nov 2026), same layout and colours, in Times New Roman, with the
recommendations brought in line with the final results:
  - coarse soils: fit l jointly with alpha and n (alpha-l is the strongest interaction, n compensates for l in Ly1)
  - fine/organic soils: fit l jointly with alpha and Ks (alpha-l strongest in both soils, Ks-l second in Ly2)
  - calibration box: check that the best fits lie inside the l bounds and that infiltration <= precipitation

Coordinates are in image pixels (origin top left), as in the original. Edit the text in BOXES/DIAMONDS and run
    python fig7_decision_chart.py        -> fig7_decision_chart.png (300 dpi) and .svg (Inkscape)
"""
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon

HERE = os.path.dirname(os.path.abspath(__file__))
matplotlib.rcParams.update({'font.family': 'Times New Roman', 'mathtext.fontset': 'stix', 'font.size': 8.6,
                            'svg.fonttype': 'none'})   # SVG keeps text as editable text (Inkscape)

RED, ORANGE, GREEN, GREY, EDGE = '#f2a7a0', '#fbd0a0', '#c8e6c0', '#e8e8e8', '#404040'
BLUE = '#c9dcf0'   # calibration box, distinct from the orange screening box
W, H = 1876, 1555
FS = 8.6          # font size in pt for the 1876-px-wide canvas


def box(ax, x0, y0, x1, y1, text, fc, size=FS):
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle='round,pad=0,rounding_size=22',
                                fc=fc, ec=EDGE, lw=1.6, zorder=2))
    ax.text((x0 + x1) / 2, (y0 + y1) / 2, text, ha='center', va='center', fontsize=size, linespacing=1.35, zorder=3)


def diamond(ax, cx, cy, w, h, text, size=FS):
    ax.add_patch(Polygon([(cx, cy - h / 2), (cx + w / 2, cy), (cx, cy + h / 2), (cx - w / 2, cy)],
                         fc='white', ec=EDGE, lw=1.6, zorder=2))
    ax.text(cx, cy, text, ha='center', va='center', fontsize=size, linespacing=1.3, zorder=3)


def path(ax, pts, label=None, at=None, arrow=True):
    xs, ys = zip(*pts)
    ax.plot(xs, ys, color=EDGE, lw=1.6, zorder=1, solid_capstyle='butt')
    if arrow:
        (x0, y0), (x1, y1) = pts[-2], pts[-1]
        ax.annotate('', xy=(x1, y1), xytext=(x0 + 0.6 * (x1 - x0), y0 + 0.6 * (y1 - y0)),
                    arrowprops=dict(arrowstyle='-|>', color=EDGE, lw=1.6, mutation_scale=16), zorder=1)
    if label:
        ax.text(*at, label, ha='center', va='center', fontsize=FS, style='italic',
                bbox=dict(fc='white', ec='none', pad=1.5), zorder=3)


def main():
    fig = plt.figure(figsize=(W / 250, H / 250), dpi=300)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis('off')

    box(ax, 355, 55, 1380, 160, 'Van Genuchten–Mualem model of a soil layer: should $l$ be fitted?', GREY)
    diamond(ax, 867, 286, 500, 170, 'Soil texture?')
    path(ax, [(867, 160), (867, 200)])

    # coarse branch
    path(ax, [(617, 286), (230, 286), (230, 438)], 'coarse', (415, 286))
    diamond(ax, 230, 543, 400, 205, 'Does the\nprofile dry out?')
    path(ax, [(230, 646), (230, 738)], 'yes', (230, 686))
    box(ax, 37, 738, 425, 928, 'FIT $l$ jointly with α and $n$\nWide prior, incl.\nnegative values', RED)
    path(ax, [(430, 543), (483, 543), (483, 713), (633, 713), (633, 738)], 'no', (483, 618))

    # medium branch
    path(ax, [(867, 371), (867, 398), (733, 398), (733, 440)], 'medium', (825, 425), arrow=True)
    diamond(ax, 733, 543, 400, 205, 'Does the\nsite stay wet?')
    path(ax, [(733, 646), (733, 738)], 'no', (733, 686))
    box(ax, 540, 738, 927, 928, 'SCREEN $l$\nMorris + pairwise\ninteraction screen\nover a wide prior', ORANGE)
    path(ax, [(933, 543), (1068, 543), (1068, 713), (1135, 713), (1135, 738)], 'yes', (1068, 618))
    box(ax, 1008, 738, 1430, 928, 'FIX $l$ = 0.5\nReport the screen\nthat justifies it', GREEN)

    # fine / organic branch
    path(ax, [(1117, 286), (1470, 286), (1470, 438)], 'fine / organic', (1320, 286))
    box(ax, 1320, 438, 1620, 568, 'FIT $l$ jointly with α\nand $K_s$', RED)

    # screening decision
    path(ax, [(733, 928), (733, 987)])
    diamond(ax, 733, 1098, 635, 222, 'Is $l$ among the top-\nranked parameters or in\nthe dominant interaction?')
    path(ax, [(1051, 1098), (1302, 1098), (1302, 928)], 'no', (1185, 1098))
    path(ax, [(733, 1209), (733, 1285)], 'yes', (766, 1250))

    # into calibration
    path(ax, [(230, 928), (230, 1285)])
    path(ax, [(1470, 568), (1470, 1285)])
    box(ax, 70, 1285, 1663, 1500,
        'CALIBRATE $l$ within physically consistent bounds\n'
        '• Pair $l$ with its interaction partners (α in all soils, also $K_s$ in fine soils) and with $n$\n'
        '• Enforce $l$ > 1 − 2/$m$ (Peters et al., 2011) and sample $n$ and $l$ jointly, not in independent boxes\n'
        '• Keep crashed or incomplete runs as information on the feasible $n$–$l$ space',
        BLUE)


    out = os.path.join(HERE, 'fig7_decision_chart.png')
    fig.savefig(out, dpi=300, facecolor='white')
    fig.savefig(out.replace('.png', '.svg'), facecolor='white')
    print('wrote', out, 'and .svg')


if __name__ == '__main__':
    main()
