#!/usr/bin/env python3
"""
Plot the reported min/max l (tortuosity/pore-connectivity parameter) values
per soil type from the "Sensitivity analysis" sheet of the literature
database. Soil type is read from the "soil types" column, l min/max from the
"l min"/"l max" columns.
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # 3.2-3.3_Literature_review/
XLSX_PATH = os.path.join(BASE_DIR, "data", "Literature Review_summary.xlsx")
SHEET_NAME = "Sensitivity analysis"
SOIL_COL = "soil types"
LMIN_COL = "l min"
LMAX_COL = "l max"
HET_COL = "heterogeneity: homogeneous heterogenous: different l values?"
STUDY_COL = "modeling or experimental study"
PLOT_DIR = os.path.join(BASE_DIR, "results")

# Columns whose Excel cells are merged across every row that belongs to the
# same study (a study can span several rows: one per soil type / l value it
# reports). openpyxl/pandas only populate the top cell of a merged range, so
# these must be forward-filled after loading or every row but the first of a
# study reads as missing.
MERGED_STUDY_COLS = [
    "author", "year", "name of paper", "experimental scale",
    STUDY_COL, HET_COL, "sensitivity analysis conducted",
    "methods used for sensitivity analysis", "l parameter included / discussed",
    "doi/ reference",
]

# Recognized single-word soil texture keywords, checked in this order so
# compound classes (e.g. "sandy loam") are matched before falling back to a
# single component.
COMPOUND_PATTERNS = [
    ("silty clay loam", r"silty\s*clay\s*loam"),
    ("sandy clay loam", r"sandy\s*clay\s*loam"),
    ("clay loam", r"clay\s*loam"),
    ("silty clay", r"silty\s*clay(?!\s*loam)"),
    ("sandy clay", r"sandy\s*clay(?!\s*loam)"),
    ("sandy loam", r"sandy\s*loam"),
    ("silt loam", r"silt(?:y)?\s*loam"),
    ("loamy sand", r"loamy\s*sand"),
    ("silty sand", r"silty\s*sand|sandy\s*silt"),
]
SIMPLE_KEYWORDS = ["sand", "silt", "clay", "loam", "peat"]

# Texture classes ordered coarse -> fine (particle size decreasing).
# "peat" is organic, not a mineral texture; placed last as its own class.
TEXTURE_ORDER = [
    "sand", "silty sand", "loamy sand", "sandy loam", "sandy clay loam",
    "loam", "silt loam", "silt", "clay loam", "silty clay loam",
    "sandy clay", "silty clay", "clay", "peat",
]

NON_SOIL_MARKERS = [
    "hele-shaw", "concrete", "man made porous medium", "granodiorite",
    "bedrock", "sapolite", "fractured", "no actual soil", "no specific soil",
]


def classify_soil(text):
    if not isinstance(text, str) or not text.strip():
        return None
    t = text.lower()
    if any(marker in t for marker in NON_SOIL_MARKERS):
        return None
    for label, pattern in COMPOUND_PATTERNS:
        if re.search(pattern, t):
            return label
    hits = [kw for kw in SIMPLE_KEYWORDS if re.search(rf"\b{kw}\w*", t)]
    if len(hits) == 1:
        return hits[0]
    if len(hits) > 1:
        # Multiple distinct textures named with no single dominant one
        # (e.g. "sand, loamy sand, loam, clay, silt") -> not a single class.
        return None
    return None


def to_numeric(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def classify_heterogeneity(text):
    if not isinstance(text, str) or not text.strip():
        return None
    t = text.lower()
    if "heterog" in t:
        return "heterogeneous"
    if "homog" in t:
        return "homogeneous"
    return None


def classify_study_type(text):
    if not isinstance(text, str) or not text.strip():
        return None
    t = text.lower()
    has_exp = "experimental" in t
    has_model = "model" in t  # matches "model(l)ing", "model(l)ed", "modelling"
    if has_exp and has_model:
        return "experimental + modelling"
    if has_model:
        return "modelling only"
    if has_exp:
        return "experimental only"
    return None


def load_data():
    """Load the sheet and return one row per study with a classified soil
    texture and cleaned (lo <= hi) l-value bounds."""
    df = pd.read_excel(XLSX_PATH, sheet_name=SHEET_NAME)
    df.columns = [str(c).strip() for c in df.columns]
    # Study-level metadata cells are merged in the source workbook whenever a
    # study spans multiple rows (e.g. several soil types or l values it
    # reports); openpyxl/pandas only populate the top cell of a merged range,
    # so forward-fill to attribute every row to its study.
    df[MERGED_STUDY_COLS] = df[MERGED_STUDY_COLS].ffill()
    # The same paper can be entered twice under differently written authors (Dettmann et al., 2014); keep only
    # the rows of the later, fuller entry of each paper title.
    key = df["name of paper"].astype(str).str.lower().str.replace(r"[^a-z0-9]", "", regex=True)
    keep_author = df.groupby(key)["author"].transform("last")
    df = df[(df["author"] == keep_author) | df["name of paper"].isna()].copy()
    df["soil_class"] = df[SOIL_COL].apply(classify_soil)
    df["heterogeneity_class"] = df[HET_COL].apply(classify_heterogeneity)
    df["study_class"] = df[STUDY_COL].apply(classify_study_type)
    df["l_min"] = df[LMIN_COL].apply(to_numeric)
    df["l_max"] = df[LMAX_COL].apply(to_numeric)

    data = df.dropna(subset=["l_min", "l_max"]).copy()
    # Guard against rows where min/max were entered swapped.
    data["l_lo"] = data[["l_min", "l_max"]].min(axis=1)
    data["l_hi"] = data[["l_min", "l_max"]].max(axis=1)
    data["l_mid"] = (data["l_lo"] + data["l_hi"]) / 2
    data["l_dev"] = (data["l_mid"] - 0.5).abs()

    author = data["author"].astype(str).str.strip()
    year = data["year"].apply(lambda y: f"{int(y)}" if pd.notna(y) else "n.d.")
    data["study_label"] = author + " (" + year + ")"
    return data


def soil_color_map():
    """Coarse (sand) -> fine (clay) texture gradient, peat colored separately
    since it is organic rather than a mineral texture."""
    mineral = [t for t in TEXTURE_ORDER if t != "peat"]
    cmap = plt.get_cmap("YlOrBr")
    shades = cmap(np.linspace(0.25, 0.95, len(mineral)))
    colors = dict(zip(mineral, shades))
    colors["peat"] = "#2f4d2f"
    return colors


def texture_order_present(index):
    return [s for s in reversed(TEXTURE_ORDER) if s in index]


def soil_type_summary(data, ascending=False):
    """One row per soil texture: overall min/max l value reported across all
    studies of that texture, plus the number of studies (n)."""
    data = data.dropna(subset=["soil_class"])
    summary = (
        data.groupby("soil_class")
        .agg(l_min=("l_lo", "min"), l_max=("l_hi", "max"), n=("l_lo", "size"))
    )
    order = texture_order_present(summary.index)
    if ascending:
        order = list(reversed(order))  # coarse -> fine, top -> bottom in a table
    return summary.loc[order]


def save_min_max_table(data):
    summary = soil_type_summary(data, ascending=True)
    out_path = os.path.join(PLOT_DIR, "l_min_max_by_soil_type.csv")
    summary.to_csv(out_path, index_label="soil_type")
    print(f"\nSaved: {out_path}")
    print(summary)
    return out_path


def plot_range_by_soil_type(data, colors):
    summary = soil_type_summary(data, ascending=False)
    summary = summary.rename(columns={"l_min": "l_lo", "l_max": "l_hi"})

    print(summary)

    fig, ax = plt.subplots(figsize=(9, max(4, 0.5 * len(summary))))
    y_pos = np.arange(len(summary))
    widths = summary["l_hi"] - summary["l_lo"]
    bar_colors = [colors[s] for s in summary.index]
    ax.barh(y_pos, widths, left=summary["l_lo"], height=0.6,
            color=bar_colors, edgecolor="black", alpha=0.9)

    for y, (lo, hi, n) in zip(y_pos, summary[["l_lo", "l_hi", "n"]].values):
        ax.text(hi + 0.15, y, f"n={int(n)}", va="center", fontsize=9)

    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"{s} " for s in summary.index], fontsize=10)
    ax.set_xlabel("l value (min to max reported)", fontsize=12)
    ax.set_title("Reported min/max l values by soil type\n(literature database, Sensitivity analysis sheet)",
                 fontsize=12)
    ax.axvline(0.5, color="crimson", linewidth=1.2, linestyle="--", label="Mualem default (l = 0.5)")
    ax.axvline(0, color="gray", linewidth=0.8, linestyle=":")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()

    out_path = os.path.join(PLOT_DIR, "l_min_max_by_soil_type.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)


def plot_dumbbell_by_study(data, colors):
    data = data.dropna(subset=["soil_class"]).copy()
    order = texture_order_present(data["soil_class"].unique())
    # Sort rows: grouped by texture (coarse -> fine, bottom to top), then by
    # midpoint l value within each texture group.
    data["texture_rank"] = data["soil_class"].apply(order.index)
    data = data.sort_values(["texture_rank", "l_mid"], ascending=[True, True])
    data = data.iloc[::-1].reset_index(drop=True)  # texture_rank 0 (sand) on top

    n = len(data)
    fig, ax = plt.subplots(figsize=(10, max(6, 0.28 * n)))
    y_pos = np.arange(n)
    bar_colors = [colors[s] for s in data["soil_class"]]

    ax.hlines(y_pos, data["l_lo"], data["l_hi"], color=bar_colors, linewidth=3, alpha=0.9)
    ax.scatter(data["l_lo"], y_pos, color=bar_colors, edgecolor="black", s=18, zorder=3)
    ax.scatter(data["l_hi"], y_pos, color=bar_colors, edgecolor="black", s=18, zorder=3)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(data["study_label"], fontsize=7)
    ax.set_xlabel("l value (min to max reported)", fontsize=12)
    ax.set_title("Reported l value range per study, grouped by soil texture\n"
                 "(coarse → fine, top → bottom; peat last)", fontsize=12)
    ax.axvline(0.5, color="crimson", linewidth=1.2, linestyle="--", label="Mualem default (l = 0.5)")
    ax.axvline(0, color="gray", linewidth=0.8, linestyle=":")
    ax.grid(True, axis="x", alpha=0.3)

    # Draw a separator line + texture label each time the group changes.
    boundaries = np.where(data["texture_rank"].values[:-1] != data["texture_rank"].values[1:])[0]
    for b in boundaries:
        ax.axhline(b + 0.5, color="black", linewidth=0.6, alpha=0.4)
    for rank in data["texture_rank"].unique():
        rows = np.where(data["texture_rank"].values == rank)[0]
        mid_y = rows.mean()
        ax.text(ax.get_xlim()[1] * 0.98, mid_y, order[rank], va="center", ha="right",
                fontsize=8, style="italic", color="dimgray")

    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()

    out_path = os.path.join(PLOT_DIR, "l_range_dumbbell_by_study.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)


def plot_box_by_soil_type(data, colors, soils=None, out_name="l_boxplot_by_soil_type.png",
                           title_suffix="", figsize=None, axis_break=None, xtick_rotation=0,
                           ylim=None):
    """Boxplot of study midpoints by soil texture. Pass `soils` (a subset of
    TEXTURE_ORDER) to restrict the figure to just those classes, e.g.
    ["sand", "silt", "clay"]. Pass `axis_break=(lo, hi)` to cut out the empty
    gap between `lo` and `hi` with a broken y-axis (two stacked panels +
    diagonal break marks), so a couple of far-flung high points don't
    compress the rest of the distribution. Alternatively pass `ylim=(lo, hi)`
    to keep a single linear axis: points outside it are drawn as labelled
    arrows at the top or bottom edge instead."""
    data = data.dropna(subset=["soil_class"])
    order = texture_order_present(data["soil_class"].unique())
    order = list(reversed(order))  # coarse -> fine, left -> right on a box plot
    if soils is not None:
        order = [s for s in order if s in soils]

    groups = [data.loc[data["soil_class"] == s, "l_mid"] for s in order]
    counts = [len(g) for g in groups]

    if figsize is None:
        figsize = (max(4, 1.6 * len(order)), 6.5)

    if axis_break is None:
        fig, ax_bottom = plt.subplots(figsize=figsize)
        ax_top = None
    else:
        fig, (ax_top, ax_bottom) = plt.subplots(
            2, 1, sharex=True, figsize=figsize,
            gridspec_kw={"height_ratios": [1, 3.5], "hspace": 0.08})

    axes = [ax_bottom] if ax_top is None else [ax_top, ax_bottom]
    for ax in axes:
        bp = ax.boxplot(groups, positions=np.arange(len(order)), widths=0.55,
                         patch_artist=True, showfliers=False)
        for patch, s in zip(bp["boxes"], order):
            patch.set_facecolor(colors[s])
            patch.set_alpha(0.85)
        for median in bp["medians"]:
            median.set_color("black")

    rng = np.random.default_rng(0)
    for i, (s, g) in enumerate(zip(order, groups)):
        jitter = rng.uniform(-0.12, 0.12, size=len(g))
        x = i + jitter
        for ax in axes:
            ax.scatter(x, g.values, color="black", s=14, alpha=0.6, zorder=3)

    ax_bottom.set_xticks(np.arange(len(order)))
    # wrap long two-word class names so neighbouring labels don't overlap
    names = [s.replace(" ", "\n", 1) if len(s) > 10 else s for s in order]
    ax_bottom.set_xticklabels([f"{s}\n(n={c})" for s, c in zip(names, counts)], fontsize=9,
                               rotation=xtick_rotation,
                               ha="right" if xtick_rotation else "center")
    if ax_top is not None:
        fig.supylabel("midpoint of reported l range", fontsize=12, x=0.07)
    else:
        ax_bottom.set_ylabel("midpoint of reported l range", fontsize=12)
    ax_bottom.axhline(0.5, color="crimson", linewidth=1.2, linestyle="--", label=r"$l$ = 0.5")
    ax_bottom.axhline(0, color="gray", linewidth=0.8, linestyle=":")
    legend_ax = ax_top if ax_top is not None else ax_bottom
    handles, labels = ax_bottom.get_legend_handles_labels()
    legend_ax.legend(handles, labels, loc="upper right", fontsize=9)
    for ax in axes:
        ax.grid(True, axis="y", alpha=0.3)

    if ylim is not None:
        lo, hi = ylim
        ax_bottom.set_ylim(lo, hi)
        for i, g in enumerate(groups):
            off = [(v, hi, hi - 1.3, "bottom") for v in g.values[g.values > hi]] + \
                  [(v, lo, lo + 1.3, "top") for v in g.values[g.values < lo]]
            for v, edge, ytext, va in off:
                ax_bottom.annotate(f"{v:.3g}", xy=(i, edge), xytext=(i, ytext), ha="center", va=va,
                                   fontsize=9, arrowprops=dict(arrowstyle="->", lw=1, color="black"),
                                   annotation_clip=False)
        ax_bottom.legend(handles, labels, loc="upper center", fontsize=9)

    if axis_break is not None:
        lo, hi = axis_break
        all_vals = np.concatenate([g.values for g in groups])
        ax_bottom.set_ylim(all_vals.min() - 1, lo)
        ax_top.set_ylim(hi, all_vals.max() + 1)

        ax_bottom.spines["top"].set_visible(False)
        ax_top.spines["bottom"].set_visible(False)
        ax_top.tick_params(bottom=False)

        d = 0.5  # diagonal break-mark size, in axes-fraction units
        kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12, linestyle="none",
                      color="k", mec="k", mew=1, clip_on=False)
        ax_top.plot([0, 1], [0, 0], transform=ax_top.transAxes, **kwargs)
        ax_bottom.plot([0, 1], [1, 1], transform=ax_bottom.transAxes, **kwargs)

    fig.tight_layout()

    out_path = os.path.join(PLOT_DIR, out_name)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)


def _deviation_box(ax, data, group_col, order, palette, title):
    groups = [data.loc[data[group_col] == g, "l_dev"].dropna() for g in order]
    counts = [len(g) for g in groups]

    bp = ax.boxplot(groups, positions=np.arange(len(order)), widths=0.5,
                     patch_artist=True, showfliers=False)
    for patch, color in zip(bp["boxes"], palette):
        patch.set_facecolor(color)
        patch.set_alpha(0.85)
    for median in bp["medians"]:
        median.set_color("black")

    rng = np.random.default_rng(0)
    for i, g in enumerate(groups):
        jitter = rng.uniform(-0.1, 0.1, size=len(g))
        ax.scatter(np.full(len(g), i) + jitter, g, color="black", s=16,
                   alpha=0.6, zorder=3)

    ax.set_xticks(np.arange(len(order)))
    ax.set_xticklabels([f"{g}\n(n={c})" for g, c in zip(order, counts)], fontsize=9)
    ax.set_title(title, fontsize=11)
    ax.grid(True, axis="y", alpha=0.3)

    print(f"\n--- {title} ---")
    print(data.groupby(group_col)["l_dev"].agg(["count", "mean", "median"]).reindex(order))


def plot_deviation_by_group(data):
    """|l_midpoint - 0.5| (deviation from the Mualem default) grouped by soil
    heterogeneity and by whether the study was experimental, modelling, or
    both. A study can appear in both panels; the two groupings are
    independent classifications of the same rows."""
    het_order = ["homogeneous", "heterogeneous"]
    het_palette = ["#4C72B0", "#DD8452"]

    study_order = [s for s in ["modelling only", "experimental + modelling", "experimental only"]
                   if (data["study_class"] == s).any()]
    study_palette = ["#55A868", "#4C72B0", "#DD8452"][:len(study_order)]

    fig, axes = plt.subplots(1, 2, figsize=(11, 6), sharey=True)

    _deviation_box(axes[0], data, "heterogeneity_class", het_order, het_palette,
                    "By soil heterogeneity")
    _deviation_box(axes[1], data, "study_class", study_order, study_palette,
                    "By study type")

    axes[0].set_ylabel("|l midpoint − 0.5|  (deviation from Mualem default)", fontsize=11)
    fig.suptitle("Deviation of reported l values from the default (l = 0.5)\n"
                 "(literature database, Sensitivity analysis sheet)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    out_path = os.path.join(PLOT_DIR, "l_deviation_by_heterogeneity_and_study_type.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {out_path}")
    plt.close(fig)


def main():
    data = load_data()
    colors = soil_color_map()
    os.makedirs(PLOT_DIR, exist_ok=True)

    # Fig. 3 of the manuscript. The other plot functions above are exploratory and not called.
    plot_box_by_soil_type(data, colors, out_name="fig3_l_by_soil_type.png", figsize=(11, 4.6), ylim=(-4, 6.5))


if __name__ == "__main__":
    main()
