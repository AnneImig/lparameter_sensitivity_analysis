#!/usr/bin/env python3
"""
Two-panel pie figure summarizing the 68 sensitivity-analysis (SA) studies in
the literature database (Sensitivity analysis sheet):

  A) which SA method family each study used (one-at-a-time perturbation,
     local derivative-based sensitivity coefficients, variance-based global
     methods (Sobol/Morris), inverse-modeling/identifiability diagnostics,
     Monte Carlo sampling, or a qualitative/visual approach). This is a
     new classification of column I ("methods used for sensitivity
     analysis"), built by keyword/context matching and cross-checked
     against the specific studies named as examples of each method family
     in the manuscript text.

  B) how l itself was treated / what outcome was reported for it, reusing
     the classification already used for l_sensitivity_treatment_pie.png
     (column H "sensitivity analysis conducted" + column J "l parameter
     included / discussed").
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # 3.2-3.3_Literature_review/
XLSX_PATH = os.path.join(BASE_DIR, "data", "Literature Review_summary.xlsx")
SHEET_NAME = "Sensitivity analysis"
PLOT_DIR = os.path.join(BASE_DIR, "results")

MERGED_STUDY_COLS = [
    "author", "year", "name of paper", "experimental scale",
    "modeling or experimental study",
    "heterogeneity: homogeneous heterogenous: different l values?",
    "sensitivity analysis conducted", "methods used for sensitivity analysis",
    "l parameter included / discussed", "doi/ reference",
]

METHOD_LABELS = {
    "oat": "One-at-a-time perturbation",
    "local": "Local derivative-based\nsensitivity coefficients",
    "global": "Variance-based global\nmethods (Sobol, Morris)",
    "inverse": "Inverse-modeling /\nidentifiability diagnostics",
    "monte_carlo": "Monte Carlo sampling",
    "qualitative": "Qualitative / visual\napproaches",
}

# Studies the manuscript text names explicitly as an example of a given SA
# method family -- these are pinned rather than left to keyword matching, so
# the classification is guaranteed consistent with the citations in the text.
PINNED = {
    "s assouline": "oat", "u fischer, ma celia": "oat",
    "jarvis, n. j., jansson": "oat", "mohammadi, m. h.": "oat",
    "dury, o., fischer, u.": "local", "kelleners": "local",
    "s lambot, m javaux": "local", "rocha, d.": "local",
    "cohen, m., schwartz": "global", "larsbo, m. and jarvis": "global",
    "moua, r., lesparre": "global", "y wang, j ma, h guan": "global",
    "bitterlich": "inverse", "durner ": "inverse",
    "sprenger, m., volkmann": "inverse", "werisch": "inverse",
    "bancheri": "monte_carlo", "coppola, a., basile, a., comegna": "monte_carlo",
    "t wang, te franz, va zlotnik": "monte_carlo",
    "chevalier": "qualitative", "cuenca": "qualitative",
    "t wang, va zlotnik, j": "qualitative",
}


def classify_method(author, text):
    a = str(author).lower()
    for key, cat in PINNED.items():
        if key in a:
            return cat

    t = str(text).lower()

    def has(*kws):
        return any(k in t for k in kws)

    if has("sobol", "morris", "polynomial chaos", "elementary effect",
           "global sensitivity", "gsa"):
        return "global"
    if has("monte carlo", "random search", "realizations)", "runs each"):
        return "monte_carlo"
    if has("identifiab", "ucode", "svd", "singular value decomposition",
           "pareto-optimal", "response surface analysis", "confidence analysis",
           "pest"):
        return "inverse"
    if has("sensitivity coefficient", "jacobian", "partial derivative",
           "elasticity coefficient", "automatic differentiation",
           "derivative-based", "1% change", "by a 1%", "scaled sensitivity",
           "aggregate sensitivity measure", "local sensitivity"):
        return "local"
    if has("one-at-a-time", "one at a time", "oat", "once-at-a-time",
           "one-factor-at-a-time", "one-group-at-a-time", "ogat",
           "univariate", "one-dimensional sensitivity", "individually",
           "individual perturbation", "one parameter value", "perturbation of",
           "varied one-at-a-time", "manual parameter sensitivity",
           "varying the coefficient of variation", "varying the mean values"):
        return "oat"
    return "qualitative"


def load_sa_studies():
    df = pd.read_excel(XLSX_PATH, sheet_name=SHEET_NAME)
    df.columns = [str(c).strip() for c in df.columns]
    df[MERGED_STUDY_COLS] = df[MERGED_STUDY_COLS].ffill()
    df = df.drop_duplicates(subset=["author", "year"]).reset_index(drop=True)
    # The same paper can be entered twice under differently written authors (Dettmann et al., 2014: "Dettmann
    # et al." and "U Dettmann, M Bechtold, ..."); keep one entry per paper title, the later, fuller one.
    title = df["name of paper"].astype(str).str.lower().str.replace(r"[^a-z0-9]", "", regex=True)
    df = df[~title.duplicated(keep="last") | df["name of paper"].isna()].reset_index(drop=True)
    return df


def main():
    df = load_sa_studies()
    df["method_cat"] = [classify_method(a, t) for a, t in
                         zip(df["author"], df["methods used for sensitivity analysis"])]

    method_order = ["oat", "local", "global", "inverse", "monte_carlo", "qualitative"]
    method_counts = df["method_cat"].value_counts().reindex(method_order).fillna(0).astype(int)

    # Panel B reuses the already-published counts/labels from
    # l_sensitivity_treatment_pie.png (fixed=33, low=12, high=8, mixed=7,
    # not_quantified=9), rather than re-deriving them from column J text
    # (whose free-text wording is too heterogeneous to re-classify reliably).
    treatment_counts = pd.Series(
        # mixed: 7 -> 6 after removing the duplicate Dettmann et al. (2014) entry (moisture-dependent l)
        {"fixed": 33, "low": 12, "not_quantified": 9, "high": 8, "mixed": 6})
    treatment_labels = {
        "fixed": "Fixed at Mualem default\n(l = 0.5), excluded from SA",
        "low": "Included in SA —\nfound low / negligible sensitivity",
        "not_quantified": "Included in SA —\noutcome not quantified",
        "high": "Included in SA —\nfound high / dominant sensitivity",
        "mixed": "Included in SA —\nmixed / context-dependent\n(texture- or moisture-dependent)",
    }
    # Muted, low-saturation shades (single blue-gray hue family) instead of a
    # bright qualitative palette -- wedges are still distinguishable by value
    # but the figure reads as understated rather than colorful.
    treatment_shades = ["#3b5468", "#6a89a3", "#94b0c4", "#bcd0dc", "#dde8ee"]
    treatment_colors = dict(zip(
        ["fixed", "low", "not_quantified", "high", "mixed"], treatment_shades))
    method_shades = ["#3b5468", "#5c7a91", "#7d9cb0", "#9ebccd", "#c0d5e1", "#e2eef3"]
    method_colors = dict(zip(
        ["oat", "local", "global", "inverse", "monte_carlo", "qualitative"], method_shades))

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(17, 7.5))

    def _luminance(hex_color):
        r, g, b = (int(hex_color[i:i + 2], 16) / 255 for i in (1, 3, 5))
        return 0.299 * r + 0.587 * g + 0.114 * b

    def draw_pie(ax, counts, labels, colors, panel_letter):
        wedge_colors = [colors[k] for k in counts.index]
        wedges, _, autotexts = ax.pie(
            counts.values,
            colors=wedge_colors,
            autopct=lambda pct: f"{pct:.0f}%\n(n={int(round(pct / 100 * counts.sum()))})",
            pctdistance=0.68, startangle=90,
            wedgeprops=dict(edgecolor="white", linewidth=1.5))
        for at, wc in zip(autotexts, wedge_colors):
            at.set_color("white" if _luminance(wc) < 0.5 else "#222222")
            at.set_fontweight("bold")
            at.set_fontsize(10)
        ax.legend(wedges, [f"{labels[k]}  (n={counts[k]})" for k in counts.index],
                  loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=9.5,
                  frameon=False)
        ax.text(0.0, 1.05, f"({panel_letter})", transform=ax.transAxes,
                fontsize=14, fontweight="bold", va="bottom", ha="left")

    draw_pie(axA, method_counts, METHOD_LABELS, method_colors, "a")
    draw_pie(axB, treatment_counts, treatment_labels, treatment_colors, "b")

    fig.tight_layout()
    out_path = os.path.join(PLOT_DIR, "fig2_SA_methods_and_l_treatment.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight", transparent=True)
    print(f"Saved: {out_path}")
    plt.close(fig)

    n_total_methods = int(method_counts.sum())
    print("\nMethod family counts:")
    for k in method_order:
        print(f"  {k:12s} n={method_counts[k]:2d}  ({method_counts[k]/n_total_methods:.0%})")


if __name__ == "__main__":
    main()
