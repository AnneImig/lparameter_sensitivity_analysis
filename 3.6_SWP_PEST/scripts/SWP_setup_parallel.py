#!/usr/bin/env python3
"""
Parallel variant of create_sweep_setup.py: builds the same PEST++ SWEEP
(pestpp-swp) setup, but runs it across NUM_WORKERS local PANTHER workers
instead of one serial process. Fully standalone/independent, like
create_sweep_setup.py and create_pest_setup.py are of each other -- its
own MODEL_CHOICES/ACTIVE_MODEL/BOUNDS/select_model()/etc, duplicated
rather than imported.

Why this needed to be a separate script rather than just adding a
"parallel" flag to create_sweep_setup.py's run_sweep(): PANTHER
(pyemu.os_utils.start_workers()) copies a template directory into one
isolated directory per worker (plus one master directory) and runs each
worker's model_command from within its own copy. create_sweep_setup.py's
forward_run.py resolves its HYDRUS working directory (tmp_copy/) via an
ABSOLUTE path derived from the script's own file location (BASE_DIR),
not from whatever directory it's actually invoked from -- so under
PANTHER, every worker would still funnel through that one shared
tmp_copy/, each overwriting SELECTOR.IN/Obs_Node.out while another
worker is mid-run. Silent wrong-answer corruption, not just slowness
(create_sweep_setup.py's own docstring already warns about exactly this
for running it alongside create_pest_setup.py).

This script avoids that by keeping everything -- including its own
private tmp_copy/-equivalent model directory -- inside RESULTS_DIR, and
generating a PANTHER-aware forward_run_parallel.py (see
write_forward_run_parallel()) that resolves its HYDRUS working directory
relative to os.getcwd() at runtime, i.e. whichever worker/master
directory PANTHER actually copied it into. Since start_workers() copies
RESULTS_DIR wholesale into each worker directory, every worker
automatically gets its own private, fully isolated copy of tmp_copy/,
sweep.pst, forward_run_parallel.py, and obs_config.py -- safe to run
concurrently. Only the HYDRUS binary itself (bin/mac/hydrus or
bin/windows/H1D_CALC.exe) stays shared/read-only at the one original
project location, which is fine since nothing ever writes to it.

Parameter sets are drawn via Latin Hypercube Sampling (or plain Monte
Carlo -- see SAMPLING_METHOD) across BOUNDS, log-space for LOG_PARAMS
(alpha, k_s) so the sample covers their full orders-of-magnitude range
evenly.

Prerequisites:
    pip install pyemu scipy matplotlib
    # pestpp-swp ships alongside pestpp-ies/pestpp-glm:
    conda install -c conda-forge pestpp

Run:
    cd pest_setup
    python create_sweep_setup_parallel.py
"""

import os
import copy
import glob
import gzip
import shutil
import string
import argparse
import numpy as np
import pandas as pd
from scipy.stats import qmc
import pyemu
import matplotlib.pyplot as plt

import obs_config

# ── PATHS ────────────────────────────────────────────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
PEST_DIR    = os.path.dirname(SCRIPT_DIR)           # this part's folder; results go to PEST_DIR/results
BASE_DIR    = os.path.dirname(PEST_DIR)          # repository root -- HYDRUS_models/, bin/ and tmp_copy/ live here
MODELS_DIR  = os.path.join(BASE_DIR, 'HYDRUS_models')

# ── MODEL SELECTION (own copy, independent of create_pest_setup.py's/
# create_sweep_setup.py's) ───────────────────────────────────────────────────
MODEL_CHOICES = {
    'profile1': 'Model_Profile 1_250 cm_bromide',
    'profile2': 'Model_Profile 2_250 cm_bromide',
    # 'profile3' intentionally omitted -- see create_pest_setup.py for why
    # (its SELECTOR.IN soil-parameter table is corrupted).
    'model_ly1': 'Model_Ly1',
    # Model_Ly2 is the second lysimeter: same HYDRUS setup as Model_Ly1
    # (identical SELECTOR.IN/ATMOSPH.IN/PROFILE.DAT -- single material,
    # 201-node profile, lChem=f/lTemp=f), differing only in the measured
    # outflow it's calibrated against (COMPARE_Ly2_Qmeas*.txt vs.
    # COMPARE_Ly1_Qmeas*.txt). Its parameter ranges therefore start from
    # LY1_BOUNDS, widened only where Ly2 needs it (see LY2_BOUNDS below),
    # and every Ly1-specific path here is matched by glob
    # (COMPARE_Ly*_Qmeas*.txt) rather than by the literal Ly1 filename.
    'model_ly2': 'Model_Ly2',
    'profile2_1layer': 'Model_Profile 2_250 cm_bromide_1layer',
    'profile2_1layer_swi': 'Model_Profile 2_250 cm_bromide_1layer',
}
ACTIVE_MODEL = 'model_ly2'  # change to any MODEL_CHOICES key to switch (e.g. 'model_ly2'); or override per run with --model

# ── PARAMETER SETUP ──────────────────────────────────────────────────────────
PARAM_SHORT = ['thr', 'ths', 'alf', 'vgn', 'ks', 'l']
PARAM_FULL  = ['theta_r', 'theta_s', 'alpha', 'n', 'k_s', 'l']

# Reference/default parameter bounds (min, max) per layer -- own copy,
# independent of the other scripts' BOUNDS. Edit freely without touching them.
BOUNDS = {
    1: {'theta_r': [0.01,   0.05],    'theta_s': [0.20, 0.41],
        'alpha':   [0.00918, 0.05748], 'n':       [1.10, 2.21244],
        'k_s':     [0.00979, 0.07449], 'l':       [-2.0,5.0]},
    2: {'theta_r': [0.01,   0.05],    'theta_s': [0.20, 0.4098],
        'alpha':   [0.00888, 0.05808], 'n':       [1.10, 2.14164],
        'k_s':     [0.00594, 0.06639], 'l':       [-2.0, 5.0]},
    3: {'theta_r': [0.01,   0.05],    'theta_s': [0.20, 0.43296],
        'alpha':   [0.00868, 0.06132], 'n':       [1.10, 2.39880],
        'k_s':     [0.00767, 0.11848], 'l':       [-2.0, 5.0]},
    4: {'theta_r': [0.01,   0.05],    'theta_s': [0.20, 0.43872],
        'alpha':   [0.00826, 0.05784], 'n':       [1.10, 3.05184],
        'k_s':     [0.01036, 0.19276], 'l':       [-2.0, 5.0]},
    5: {'theta_r': [0.01,   0.05],    'theta_s': [0.20, 0.49920],
        'alpha':   [0.00624, 0.04200], 'n':       [1.10, 5.24904],
        'k_s':     [0.12017, 1.07621], 'l':       [-2.0, 5.0]},
}

# l spans [-8, 2], matching BOUNDS'/BOUNDS_1LAYER's range for the profile
# models rather than a lysimeter-specific one -- one consistent
# pore-connectivity-exponent range across every model this script can run.
LY1_BOUNDS = {
    1: {'theta_r': [0.001, 0.1],  'theta_s': [0.1, 0.40],
        'alpha':   [0.005, 0.5],  'n':       [1, 4],
        'k_s':     [4000, 80000], 'l':       [-2.0, 5.0]},
}

# Model_Ly2 runs the same single-material soil setup as Model_Ly1 (same
# SELECTOR.IN initials), so it starts from LY1_BOUNDS -- but alpha is
# opened down to 1e-5, four orders of magnitude below Ly1's 0.005 lower
# bound, to let the sweep reach the much flatter retention curves Ly2's
# measured outflow may need. Since alpha is in LOG_PARAMS, the sample is
# uniform in log10 space, so this widening genuinely spends realizations
# across those low decades rather than crowding them all near the upper
# end (which is what a linear draw over [1e-5, 0.5] would do).
LY2_BOUNDS = {
    1: {'theta_r': [0.001, 0.1],  'theta_s': [0.1, 0.40],
        'alpha':   [0.000001, 0.5],  'n':       [1, 4],
        'k_s':     [10, 80000], 'l':       [-2.0, 5.0]},
}


# profile2_1layer(_swi) collapses BOUNDS' 5 per-layer ranges into a single
# material -- see create_pest_setup.py's BOUNDS_1LAYER for the derivation
# (union of the 5-layer ranges above). Own copy here.
BOUNDS_1LAYER = {
    1: {'theta_r': [0.01,   0.05],    'theta_s': [0.30, 0.4992],
        'alpha':   [0.00624, 0.03],   'n':       [1.10, 5.24904],
        'k_s':     [0.00594, 0.27621], 'l':      [-2.0, 5.0]},
}

BOUNDS_OVERRIDES = {
    'model_ly1': LY1_BOUNDS,
    'model_ly2': LY2_BOUNDS,
    'profile2_1layer': BOUNDS_1LAYER,
    'profile2_1layer_swi': BOUNDS_1LAYER,
}

LOG_PARAMS = {'alpha', 'k_s'}

# Observed-data files select_model() purges from MODEL_DIR when they don't
# belong to the newly-selected model. The COMPARE pattern is deliberately
# Ly-generic (not COMPARE_Ly1_...): switching between model_ly1 and
# model_ly2 in the same RESULTS_DIR would otherwise leave the previous
# lysimeter's COMPARE file sitting in tmp_copy/ next to the new one, and
# obs_config._find_compare_file() would then pick whichever glob returned
# first -- i.e. silently calibrate against the wrong lysimeter's outflow.
OBS_INPUT_PATTERNS = [
    'SWC_observed.txt', 'SWI_observed.txt', 'SP_observed.txt',
    'ST_observed.txt', 'COMPARE_Ly*_Qmeas*.txt',
]

# Restricts which observation groups feed the objective function -- own
# copy of create_pest_setup.py's OBS_GROUP_FILTER.
OBS_GROUP_FILTER = {
    'profile2_1layer_swi': {'swi'},
}

# ── SWEEP CONFIG ──────────────────────────────────────────────────────────
NUM_REALS = 500   # number of parameter sets to run through HYDRUS
LHS_SEED  = 42    # fixed seed so re-running reproduces the same ensemble
APPEND_BATCH_SIZE = 20000   # appends run in batches of this size, each saved to sweep_in/out.csv when it finishes (None = one single batch)
APPEND_REALS = None   # set to e.g. 1000 to ADD that many realizations to the existing sweep (None = fresh run of NUM_REALS); --append N on the command line overrides this
# The archived Ly2 sweep (results/ly2_1M) was built this way: a first run, then appends in batches of
# APPEND_BATCH_SIZE (the last call had APPEND_REALS = 879500) up to 1,000,000 realizations.

# Which sampler draws the prior ensemble -- 'lhs' (Latin Hypercube,
# stratified/space-filling) or 'mc' (plain Monte Carlo, independent
# uniform random draws per parameter). See draw_lhs_ensemble() vs.
# draw_mc_ensemble() below.
SAMPLING_METHOD = 'mc'

# GLUE (Generalized Likelihood Uncertainty Estimation, see run_glue_analysis()
# at the end of this script) classifies a realization as "behavioral" when
# its R^2 between measured and modelled SWI exceeds this. Arbitrary/
# conventional choice -- tune to taste, there's no universally correct value.
GLUE_THRESHOLD = 0.8

# Local PANTHER workers -- each is a full copy of RESULTS_DIR running its own
# HYDRUS instance concurrently. Tune to your machine's core count (HYDRUS
# itself is single-threaded per run, so this is the parallelism); leave
# some headroom (don't set this to os.cpu_count() exactly) since the
# PANTHER master and OS also need cycles.
NUM_WORKERS = 20

# Every model-dependent path lives in set_active_model() below, which is
# called once at import time for ACTIVE_MODEL and again from main() when
# --model picks a different one. They're module-level globals (rather than
# arguments threaded through every function) so the rest of the script
# stays exactly as it reads for a single hardcoded ACTIVE_MODEL.
RESULTS_DIR = MODEL_DIR = SELECTOR_IN = TPL_FILE = INS_FILE = PST_FILE = None


def set_active_model(model_key):
    """Point ACTIVE_MODEL and every path derived from it at `model_key`,
    creating its results/working directories. Each model key gets its own
    RESULTS_DIR, so e.g. model_ly1 and model_ly2 sweeps coexist (separate
    tmp_copy/, sweep.pst, sweep_in.csv/sweep_out.csv, plots) instead of
    overwriting each other -- and re-running one never invalidates the
    other's already-invested HYDRUS runs."""
    global ACTIVE_MODEL, RESULTS_DIR, MODEL_DIR, SELECTOR_IN, TPL_FILE, INS_FILE, PST_FILE

    if model_key not in MODEL_CHOICES:
        raise ValueError(f"Unknown model '{model_key}'. Choices: {list(MODEL_CHOICES)}")
    ACTIVE_MODEL = model_key

    # This script's own results dir -- deliberately named *_sweep_parallel, not
    # *_sweep, so it never collides with create_sweep_setup.py's serial run of
    # the same ACTIVE_MODEL (both could otherwise be run against the same model
    # without stepping on each other's files).
    RESULTS_DIR = os.path.join(PEST_DIR, 'results', f'{ACTIVE_MODEL}_sweep_parallel')
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # This script's own private HYDRUS working directory, INSIDE RESULTS_DIR
    # (unlike create_sweep_setup.py's MODEL_DIR, which is the one global
    # BASE_DIR/tmp_copy shared by every script) -- so that when
    # pyemu.os_utils.start_workers() copies RESULTS_DIR into each worker/master
    # directory, every worker gets its own private copy of it, and this script
    # never touches (or collides with) the global tmp_copy/ the other two
    # scripts use.
    MODEL_DIR = os.path.join(RESULTS_DIR, 'tmp_copy')
    os.makedirs(MODEL_DIR, exist_ok=True)

    SELECTOR_IN = os.path.join(MODEL_DIR, 'SELECTOR.IN')

    TPL_FILE = os.path.join(RESULTS_DIR, 'selector.tpl')
    INS_FILE = os.path.join(RESULTS_DIR, 'sim_obs.ins')
    PST_FILE = os.path.join(RESULTS_DIR, 'sweep.pst')


set_active_model(ACTIVE_MODEL)

SWEEP_IN_FILE  = 'sweep_in.csv'    # relative to RESULTS_DIR, like sim_obs.dat
SWEEP_OUT_FILE = 'sweep_out.csv'   # pestpp-swp's default sweep output name
FORWARD_RUN_FILE = 'forward_run_parallel.py'  # relative to RESULTS_DIR


# ── STEP 0: SELECT MODEL ─────────────────────────────────────────────────────
def select_model(model_key):
    """Copy the chosen model's input files into THIS SCRIPT'S OWN MODEL_DIR
    (RESULTS_DIR/tmp_copy/, not the global tmp_copy/), purge any stale
    observed-data file a previous selection left behind, and write/clear
    obs_config's group-filter marker. Own copy of create_sweep_setup.py's
    select_model(), minus its LEVEL_01.dir syncing -- that's handled per-
    worker at runtime instead, by the generated forward_run_parallel.py
    (see write_forward_run_parallel())."""
    if model_key not in MODEL_CHOICES:
        raise ValueError(f"Unknown model '{model_key}'. Choices: {list(MODEL_CHOICES)}")
    src_dir = os.path.join(MODELS_DIR, MODEL_CHOICES[model_key])
    if not os.path.isdir(src_dir):
        raise FileNotFoundError(src_dir)

    src_fnames = {f for f in os.listdir(src_dir) if not f.startswith('.')}
    for pattern in OBS_INPUT_PATTERNS:
        for stale in glob.glob(os.path.join(MODEL_DIR, pattern)):
            if os.path.basename(stale) not in src_fnames:
                os.remove(stale)

    for fname in src_fnames:
        src = os.path.join(src_dir, fname)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(MODEL_DIR, fname))

    marker_path = os.path.join(MODEL_DIR, obs_config.OBS_GROUP_FILTER_MARKER)
    allowed = OBS_GROUP_FILTER.get(model_key)
    if allowed is not None:
        with open(marker_path, 'w') as f:
            f.write(','.join(sorted(allowed)))
    elif os.path.exists(marker_path):
        os.remove(marker_path)

    print(f"Selected model: {MODEL_CHOICES[model_key]} -> copied into {MODEL_DIR}")


def read_model_params(selector_path=None):
    """Parse NMat and the Block J soil-hydraulic-parameter table from the
    currently active SELECTOR.IN. Own copy of create_sweep_setup.py's
    read_model_params()."""
    selector_path = selector_path or SELECTOR_IN
    with open(selector_path) as f:
        lines = f.readlines()

    nmat_hdr = next(i for i, l in enumerate(lines) if l.strip().startswith('NMat'))
    n_mat = int(lines[nmat_hdr + 1].split()[0])

    param_hdr = next(i for i, l in enumerate(lines)
                      if l.strip().startswith('thr') and 'Alfa' in l)
    initials = {}
    for layer in range(1, n_mat + 1):
        line = lines[param_hdr + layer]
        tokens = line.split()
        if len(tokens) != 6:
            raise ValueError(
                f"{selector_path}: material {layer} parameter line has "
                f"{len(tokens)} values (expected 6): {line!r}."
            )
        thr, ths, alfa, n, ks, l = (float(t) for t in tokens)
        initials[layer] = {
            'theta_r': thr, 'theta_s': ths, 'alpha': alfa, 'n': n, 'k_s': ks,
            # Use SELECTOR.IN's own l rather than forcing Mualem's default
            # (1.0, create_pest_setup.py's convention when l is just a
            # generic placeholder): Model_Ly1/Model_Ly2's l=0.5 is now a
            # literature-fitted value (see the models' SELECTOR.IN edit
            # history), so it's a meaningful starting point, not a
            # placeholder to override.
            'l': l,
        }
    return n_mat, initials


def validated_bounds(n_mat, initials, model_key):
    """Return a copy of the bounds table for model_key, widened (with a
    printed warning) wherever the selected model's initial value falls
    outside it. Own copy of create_sweep_setup.py's validated_bounds()."""
    base = BOUNDS_OVERRIDES.get(model_key, BOUNDS)
    bounds = copy.deepcopy(base)
    for layer in range(1, n_mat + 1):
        if layer not in bounds:
            raise KeyError(
                f"Bounds table for '{model_key}' has no entry for material "
                f"{layer}, but the model has {n_mat} materials."
            )
        for full in PARAM_FULL:
            lo, hi = bounds[layer][full]
            val = initials[layer][full]
            if not (lo <= val <= hi):
                new_lo, new_hi = min(lo, val * 0.8), max(hi, val * 1.2)
                print(f"  WARNING: layer{layer} {full} initial={val:.5g} "
                      f"outside bounds [{lo:.5g},{hi:.5g}] -> widened to "
                      f"[{new_lo:.5g},{new_hi:.5g}]")
                bounds[layer][full] = [new_lo, new_hi]
    return bounds


# ── STEP 1: WRITE SELECTOR.TPL ──────────────────────────────────────────────
def write_template(n_mat):
    """Create PEST++ template file for SELECTOR.IN. Own copy of
    create_sweep_setup.py's write_template()."""
    with open(SELECTOR_IN, 'r') as f:
        lines = f.readlines()

    tpl_lines = ['ptf ~\n'] + lines[:26]  # header + unchanged lines 1-26
    for layer in range(1, n_mat + 1):
        parts = []
        for short in PARAM_SHORT:
            pname = f"{short}{layer}"
            parts.append(f"~{pname:>10s}~")
        tpl_lines.append(' '.join(parts) + '\n')
    tpl_lines += lines[26 + n_mat:]  # rest of SELECTOR.IN unchanged (Block C onward)

    with open(TPL_FILE, 'w') as f:
        f.writelines(tpl_lines)
    print(f"Wrote {TPL_FILE}")


# ── STEP 2: LOAD OBSERVATIONS ────────────────────────────────────────────────
def load_observations():
    """Load the active model's observed data via obs_config.py (shared,
    generic -- not tied to either script's ACTIVE_MODEL state)."""
    obs_list = obs_config.build_observations(MODEL_DIR)
    obs_names  = [o['name']  for o in obs_list]
    obs_vals   = [o['value'] for o in obs_list]
    obs_groups = [o['group'] for o in obs_list]

    from collections import Counter
    counts = Counter(obs_groups)
    print(f"  observation counts by group: {dict(counts)}")

    return obs_names, obs_vals, obs_groups


# ── STEP 3: WRITE SIM_OBS.INS ────────────────────────────────────────────────
def write_instruction(obs_names):
    """Write PEST++ instruction file that reads sim_obs.dat line by line."""
    with open(INS_FILE, 'w') as f:
        f.write('pif #\n')
        for name in obs_names:
            f.write(f'l1 [{name}]1:15\n')
    print(f"Wrote {INS_FILE}")


# ── STEP 4: DRAW LHS PARAMETER ENSEMBLE ──────────────────────────────────────
def draw_lhs_ensemble(bounds, n_mat, num_reals=NUM_REALS, seed=LHS_SEED):
    """
    Latin Hypercube sample across every parameter's [lo, hi] bounds
    (log-space for LOG_PARAMS), returning a DataFrame of shape
    (num_reals, n_mat * len(PARAM_SHORT)) with columns named like the .tpl
    parameters (thr1, ths1, ..., ks2, l2, ...).
    """
    par_names = [f"{s}{i}" for i in range(1, n_mat + 1) for s in PARAM_SHORT]
    d = len(par_names)

    sampler = qmc.LatinHypercube(d=d, seed=seed)
    unit = sampler.random(n=num_reals)  # (num_reals, d), each column ~ U(0,1), space-filling jointly

    data = {}
    col = 0
    for i in range(1, n_mat + 1):
        for short, full in zip(PARAM_SHORT, PARAM_FULL):
            name = f"{short}{i}"
            lo, hi = bounds[i][full]
            u = unit[:, col]
            if full in LOG_PARAMS:
                lo_l, hi_l = np.log10(lo), np.log10(hi)
                data[name] = 10 ** (lo_l + u * (hi_l - lo_l))
            else:
                data[name] = lo + u * (hi - lo)
            col += 1

    df = pd.DataFrame(data, columns=par_names)
    df.index.name = 'real_name'
    return df


def draw_mc_ensemble(bounds, n_mat, num_reals=NUM_REALS, seed=LHS_SEED):
    """
    Plain Monte Carlo: independent uniform random draws across every
    parameter's [lo, hi] bounds (log-space for LOG_PARAMS, same as
    draw_lhs_ensemble()), returning a DataFrame of the same shape/columns.
    Unlike LHS, draws for different parameters aren't jointly stratified,
    so for the same num_reals the joint parameter space is covered less
    evenly (more clumping/gaps) -- this is the classic/simpler sampler
    LHS is usually chosen over for exactly that reason.
    """
    par_names = [f"{s}{i}" for i in range(1, n_mat + 1) for s in PARAM_SHORT]
    rng = np.random.default_rng(seed)

    data = {}
    for i in range(1, n_mat + 1):
        for short, full in zip(PARAM_SHORT, PARAM_FULL):
            name = f"{short}{i}"
            lo, hi = bounds[i][full]
            u = rng.random(num_reals)
            if full in LOG_PARAMS:
                lo_l, hi_l = np.log10(lo), np.log10(hi)
                data[name] = 10 ** (lo_l + u * (hi_l - lo_l))
            else:
                data[name] = lo + u * (hi - lo)

    df = pd.DataFrame(data, columns=par_names)
    df.index.name = 'real_name'
    return df


def _plot_ensemble_histograms(ensembles, plot_cols, filename, title, bins=25):
    """Grid of one histogram per column in plot_cols, overlaying every
    (facecolor, dataframe) pair in `ensembles` -- own lightweight PNG
    replacement for pyemu.plot_utils.ensemble_helper(), which hardcodes
    multi-page PDF output via matplotlib's PdfPages (which writes PDF
    bytes regardless of the filename's extension -- passing it a .png
    name doesn't produce a real PNG) and doesn't return its figures
    either, so there's no way to redirect its output format. The
    parameter counts here (6-30) comfortably fit in one plain grid,
    unlike ensemble_helper()'s pagination (built for much larger PEST
    problems), so a single PNG is enough."""
    ncols = min(6, len(plot_cols))
    nrows = int(np.ceil(len(plot_cols) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 2.6 * nrows), squeeze=False)
    axes = axes.ravel()

    for ax, col in zip(axes, plot_cols):
        series = {fc: df[col].dropna().to_numpy() for fc, df in ensembles.items() if col in df.columns}
        series = {fc: v for fc, v in series.items() if len(v)}
        if not series:
            continue
        # Shared bin edges + per-ensemble fractions (not counts), so a small
        # behavioral subset is visible against a 1e6-run prior instead of
        # flattened to the axis. Log-spaced bins for parameters spanning
        # orders of magnitude (alpha, Ks), which linear bins squash into one.
        all_vals = np.concatenate(list(series.values()))
        lo, hi = all_vals.min(), all_vals.max()
        log_x = lo > 0 and hi / lo > 100
        edges = np.logspace(np.log10(lo), np.log10(hi), bins + 1) if log_x else np.linspace(lo, hi, bins + 1)
        for facecolor, vals in series.items():
            ax.hist(vals, bins=edges, weights=np.full(len(vals), 1.0 / len(vals)),
                    color=facecolor, alpha=0.6 if len(ensembles) > 1 else 1.0)
        if log_x:
            ax.set_xscale('log')
        ax.set_title(col, fontsize=9)
        ax.set_yticks([])
    for ax in axes[len(plot_cols):]:
        ax.set_axis_off()

    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(filename, format='png', dpi=150)
    plt.close(fig)


def plot_prior_ensemble(df):
    """Histogram each drawn parameter (one panel per parameter) via
    _plot_ensemble_histograms() -- a sanity check on the sampler's
    coverage before spending HYDRUS runs on it."""
    fig_out = os.path.join(RESULTS_DIR, 'param_prior_sampled.png')
    _plot_ensemble_histograms(
        ensembles={'0.5': df},  # gray=prior
        plot_cols=list(df.columns),
        filename=fig_out,
        title=f'VGN parameters: {SAMPLING_METHOD.upper()} prior sample (n={len(df)})',
    )
    print(f"Plot saved -> {fig_out}")


def write_sweep_in(pst, df):
    """Write the LHS ensemble as sweep_in.csv via pyemu.ParameterEnsemble,
    the format pestpp-swp expects (matches par_names order/columns)."""
    pe = pyemu.ParameterEnsemble(pst=pst, df=df[pst.par_names])
    path = os.path.join(RESULTS_DIR, SWEEP_IN_FILE)
    pe.to_csv(path)
    print(f"Wrote {path}  ({len(df)} realizations)")


# ── STEP 5: BUILD SWEEP.PST WITH pyEMU ───────────────────────────────────────
def write_placeholder_sim_obs(obs_names):
    """Write a dummy RESULTS_DIR/sim_obs.dat before Pst.from_io_files()
    reads it.

    Not cosmetic -- without it the whole script hangs forever on a first
    run for a model. from_io_files() tries to pre-populate obsval by
    parsing the model output file named in sim_obs.ins; when that file
    doesn't exist yet (which is always the case before the first HYDRUS
    run of a fresh RESULTS_DIR), pyemu falls back to shelling out to
    PEST's `inschek`, and inschek blocks on an interactive prompt with
    stdin still attached -- no timeout, no error, just a stalled process.
    Runs for an already-swept model only got away with it because a real
    sim_obs.dat was left behind by the previous run.

    The values written here are irrelevant and never reach sweep.pst:
    build_sweep_pst() overwrites every obsval from obs_vals immediately
    after. Format matches forward_run_parallel.py's interpolate_and_write()
    (one 15-wide float per line, in obs order) so the .ins file's
    'l1 [name]1:15' fields line up."""
    path = os.path.join(RESULTS_DIR, 'sim_obs.dat')
    with open(path, 'w') as f:
        f.writelines(f"{0.0:15.6f}\n" for _ in obs_names)


def build_sweep_pst(obs_names, obs_vals, obs_groups, initials, bounds, n_mat):
    """Build sweep.pst: io files + parameter/observation wiring, with
    sweep-specific control data/pestpp_options."""
    write_placeholder_sim_obs(obs_names)
    orig_dir = os.getcwd()
    os.chdir(RESULTS_DIR)
    try:
        pst = pyemu.Pst.from_io_files(
            tpl_files=['selector.tpl'],
            in_files=[os.path.relpath(SELECTOR_IN, RESULTS_DIR)],
            ins_files=['sim_obs.ins'],
            out_files=['sim_obs.dat'],
            pst_path='.',
        )
    finally:
        os.chdir(orig_dir)

    # ── Parameter data (bounds set parlbnd/parubnd -- the actual sampled
    # values live in sweep_in.csv and override parval1 per realization) ──
    par = pst.parameter_data
    for i in range(1, n_mat + 1):
        for short, full in zip(PARAM_SHORT, PARAM_FULL):
            name = f"{short}{i}"
            lo, hi = bounds[i][full]
            init   = initials[i][full]
            par.loc[par.parnme == name, 'parval1'] = init
            par.loc[par.parnme == name, 'parlbnd'] = lo
            par.loc[par.parnme == name, 'parubnd'] = hi
            par.loc[par.parnme == name, 'pargp']   = f"layer{i}"
            par.loc[par.parnme == name, 'partrans'] = 'log' if full in LOG_PARAMS else 'none'

    # ── Observation data ──────────────────────────────────────────────────────
    obs = pst.observation_data
    for name, val, group in zip(obs_names, obs_vals, obs_groups):
        obs.loc[obs.obsnme == name, 'obsval'] = val
        obs.loc[obs.obsnme == name, 'weight'] = 1.0
        obs.loc[obs.obsnme == name, 'obgnme'] = group

    # ── Control data ──────────────────────────────────────────────────────────
    # noptmax is ignored by pestpp-swp (no iterations happen), left at 0
    # since pyemu still requires a valid value.
    pst.control_data.noptmax = 0

    # ── Model command ─────────────────────────────────────────────────────────
    # Bare relative filename, NOT an absolute path (unlike
    # create_sweep_setup.py's model_command): PANTHER runs this command
    # with cwd set to whichever worker/master directory it copied
    # RESULTS_DIR into, so "python forward_run_parallel.py" picks up that
    # worker's own private copy of the script (written by
    # write_forward_run_parallel()) rather than a single shared file.
    pst.model_command = [f"python {FORWARD_RUN_FILE}"]

    # ── PEST++ / SWEEP options ────────────────────────────────────────────────
    pst.pestpp_options['sweep_parameter_csv_file'] = SWEEP_IN_FILE
    pst.pestpp_options['sweep_output_csv_file']    = SWEEP_OUT_FILE

    pst.write(PST_FILE)
    print(f"Wrote {PST_FILE}  ({len(pst.par_names)} params, {len(obs_names)} obs)")
    return pst


# ── STEP 5b: PREPARE PER-WORKER FILES (forward_run_parallel.py + obs_config.py) ──
# Template for the PANTHER-aware forward runner -- own copy of
# forward_run.py, adapted so every worker/master directory PANTHER copies
# it into resolves its HYDRUS working directory from os.getcwd() (i.e.
# wherever it actually is) instead of a fixed BASE_DIR. $hydrus_bin_base_dir
# is the one thing NOT worker-relative -- it's substituted in at generation
# time by write_forward_run_parallel() below, always pointing back at the
# single original project checkout's bin/ dir (read-only, safe to share).
_FORWARD_RUN_PARALLEL_TEMPLATE = string.Template('''#!/usr/bin/env python3
"""
PANTHER-worker forward model runner, generated by
create_sweep_setup_parallel.py's write_forward_run_parallel() -- own copy
of forward_run.py, adapted so this resolves its HYDRUS working directory
relative to os.getcwd() (whichever worker/master directory
pyemu.os_utils.start_workers() copied this file into), instead of a fixed
BASE_DIR shared by every run. That's what makes concurrent PANTHER workers
safe here: each gets its own private tmp_copy/ and LEVEL_01.dir instead of
all colliding on one shared directory.

DO NOT EDIT BY HAND -- regenerated every time
create_sweep_setup_parallel.py runs (see write_forward_run_parallel()).
"""

import io
import os
import platform
import shutil
import subprocess
import sys
import numpy as np
import pandas as pd

import obs_config

# The one thing that stays fixed across all workers: where the original
# project's (read-only, shared) HYDRUS binaries live.
HYDRUS_BIN_BASE_DIR = $hydrus_bin_base_dir

WORKER_DIR  = os.getcwd()                          # this worker's/master's own private copy
MODEL_DIR   = os.path.join(WORKER_DIR, 'tmp_copy')  # this worker's own HYDRUS working dir
OBS_NODE    = os.path.join(MODEL_DIR, 'Obs_Node.out')
LEVEL01_PATH = os.path.join(WORKER_DIR, 'LEVEL_01.dir')
PEST_FILLED_SELECTOR_IN = 'SELECTOR.IN'   # PEST++ fills selector.tpl -> ./SELECTOR.IN in its own cwd
TARGET_SELECTOR_IN = os.path.join(MODEL_DIR, 'SELECTOR.IN')  # what HYDRUS reads
SIM_OBS_OUT = 'sim_obs.dat'


def sync_selector_in():
    """Copy PEST++'s freshly-filled ./SELECTOR.IN into this worker's own
    tmp_copy/ -- the only place this worker's HYDRUS instance reads it
    from."""
    shutil.copy2(PEST_FILLED_SELECTOR_IN, TARGET_SELECTOR_IN)


def sync_level01_dir():
    """Point this worker's own LEVEL_01.dir at this worker's own tmp_copy.
    Written fresh on every model call (each worker is an independent copy
    of the same RESULTS_DIR template, so this can't be precomputed once at
    generation time -- every worker needs its own absolute path, known
    only once PANTHER has actually copied this directory somewhere)."""
    with open(LEVEL01_PATH, 'w') as f:
        f.write(f"{MODEL_DIR}\\n\\n")


def run_hydrus(timeout=30):
    system = platform.system()
    if system == 'Windows':
        exe = os.path.join(HYDRUS_BIN_BASE_DIR, 'bin', 'windows', 'H1D_CALC.exe')
    elif system == 'Darwin':
        exe = os.path.join(HYDRUS_BIN_BASE_DIR, 'bin', 'mac', 'hydrus')
    else:
        raise RuntimeError(f"Unsupported OS: {system}")

    # Remove any previous Obs_Node.out first: its presence afterward is the
    # only reliable signal that HYDRUS actually completed and wrote fresh
    # output this call (see forward_run.py's run_hydrus() for the incident
    # this guards against).
    if os.path.exists(OBS_NODE):
        os.remove(OBS_NODE)

    # stdin=DEVNULL: on a non-converging trial parameter set, HYDRUS writes
    # tmp_copy/Error.msg and then blocks on a read/prompt instead of exiting
    # cleanly, which would hang this worker (and the run manager waiting on
    # it) forever without this.
    try:
        result = subprocess.run(
            exe, cwd=WORKER_DIR, capture_output=True, stdin=subprocess.DEVNULL,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        print(f"HYDRUS exceeded {timeout}s -- killed, failing this model run.")
        return False

    if result.returncode != 0:
        print("HYDRUS stderr:", result.stderr.decode(errors='ignore'))

    return os.path.exists(OBS_NODE)


def read_obs_node():
    """Read Obs_Node.out and return time array + {(group, node): sim_series}.

    Header is located by CONTENT (the "time  h  theta ..." column-label
    line), not by a fixed row count: a fixed header=7 silently breaks
    under pandas' python engine, which skips blank lines before applying
    a numeric header index -- Obs_Node.out has blank lines that shift a
    literal row-7 count onto the row for t=2 instead of the real
    column-label row, discarding it AND t=1 as "header", so every run's
    first two simulated days went missing (compared as -9999)."""
    with open(OBS_NODE) as f:
        lines = f.readlines()
    hdr_idx = next(i for i, l in enumerate(lines) if l.split()[:1] == ['time'])
    df = pd.read_csv(
        io.StringIO(''.join(lines[hdr_idx + 1:])),
        header=None,
        sep=r'\\s+',
        engine='python',
        on_bad_lines='skip',
    )
    df = df.apply(pd.to_numeric, errors='coerce').dropna(subset=[df.columns[0]])
    time = df.iloc[:, 0].values

    is_swi = obs_config.is_swi_model(MODEL_DIR)
    cols_map = obs_config.SWI_OBS_NODE_COLS if is_swi else obs_config.OBS_NODE_COLS

    sim_series = {}
    for group, cols in cols_map.items():
        for node, col in cols.items():
            vals = df.iloc[:, col].values
            if group == 'sp':
                vals = -vals
            elif group == 'q':
                # HYDRUS reports Flux in cm/day (a flux density); Qmeas in
                # COMPARE_Ly*_Qmeas*.txt is an absolute lysimeter outflow
                # rate in L/day -- scale by the lysimeter's footprint (see
                # obs_config.SWI_LYSIMETER_AREA_CM2), then flip sign to
                # match Qmeas's positive-outflow convention.
                vals = -vals * obs_config.SWI_LYSIMETER_AREA_CM2 / 1000.0
            sim_series[(group, node)] = vals
    return time, sim_series


def interpolate_and_write(sim_time, sim_series, obs_list):
    """Interpolate each obs's matching simulated series to its time and
    write sim_obs.dat in the same order as obs_list."""
    lines = []
    for o in obs_list:
        series = sim_series[(o['group'], o['node'])]
        v = np.interp(o['time'], sim_time, series, left=np.nan, right=np.nan)
        val = v if np.isfinite(v) else -9999.0
        lines.append(f"{val:15.6f}\\n")

    with open(SIM_OBS_OUT, 'w') as f:
        f.writelines(lines)


if __name__ == '__main__':
    sync_selector_in()
    sync_level01_dir()
    ok = run_hydrus()
    if not ok:
        print("HYDRUS did not produce Obs_Node.out this run -- failing this model run.")
        sys.exit(1)
    sim_time, sim_series = read_obs_node()
    obs_list = obs_config.build_observations(MODEL_DIR)
    interpolate_and_write(sim_time, sim_series, obs_list)
''')


def write_forward_run_parallel():
    """Generate RESULTS_DIR/forward_run_parallel.py from
    _FORWARD_RUN_PARALLEL_TEMPLATE (with this machine's BASE_DIR baked in
    for locating the shared HYDRUS binaries), and copy obs_config.py
    alongside it -- both needed in every worker/master directory since
    pyemu.os_utils.start_workers() copies RESULTS_DIR wholesale into each
    one and runs the model command from within that copy."""
    content = _FORWARD_RUN_PARALLEL_TEMPLATE.substitute(hydrus_bin_base_dir=repr(BASE_DIR))
    fr_path = os.path.join(RESULTS_DIR, FORWARD_RUN_FILE)
    with open(fr_path, 'w') as f:
        f.write(content)
    print(f"Wrote {fr_path}")

    obs_config_dst = os.path.join(RESULTS_DIR, 'obs_config.py')
    shutil.copy2(os.path.join(SCRIPT_DIR, 'obs_config.py'), obs_config_dst)
    print(f"Copied obs_config.py -> {obs_config_dst}")


# ── STEP 6: RUN PESTPP-SWP ACROSS NUM_WORKERS LOCAL PANTHER WORKERS ─────────
def run_sweep_parallel(num_workers=NUM_WORKERS):
    """Run the sweep across `num_workers` local PANTHER workers instead of
    a single serial process. pyemu.os_utils.start_workers() copies
    RESULTS_DIR (this script's own tmp_copy/, forward_run_parallel.py,
    obs_config.py, sweep.pst, etc.) into one isolated directory per worker
    plus one master directory -- see write_forward_run_parallel()'s
    docstring for why that isolation, not just the copying itself, is what
    makes this safe to run concurrently.

    pestpp-swp's master instance writes sweep_out.csv into the master
    directory; it is copied back to RESULTS_DIR/BATCH_OUT_FILE (the raw
    output of JUST the batch that was run -- the caller scores and archives
    it, see append_and_run_sweep()/store_fresh_sweep()).
    """
    # Dedicated subfolder for worker_0../worker_{N-1} dirs -- NOT the shared
    # results/ folder itself, so these can never collide with another
    # model's own results/<model>[_sweep[_parallel]] directory living
    # alongside this one.
    workers_root = os.path.join(PEST_DIR, 'results', f'{ACTIVE_MODEL}_sweep_parallel_workers')
    os.makedirs(workers_root, exist_ok=True)
    master_dir = os.path.join(PEST_DIR, 'results', f'{ACTIVE_MODEL}_sweep_parallel_master')

    # start_workers() copies worker_dir into EVERY worker and the master.
    # RESULTS_DIR also holds the archived batches/, scores and .bak copies,
    # which grow to GBs on appended sweeps -- so copy it to a lean staging dir
    # without those (workers only need the model files and this batch's
    # sweep_in.csv; all results are collected in the master dir).
    stage_dir = os.path.join(PEST_DIR, 'results', f'{ACTIVE_MODEL}_sweep_parallel_stage')
    shutil.rmtree(stage_dir, ignore_errors=True)
    shutil.copytree(RESULTS_DIR, stage_dir,
                    ignore=shutil.ignore_patterns('*.bak', '*.png', BATCH_DIR_NAME, SCORES_FILE,
                                                  SWEEP_OUT_FILE, BATCH_OUT_FILE))

    try:
        pyemu.os_utils.start_workers(
            worker_dir=stage_dir,
            exe_rel_path='pestpp-swp',
            pst_rel_path='sweep.pst',
            num_workers=num_workers,
            worker_root=workers_root,
            master_dir=master_dir,
            local=True,
            verbose=True,
        )
    finally:
        shutil.rmtree(stage_dir, ignore_errors=True)

    src = os.path.join(master_dir, SWEEP_OUT_FILE)
    if not os.path.exists(src):
        raise FileNotFoundError(f"pestpp-swp produced no {SWEEP_OUT_FILE} in {master_dir}")
    shutil.copy2(src, os.path.join(RESULTS_DIR, BATCH_OUT_FILE))
    print(f"Copied {SWEEP_OUT_FILE} from {master_dir} -> {RESULTS_DIR}/{BATCH_OUT_FILE}")


# ── STEP 6b: STORAGE LAYOUT, SCORING, AND APPENDING BATCHES ─────────────────
# Raw pestpp-swp output is huge (~8 KB/run: one column per observation), so it
# is never kept as ONE cumulative file (that needs ~10 GB of RAM at 1M runs and
# was copied into every worker). Layout inside RESULTS_DIR:
#   sweep_in.csv                       cumulative parameter sets (small)
#   batches/batch_NNNN_out.csv[.gz]    raw output, one file per batch (archive;
#                                      only needed to re-score -- can be deleted
#                                      if disk space runs out)
#   sweep_scores.csv                   slim cumulative table, row-aligned with
#                                      sweep_in.csv: run meta (failed_flag, phi,
#                                      ...), n_failed_obs, r2, kge -- this is
#                                      what GLUE and the plots read
BATCH_DIR_NAME   = 'batches'
SCORES_FILE      = 'sweep_scores.csv'
BATCH_OUT_FILE   = 'sweep_out_batch.csv'   # temp: master's output for the batch just run
SCORE_CHUNK_ROWS = 5000                    # rows scored at a time (memory)

FAILED_SIM_VALUE = -9999.0   # what forward_run writes to sim_obs.dat when HYDRUS has no value for an observation
MAX_FAILED_FRAC  = 0.05      # runs with >= this fraction of -9999 observations get R2/KGE = NaN (never behavioral); below it, the -9999 points are just dropped


def _batches_dir():
    return os.path.join(RESULTS_DIR, BATCH_DIR_NAME)


def _batch_files():
    return sorted(glob.glob(os.path.join(_batches_dir(), 'batch_*_out.csv*')))


def _migrate_legacy_sweep_out():
    """Older sweeps kept one big cumulative sweep_out.csv -- move it in as batch 0."""
    legacy = os.path.join(RESULTS_DIR, SWEEP_OUT_FILE)
    if os.path.exists(legacy) and not _batch_files():
        os.makedirs(_batches_dir(), exist_ok=True)
        shutil.move(legacy, os.path.join(_batches_dir(), 'batch_0000_out.csv'))
        print(f"Moved legacy {SWEEP_OUT_FILE} -> {BATCH_DIR_NAME}/batch_0000_out.csv")


def _store_batch(src_path):
    """gzip src_path into batches/ as the next batch file, then delete src_path."""
    os.makedirs(_batches_dir(), exist_ok=True)
    dst = os.path.join(_batches_dir(), f'batch_{len(_batch_files()):04d}_out.csv.gz')
    with open(src_path, 'rb') as fin, gzip.open(dst, 'wb', compresslevel=3) as fout:
        shutil.copyfileobj(fin, fout)
    os.remove(src_path)
    return dst


def _r2_kge_matrix(sim, obs):
    """Vectorised R2 and KGE (Gupta et al., 2009) for many runs at once:
    sim is (n_runs, n_obs), obs is (n_obs,). Points where the simulation is the
    HYDRUS failure sentinel (-9999, or non-finite) are dropped pairwise before
    computing anything. A run gets NaN if >= MAX_FAILED_FRAC of its points
    were failed, fewer than 3 valid points remain, or either series is
    constant. Returns (r2, kge, n_failed_obs), each of length n_runs."""
    sim = np.asarray(sim, dtype=float)
    obs = np.asarray(obs, dtype=float)[None, :]
    valid = np.isfinite(sim) & np.isfinite(obs) & (sim != FAILED_SIM_VALUE)
    n_valid = valid.sum(axis=1)
    n_failed = sim.shape[1] - n_valid
    n = np.maximum(n_valid, 1)

    mean_s = np.where(valid, sim, 0.0).sum(axis=1) / n
    mean_o = np.where(valid, obs, 0.0).sum(axis=1) / n
    ds = np.where(valid, sim - mean_s[:, None], 0.0)
    do = np.where(valid, obs - mean_o[:, None], 0.0)
    cov, var_s, var_o = (ds * do).sum(axis=1), (ds ** 2).sum(axis=1), (do ** 2).sum(axis=1)

    # constant series (e.g. PEST++'s -1e100 filler for failed runs) must be NaN, but
    # their float variance is ~1e-30, not exactly 0 -- so test max == min instead
    const_s = np.where(valid, sim, -np.inf).max(axis=1) == np.where(valid, sim, np.inf).min(axis=1)
    const_o = np.where(valid, obs, -np.inf).max(axis=1) == np.where(valid, obs, np.inf).min(axis=1)
    ok = (n_failed < MAX_FAILED_FRAC * sim.shape[1]) & (n_valid >= 3) & ~const_s & ~const_o
    with np.errstate(divide='ignore', invalid='ignore'):
        r = cov / np.sqrt(var_s * var_o)
        alpha = np.sqrt(var_s / var_o)
        beta = mean_s / mean_o
        kge = 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)
    return np.where(ok, r ** 2, np.nan), np.where(ok, kge, np.nan), n_failed


def _r2_kge(sim, obs):
    """R2 and KGE for ONE run -- see _r2_kge_matrix()."""
    r2, kge, _ = _r2_kge_matrix(np.asarray(sim, dtype=float)[None, :], obs)
    return r2[0], kge[0]


def _score_file(path, obs_names, obs_vals):
    """Read one raw batch output file in chunks and return its slim scores
    table: every non-observation column (run_id, failed_flag, phi, ...) plus
    n_failed_obs, r2, kge. Runs PEST++ flagged as failed get NaN r2/kge."""
    obs_arr = np.array(obs_vals, dtype=float)
    obs_set = set(obs_names)
    parts = []
    for chunk in pd.read_csv(path, chunksize=SCORE_CHUNK_ROWS):
        meta_cols = [c for c in chunk.columns if c not in obs_set and c not in ('r2', 'kge')]
        r2, kge, n_failed = _r2_kge_matrix(chunk[obs_names].to_numpy(dtype=float), obs_arr)
        sc = chunk[meta_cols].copy()
        sc['n_failed_obs'] = n_failed
        sc['r2'] = r2
        sc['kge'] = kge
        if 'failed_flag' in sc.columns:
            sc.loc[sc['failed_flag'] != 0, ['r2', 'kge']] = np.nan
        parts.append(sc)
    return pd.concat(parts, ignore_index=True)


def _append_scores(scores):
    path = os.path.join(RESULTS_DIR, SCORES_FILE)
    scores.to_csv(path, mode='a', header=not os.path.exists(path), index=False)


def _check_inputs_match_scores():
    """sweep_in.csv and sweep_scores.csv must have the same number of rows. If a
    previous batch crashed while sweep_in.csv held only that batch, restore the
    cumulative one from sweep_in.csv.bak."""
    in_path = os.path.join(RESULTS_DIR, SWEEP_IN_FILE)
    n_sc = len(pd.read_csv(os.path.join(RESULTS_DIR, SCORES_FILE), usecols=[0]))
    n_in = len(pd.read_csv(in_path, usecols=[0]))
    if n_in == n_sc:
        return
    bak = in_path + '.bak'
    if os.path.exists(bak) and len(pd.read_csv(bak, usecols=[0])) == n_sc:
        shutil.copy2(bak, in_path)
        print(f"{SWEEP_IN_FILE} had {n_in} rows vs. {n_sc} scored runs (previous batch crashed?) "
              f"-- restored the cumulative parameter sets from {SWEEP_IN_FILE}.bak")
        return
    raise RuntimeError(
        f"{SWEEP_IN_FILE} has {n_in} rows but {SCORES_FILE} has {n_sc} -- can't tell which runs "
        "belong together; fix or delete one of them before appending."
    )


def append_and_run_sweep(n_new_reals, n_mat, model_bounds, pst, obs_names, obs_vals,
                          num_workers=NUM_WORKERS):
    """Draw n_new_reals ADDITIONAL realizations and run them through
    pestpp-swp, APPENDING to the existing sweep -- e.g. already have 500 runs
    done, append_and_run_sweep(200, ...) ends up with 700 cumulative runs, not
    a fresh 200 (invoke via `python SWP_setup_parallel.py --append 200` or
    APPEND_REALS).

    Draws just the new batch (seed derived from how many realizations already
    exist, so repeat calls each draw a genuinely NEW batch), runs ONLY that
    batch through PANTHER (same sweep.pst), then:
      - scores it (r2/kge) and appends the slim rows to sweep_scores.csv,
      - gzips its raw output into batches/ (never loaded together with the
        earlier batches),
      - appends its parameter sets to the cumulative sweep_in.csv, so
        sweep_in.csv and sweep_scores.csv stay row-aligned.
    sweep_in.csv is backed up to sweep_in.csv.bak first (it is overwritten with
    just this batch while pestpp-swp runs)."""
    in_path = os.path.join(RESULTS_DIR, SWEEP_IN_FILE)
    if not os.path.exists(in_path):
        raise FileNotFoundError(
            f"No existing {SWEEP_IN_FILE} in {RESULTS_DIR} to append to -- "
            "run this script without --append first to create an initial sweep."
        )
    _migrate_legacy_sweep_out()
    if not _batch_files():
        raise FileNotFoundError(f"No earlier results in {RESULTS_DIR} to append to -- run a sweep first.")
    if not os.path.exists(os.path.join(RESULTS_DIR, SCORES_FILE)):
        print(f"No {SCORES_FILE} yet -- scoring the existing batches first ...")
        summarize_sweep(obs_names, obs_vals, rescore=True)
    _check_inputs_match_scores()

    shutil.copy2(in_path, in_path + '.bak')
    prev_in = pd.read_csv(in_path, index_col=0)
    n_existing = len(prev_in)
    batch_seed = LHS_SEED + n_existing
    print(f"Existing sweep: {n_existing} realizations. Drawing {n_new_reals} more "
          f"(method={SAMPLING_METHOD}, batch seed={batch_seed}) ...")

    if SAMPLING_METHOD == 'lhs':
        batch_df = draw_lhs_ensemble(model_bounds, n_mat, num_reals=n_new_reals, seed=batch_seed)
    elif SAMPLING_METHOD == 'mc':
        batch_df = draw_mc_ensemble(model_bounds, n_mat, num_reals=n_new_reals, seed=batch_seed)
    else:
        raise ValueError(f"Unknown SAMPLING_METHOD '{SAMPLING_METHOD}' -- must be 'lhs' or 'mc'")
    plot_prior_ensemble(batch_df)  # sanity-check just this new batch's coverage (overwrites param_prior_sampled.pdf)

    write_sweep_in(pst, batch_df)  # overwrites sweep_in.csv with JUST this batch, temporarily
    print(f"=== Running pestpp-swp on this {n_new_reals}-realization batch across {num_workers} PANTHER workers ===")
    run_sweep_parallel(num_workers=num_workers)  # leaves this batch's raw output in BATCH_OUT_FILE

    batch_path = os.path.join(RESULTS_DIR, BATCH_OUT_FILE)
    batch_scores = _score_file(batch_path, obs_names, obs_vals)
    batch_in = pd.read_csv(in_path, index_col=0)
    if len(batch_scores) != len(batch_in):
        print(f"WARNING: batch has {len(batch_in)} parameter sets but {len(batch_scores)} output rows.")

    _store_batch(batch_path)
    _append_scores(batch_scores)
    combined_in = pd.concat([prev_in, batch_in], ignore_index=True)
    combined_in.index.name = 'real_name'
    combined_in.to_csv(in_path)
    print(f"Appended {len(batch_scores)} realizations -> {len(combined_in)} cumulative "
          f"({SWEEP_IN_FILE} + {SCORES_FILE}; raw output archived in {BATCH_DIR_NAME}/)")


def store_fresh_sweep(obs_names, obs_vals):
    """After run_sweep_parallel() of a brand-new (non-append) sweep: throw away any
    previous results and store this run as batch 0 (scores + archived raw output)."""
    batch_path = os.path.join(RESULTS_DIR, BATCH_OUT_FILE)
    scores = _score_file(batch_path, obs_names, obs_vals)
    shutil.rmtree(_batches_dir(), ignore_errors=True)
    for f in (SCORES_FILE, SWEEP_OUT_FILE):
        p = os.path.join(RESULTS_DIR, f)
        if os.path.exists(p):
            os.remove(p)
    _store_batch(batch_path)
    _append_scores(scores)


# ── STEP 7: SUMMARIZE SWEEP OUTPUT ───────────────────────────────────────────
def summarize_sweep(obs_names=None, obs_vals=None, rescore=True):
    """Print a quick pass/fail and phi/R2/KGE-range summary of the sweep.
    rescore=True (default) recomputes r2/kge from every raw batch file in
    batches/ with the CURRENT scoring rules (e.g. after changing
    MAX_FAILED_FRAC) and rewrites sweep_scores.csv -- needs obs_names/obs_vals.
    rescore=False just reads the existing sweep_scores.csv (used right after a
    run, which already scored its batch). Returns the scores table."""
    _migrate_legacy_sweep_out()
    scores_path = os.path.join(RESULTS_DIR, SCORES_FILE)

    if rescore or not os.path.exists(scores_path):
        files = _batch_files()
        if not files:
            print(f"No raw batch output in {_batches_dir()} (or legacy {SWEEP_OUT_FILE}) -- run pestpp-swp first.")
            return None
        if not (obs_names and obs_vals):
            print("Need obs_names/obs_vals to compute r2/kge.")
            return None
        if os.path.exists(scores_path):
            os.remove(scores_path)
        for f in files:
            print(f"  scoring {os.path.basename(f)} ...")
            _append_scores(_score_file(f, obs_names, obs_vals))

    df = pd.read_csv(scores_path)
    print(f"{SCORES_FILE}: {len(df)} rows")
    ok_mask = df['failed_flag'] == 0 if 'failed_flag' in df.columns else pd.Series(True, index=df.index)
    if 'failed_flag' in df.columns:
        print(f"  failed runs: {int((~ok_mask).sum())} / {len(df)}")
    if 'phi' in df.columns:
        print(f"  phi range (successful runs): {df.loc[ok_mask, 'phi'].min():.4g} - {df.loc[ok_mask, 'phi'].max():.4g}")

    n_obs = len(obs_names) if obs_names else None
    if n_obs:
        nf = df.loc[ok_mask, 'n_failed_obs']
        print(f"  runs with some {FAILED_SIM_VALUE:.0f} observations: {int(((nf > 0) & (nf < n_obs)).sum())}; "
              f"with all: {int((nf == n_obs).sum())}")
    print(f"  runs scored (< {MAX_FAILED_FRAC:.0%} of observations failed): "
          f"{int(df.loc[ok_mask, 'r2'].notna().sum())} / {int(ok_mask.sum())}")
    print(f"  R2  range (scored runs): {df.loc[ok_mask, 'r2'].min():.4g} - {df.loc[ok_mask, 'r2'].max():.4g}")
    print(f"  KGE range (scored runs): {df.loc[ok_mask, 'kge'].min():.4g} - {df.loc[ok_mask, 'kge'].max():.4g}")
    return df


# ── STEP 8: GLUE ANALYSIS -- PLOT PRIOR VS. BEHAVIORAL PARAMETER SAMPLES ─────
def run_glue_analysis(threshold=GLUE_THRESHOLD):
    """GLUE (Generalized Likelihood Uncertainty Estimation): classify each
    swept realization as 'behavioral' when its R2 (sweep_scores.csv, from
    summarize_sweep()) exceeds `threshold`, then overlay the full prior
    parameter ensemble (sweep_in.csv) with just the behavioral subset -- one
    histogram per parameter, gray=prior, blue=GLUE-behavioral -- via
    _plot_ensemble_histograms()."""
    sweep_in_path = os.path.join(RESULTS_DIR, SWEEP_IN_FILE)
    scores_path   = os.path.join(RESULTS_DIR, SCORES_FILE)
    if not os.path.exists(sweep_in_path) or not os.path.exists(scores_path):
        print(f"Need both {SWEEP_IN_FILE} and {SCORES_FILE} -- run the sweep / summarize_sweep() first.")
        return None

    prior = pd.read_csv(sweep_in_path, index_col=0)
    out   = pd.read_csv(scores_path)
    if 'r2' not in out.columns:
        print(f"No 'r2' column in {SCORES_FILE} -- run summarize_sweep(obs_names, obs_vals) first.")
        return None
    if len(prior) != len(out):
        print(f"WARNING: {SWEEP_IN_FILE} has {len(prior)} rows but {SCORES_FILE} has "
              f"{len(out)} -- can't align realizations positionally, skipping GLUE plot.")
        return None

    # sweep_scores.csv preserves sweep_in.csv's row order (one row per
    # realization, failed or not); r2 is NaN for failed/unscored runs so
    # they're naturally excluded from the behavioral mask.
    behavioral_mask = (out['r2'] > threshold).to_numpy()
    n_behavioral = int(behavioral_mask.sum())
    print(f"GLUE: {n_behavioral} / {len(out)} realizations behavioral (R2 > {threshold})")
    if n_behavioral == 0:
        print("No behavioral realizations -- skipping plot.")
        return prior.iloc[0:0]

    behavioral = prior.iloc[behavioral_mask]

    fig_out = os.path.join(RESULTS_DIR, 'param_prior_glue.png')
    _plot_ensemble_histograms(
        ensembles={'0.5': prior, 'b': behavioral},  # gray=prior, blue=GLUE-behavioral
        plot_cols=list(prior.columns),
        filename=fig_out,
        title=f'VGN parameters: prior (gray) vs. GLUE-behavioral R2>{threshold} (blue)',
    )
    print(f"Plot saved -> {fig_out}")
    return behavioral


# ── MAIN ─────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Build/run the parallel pestpp-swp sweep, or just re-examine existing results "
                     "without resampling or re-running anything."
    )
    parser.add_argument(
        '--model', choices=list(MODEL_CHOICES), default=ACTIVE_MODEL,
        help=f"Which model to sweep (default: {ACTIVE_MODEL}, the ACTIVE_MODEL constant). Each "
             "model gets its own results/<model>_sweep_parallel directory, so switching here "
             "never touches another model's existing sweep."
    )
    parser.add_argument(
        '--reals', type=int, metavar='N', default=NUM_REALS,
        help=f"How many realizations to draw and run (default: {NUM_REALS}, the NUM_REALS "
             "constant). Handy for a smaller test run before committing to a full sweep. "
             "Ignored when --append is given, which takes its own count."
    )
    parser.add_argument(
        '--append', type=int, metavar='N', default=APPEND_REALS,
        help="Draw N additional realizations and run them through pestpp-swp, APPENDING to the "
             "existing sweep_in.csv/sweep_out.csv in RESULTS_DIR instead of overwriting them "
             "(e.g. already have 500 runs -- '--append 200' gives 700 cumulative runs)."
    )
    parser.add_argument(
        '--glue-only', type=float, metavar='THRESHOLD', default=None,
        help="Skip straight to re-plotting the GLUE prior-vs-behavioral figure at R2 > THRESHOLD "
             "using the existing sweep_in.csv/sweep_out.csv -- no resampling or re-running "
             "pestpp-swp at all."
    )
    parser.add_argument(
        '--summarize-only', action='store_true',
        help="Recompute r2/kge for the existing sweep_out.csv (with the current -9999 filtering) "
             "and redo the GLUE plot -- no sampling, no pestpp-swp, no HYDRUS. Don't run this "
             "while a sweep for the same model is still running."
    )
    args = parser.parse_args()

    # Before anything reads RESULTS_DIR/MODEL_DIR -- including --glue-only,
    # which re-reads the selected model's own sweep_in.csv/sweep_out.csv.
    set_active_model(args.model)

    if args.summarize_only:
        print(f"=== Summarize-only: recomputing r2/kge for '{ACTIVE_MODEL}' ===")
        select_model(ACTIVE_MODEL)
        obs_names, obs_vals, obs_groups = load_observations()
        summarize_sweep(obs_names, obs_vals)
        run_glue_analysis()
        raise SystemExit

    if args.glue_only is not None:
        print(f"=== GLUE-only: re-plotting behavioral threshold R2 > {args.glue_only} ===")
        run_glue_analysis(threshold=args.glue_only)
        raise SystemExit

    print(f"=== Step 0: selecting model '{ACTIVE_MODEL}' ===")
    select_model(ACTIVE_MODEL)
    n_mat, model_initials = read_model_params()
    model_bounds = validated_bounds(n_mat, model_initials, ACTIVE_MODEL)

    print("=== Step 1: writing selector.tpl ===")
    write_template(n_mat)

    print("=== Step 2: loading observations ===")
    obs_names, obs_vals, obs_groups = load_observations()
    print(f"  {len(obs_names)} observations total")

    print("=== Step 3: writing sim_obs.ins ===")
    write_instruction(obs_names)

    print("=== Step 5: building sweep.pst ===")
    pst = build_sweep_pst(obs_names, obs_vals, obs_groups, model_initials, model_bounds, n_mat)

    print("=== Step 5b: preparing per-worker forward_run_parallel.py/obs_config.py ===")
    write_forward_run_parallel()

    if args.append is not None:
        remaining = args.append
        batch = APPEND_BATCH_SIZE or remaining
        while remaining > 0:
            n_now = min(batch, remaining)
            append_and_run_sweep(n_now, n_mat, model_bounds, pst, obs_names, obs_vals)
            remaining -= n_now
            print(f"=== {remaining} realizations still to append ===")
    else:
        print(f"=== Step 4: drawing {args.reals}-realization {SAMPLING_METHOD.upper()} ensemble ===")
        if SAMPLING_METHOD == 'lhs':
            ensemble_df = draw_lhs_ensemble(model_bounds, n_mat, num_reals=args.reals)
        elif SAMPLING_METHOD == 'mc':
            ensemble_df = draw_mc_ensemble(model_bounds, n_mat, num_reals=args.reals)
        else:
            raise ValueError(f"Unknown SAMPLING_METHOD '{SAMPLING_METHOD}' -- must be 'lhs' or 'mc'")
        plot_prior_ensemble(ensemble_df)
        write_sweep_in(pst, ensemble_df)

        print(f"=== Step 6: running pestpp-swp across {NUM_WORKERS} PANTHER workers ===")
        run_sweep_parallel()
        store_fresh_sweep(obs_names, obs_vals)

    print("=== Step 7: summarizing sweep_scores.csv ===")
    summarize_sweep(obs_names, obs_vals, rescore=False)

    print("=== Step 8: GLUE analysis -- prior vs. behavioral parameter samples ===")
    run_glue_analysis()
