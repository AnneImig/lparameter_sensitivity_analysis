#!/usr/bin/env python3
"""
Own copy of interaction_test_ly1.py, retargeted at Model_Ly2 instead of
Model_Ly1 -- same 15-pair x 5x5-grid pairwise interaction screen (375
real HYDRUS runs, other 4 parameters held at Model_Ly2's own SELECTOR.IN
defaults for each pair), so the two lysimeters' interaction matrices can
be compared directly.

Run:
    cd pest_setup
    python interaction_test_ly2.py
"""

import io
import os
import csv
import platform
import shutil
import subprocess
import itertools
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PEST_DIR = os.path.dirname(SCRIPT_DIR)           # this part's folder; results go to PEST_DIR/results
BASE_DIR = os.path.dirname(PEST_DIR)          # repository root -- HYDRUS_models/, bin/ and tmp_copy/ live here
MODEL_SRC = os.path.join(BASE_DIR, 'HYDRUS_models', 'Model_Ly2')
MODEL_DIR = os.path.join(BASE_DIR, 'tmp_copy')
LEVEL01_PATH = os.path.join(BASE_DIR, 'LEVEL_01.dir')
SELECTOR_IN = os.path.join(MODEL_DIR, 'SELECTOR.IN')
OBS_NODE = os.path.join(MODEL_DIR, 'Obs_Node.out')
HYDRUS_TIMEOUT = 30

RESULTS_DIR = os.path.join(PEST_DIR, 'results', 'interaction_test_model_ly2')
os.makedirs(RESULTS_DIR, exist_ok=True)
OUT_CSV = os.path.join(RESULTS_DIR, 'pairwise_grid.csv')

# Model_Ly2's own SELECTOR.IN defaults (thr=0.026, ths=0.29, alfa=0.005,
# n=1.23, Ks=146.34, l=0.5) -- note how different these are from Ly1's
# (thr=0.007, ths=0.275, alfa=0.35, n=1.41, Ks=6040.20, l=0.5): Ly2 sits
# at a much lower Ks and alpha, closer to a finer-textured soil.
BASELINE = {
    'thr1': 0.026, 'ths1': 0.29, 'alf1': 0.005,
    'vgn1': 1.23, 'ks1': 146.34, 'l1': 0.5,
}

# 5-level grids, inset from LY2_BOUNDS (SWP_setup_parallel.py: theta_r
# [0.001,0.1], theta_s[0.1,0.40], alpha[1e-6,0.5], n[1,4], Ks[10,80000],
# l[-8,2]) around each parameter's own default -- alf1/ks1 log-spaced
# (LOG_PARAMS elsewhere in this repo) since both span multiple decades
# and Ly2's defaults sit far from the middle of their bounds.
GRIDS = {
    'thr1': np.array([0.005, 0.030, 0.055, 0.080, 0.099]),
    'ths1': np.array([0.15, 0.21, 0.27, 0.33, 0.38]),
    'alf1': np.array([0.0005, 0.002, 0.008, 0.04, 0.2]),
    'vgn1': np.array([1.05, 1.30, 1.55, 1.80, 2.05]),
    'ks1':  np.array([20.0, 100.0, 600.0, 4000.0, 30000.0]),
    'l1':   np.array([-2.0, -0.25, 1.5, 3.25, 5.0]),  # same l range as the Monte Carlo sweeps (Sect. 3.6)
}

PARAM_KWARG = {'thr1': 'theta_r', 'ths1': 'theta_s', 'alf1': 'alpha',
               'vgn1': 'n', 'ks1': 'ks', 'l1': 'l'}


def select_model_ly2():
    os.makedirs(MODEL_DIR, exist_ok=True)
    for fname in os.listdir(MODEL_SRC):
        src = os.path.join(MODEL_SRC, fname)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(MODEL_DIR, fname))
    with open(LEVEL01_PATH, 'w') as f:
        f.write(f"{MODEL_DIR}\n\n")


def write_params(theta_r, theta_s, alpha, n, ks, l):
    with open(SELECTOR_IN) as f:
        lines = f.readlines()
    hdr = next(i for i, l_ in enumerate(lines)
               if l_.strip().startswith('thr') and 'Alfa' in l_)
    lines[hdr + 1] = f"   {theta_r:.6g}   {theta_s:.6g}   {alpha:.6g}    {n:.6g}      {ks:.6g}   {l:.6g}\n"
    with open(SELECTOR_IN, 'w') as f:
        f.writelines(lines)


def run_hydrus():
    exe = os.path.join(BASE_DIR, 'bin', 'mac' if platform.system() == 'Darwin' else 'windows',
                        'hydrus' if platform.system() == 'Darwin' else 'H1D_CALC.exe')
    if os.path.exists(OBS_NODE):
        os.remove(OBS_NODE)
    try:
        subprocess.run(exe, cwd=BASE_DIR, capture_output=True,
                        stdin=subprocess.DEVNULL, timeout=HYDRUS_TIMEOUT)
    except subprocess.TimeoutExpired:
        return False
    return os.path.exists(OBS_NODE)


def cumulative_outflow():
    with open(OBS_NODE) as f:
        lines = f.readlines()
    hdr_idx = next(i for i, l_ in enumerate(lines) if l_.split()[:1] == ['time'])
    df = pd.read_csv(io.StringIO(''.join(lines[hdr_idx + 1:])), header=None,
                      sep=r'\s+', engine='python', on_bad_lines='skip')
    df = df.apply(pd.to_numeric, errors='coerce').dropna(subset=[df.columns[0]])
    if df.empty:
        return None, None
    time = df.iloc[:, 0].to_numpy()
    flux = -df.iloc[:, 15].to_numpy()  # node201 Flux, sign-flipped to positive outflow
    return time[-1], np.trapz(flux, time)


def evaluate(theta_s, n, ks, theta_r, alpha, l):
    write_params(theta_r, theta_s, alpha, n, ks, l)
    if not run_hydrus():
        return None, None
    t_end, q = cumulative_outflow()
    if t_end is None or t_end < 1216:
        return None, None
    return t_end, q


def load_done():
    done = set()
    if os.path.exists(OUT_CSV):
        with open(OUT_CSV) as f:
            for row in csv.DictReader(f):
                done.add((row['p1'], row['p2'], row['v1'], row['v2']))
    return done


def main():
    select_model_ly2()
    done = load_done()
    write_header = not os.path.exists(OUT_CSV)

    pairs = list(itertools.combinations(GRIDS.keys(), 2))
    total = len(pairs) * 25
    i = 0
    with open(OUT_CSV, 'a', newline='') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(['p1', 'p2', 'v1', 'v2', 'q_sim', 't_end'])

        for p1, p2 in pairs:
            for v1 in GRIDS[p1]:
                for v2 in GRIDS[p2]:
                    i += 1
                    key = (p1, p2, f"{v1:.6g}", f"{v2:.6g}")
                    if key in done:
                        continue
                    kwargs = {PARAM_KWARG[p]: BASELINE[p] for p in GRIDS}
                    kwargs[PARAM_KWARG[p1]] = float(v1)
                    kwargs[PARAM_KWARG[p2]] = float(v2)
                    t_end, q = evaluate(kwargs.pop('theta_s'), kwargs.pop('n'),
                                        kwargs.pop('ks'), theta_r=kwargs['theta_r'],
                                        alpha=kwargs['alpha'], l=kwargs['l'])
                    status = 'FAIL' if q is None else f"Q={q:.3f}"
                    print(f"[{i}/{total}] {p1}={v1:.4g} {p2}={v2:.4g} -> {status}")
                    writer.writerow([p1, p2, f"{v1:.6g}", f"{v2:.6g}",
                                      '' if q is None else q, '' if t_end is None else t_end])
                    f.flush()

    print(f"Done -> {OUT_CSV}")


if __name__ == '__main__':
    main()
