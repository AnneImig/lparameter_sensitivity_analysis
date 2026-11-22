#!/usr/bin/env python3
"""
Shared observation definitions for the PEST setup.

Both create_pest_setup.py (builds obs_names/obs_vals for the .pst) and
forward_run.py (interpolates model output and writes sim_obs.dat) import
this module so the two stay in lock-step: the instruction file order and
the sim_obs.dat write order are generated from the exact same list.

Column layout of tmp_copy/Obs_Node.out (0-indexed, time is column 0),
confirmed against the file header (nodes 1, 21, 61, 101, 176, 251, each
with h, theta, Temp, Conc):

    node1:   h=1  theta=2   Temp=3   Conc=4
    node21:  h=5  theta=6   Temp=7   Conc=8
    node61:  h=9  theta=10  Temp=11  Conc=12
    node101: h=13 theta=14  Temp=15  Conc=16
    node176: h=17 theta=18  Temp=19  Conc=20
    node251: h=21 theta=22  Temp=23  Conc=24

ST_observed.txt (soil temperature) is intentionally NOT wired in here:
tmp_copy/SELECTOR.IN has lTemp = f (heat transport disabled), so HYDRUS
is not simulating temperature in this setup -- the Temp columns above are
placeholder near-zero values, not a real model output to compare against.
"""

import glob
import os
import numpy as np
import pandas as pd

# Nodes carrying real data per observed file (checked against tmp_copy data:
# SWI's 'n176' column is all zeros -- no isotope/tracer data at that depth --
# and is dropped automatically by the zero-as-missing rule below).
WC_NODES  = ['n21', 'n61', 'n101', 'n176']
SWI_NODES = ['n21', 'n61', 'n101', 'n176', 'n251']
SP_NODES  = ['sp']  # single column: pressure head at node 1

# Obs_Node.out column index for each (group, node)
OBS_NODE_COLS = {
    'swc': {'n21': 6,  'n61': 10, 'n101': 14, 'n176': 18},               # theta
    'swi': {'n21': 8,  'n61': 12, 'n101': 16, 'n176': 20, 'n251': 24},   # Conc
    'sp':  {'sp': 1},                                                    # h @ node1
}
# NOTE: forward_run.py negates the 'sp' model series before comparison --
# SP_observed.txt is positive (surface ponding depth) while HYDRUS node-1 h
# is negative matric suction under dry conditions. Assumed sign flip for
# now; revisit once the ATMOSPH.IN forcing/timing for that window is
# confirmed.

# (group label, observed-data filename, node columns in that file)
OBS_FILE_SPECS = [
    ('swc', 'SWC_observed.txt', WC_NODES),
    ('swi', 'SWI_observed.txt', SWI_NODES),
    ('sp',  'SP_observed.txt',  SP_NODES),
]

# ── Lysimeter models: Model_Ly1, Model_Ly2 (single-layer, lChem=f/lTemp=f) ────
# They share this layout exactly (identical SELECTOR.IN/PROFILE.DAT),
# differing only in the measured outflow in their COMPARE_Ly<N>_Qmeas*.txt.
# Their Obs_Node.out has a different node set and layout than the profile
# models: 5 nodes (1, 11, 21, 51, 201), each with h/theta/Flux (no
# Temp/Conc, since solute/heat transport are off). Node 201 is the bottom
# of the 201-node profile, so its Flux column is the simulated outflow to
# compare against the measured lysimeter outflow in COMPARE_Ly1_Qmeas*.txt:
#   node1:   h=1  theta=2   Flux=3
#   node11:  h=4  theta=5   Flux=6
#   node21:  h=7  theta=8   Flux=9
#   node51:  h=10 theta=11  Flux=12
#   node201: h=13 theta=14  Flux=15
SWI_Q_NODE = 'n201'
SWI_OBS_NODE_COLS = {'q': {'n201': 15}}

# HYDRUS reports Flux as a flux density (cm/day, per Obs_Node.out's header
# "Units: L = cm, T = days") -- volume per unit cross-sectional area per
# time -- while Qmeas in COMPARE_Ly1_Qmeas*.txt is an absolute lysimeter
# outflow rate in L/day. They're different physical quantities until
# scaled by the lysimeter's actual footprint: L/day = Flux[cm/day] *
# SWI_LYSIMETER_AREA_CM2 / 1000 (cm^3 -> L). See forward_run.py's
# read_obs_node() for where this is applied.
SWI_LYSIMETER_AREA_CM2 = 10_000.0  # 1 m^2


# Marker file select_model() writes into MODEL_DIR (tmp_copy/) to restrict
# build_observations_profile() to a subset of groups (e.g. 'swi' only for
# 'profile2_1layer_swi' in create_pest_setup.py). Needed because both
# create_pest_setup.py (building sim_obs.ins) and forward_run.py (writing
# sim_obs.dat) call build_observations() independently -- filtering only in
# one place would leave the two out of line-order sync. tmp_copy/'s file
# contents alone can't distinguish 'profile2_1layer' from
# 'profile2_1layer_swi' (same source folder), unlike is_swi_model()'s
# COMPARE_Ly1_Qmeas*.txt detection, hence the explicit marker.
OBS_GROUP_FILTER_MARKER = '.obs_group_filter'


def _read_obs_group_filter(model_dir):
    """Return the set of allowed groups if select_model() wrote a filter
    marker, else None (no filter -- keep every group with data)."""
    path = os.path.join(model_dir, OBS_GROUP_FILTER_MARKER)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        content = f.read().strip()
    return set(content.split(',')) if content else None


def _find_compare_file(model_dir):
    """Return the lysimeter model's COMPARE_Ly<N>_Qmeas*.txt path in
    model_dir, or None. Matched by glob rather than by a fixed Ly1 name so
    every lysimeter model (Model_Ly1, Model_Ly2, ...) is picked up -- they
    share this identical wiring and differ only in the measured outflow
    series. select_model() purges the previous model's COMPARE file from
    model_dir, so exactly one is ever present; if that ever breaks, this
    raises rather than silently calibrating against the wrong lysimeter."""
    matches = sorted(glob.glob(os.path.join(model_dir, 'COMPARE_Ly*_Qmeas*.txt')))
    if len(matches) > 1:
        raise ValueError(
            f"{model_dir} contains {len(matches)} COMPARE_Ly*_Qmeas*.txt files "
            f"({[os.path.basename(m) for m in matches]}) -- can't tell which lysimeter's "
            "outflow to calibrate against. Delete the stale one(s) and re-run select_model()."
        )
    return matches[0] if matches else None


def is_swi_model(model_dir):
    """True if the currently-selected model is a lysimeter model
    (Model_Ly1/Model_Ly2/..., detected by its distinctive
    COMPARE_Ly<N>_Qmeas*.txt file), vs. one of the SWC/SWI/SP profile
    models."""
    return _find_compare_file(model_dir) is not None


def _read_compare_q(path):
    """Read a lysimeter model's COMPARE_Ly<N>_Qmeas*.txt (tab-separated:
    Date, t[d], Qmeas[L/d]). Returns list of (time, value)."""
    df = pd.read_csv(path, sep='\t', usecols=[1, 2],
                      names=['time', 'value'], header=0, engine='python')
    df = df.dropna()
    return list(zip(df['time'].astype(float), df['value'].astype(float)))


def build_observations_swi(model_dir):
    """Build the observation list for a lysimeter model (Model_Ly1,
    Model_Ly2, ...): measured lysimeter outflow (Qmeas, from
    COMPARE_Ly<N>_Qmeas*.txt) vs. simulated Flux at the last profile node
    (201, the bottom boundary). See forward_run.py's read_obs_node() for
    the matching sign flip (HYDRUS reports outflow as negative; Qmeas is
    positive).

    TIME BASE: the COMPARE file's t[d] column counts days on the
    lysimeter record's own origin (t=185 is 2013-01-01), while HYDRUS
    counts from tInit=0 with ATMOSPH.IN record 1 = 2013-01-01. Verified
    against the source data: ATMOSPH.IN's first six Prec values (0.021,
    0.13, 0.155, 0.26, 2.904, 0.22 cm) are exactly the measured
    precipitation for 2013-01-01..06, and the file's 1216 records match
    the 1216 COMPARE rows and SELECTOR.IN's tMax=1216 one-for-one. So
    model day 1 IS compare row 1, and observation times must be shifted
    down by (first t - 1) before being interpolated onto model output.

    Without this shift every observation was compared against simulated
    outflow 184 days off (roughly the opposite season), and the last 184
    observations fell past tMax entirely -- forward_run.py writes -9999
    for those, which alone contributed ~184 * 9999^2 = 1.84e10 to phi and
    swamped everything else in it.

    The offset is derived from the file rather than hardcoded, so a
    COMPARE file on a different day-origin still lines up as long as its
    first row is the model's first day. Observation NAMES deliberately
    keep the file's own t so they stay traceable back to a date in the
    COMPARE file (and stable against previously-built .pst files); only
    the interpolation time is shifted."""
    path = _find_compare_file(model_dir)
    if path is None:
        raise FileNotFoundError(f"No COMPARE_Ly*_Qmeas*.txt found in {model_dir}")
    records = _read_compare_q(path)
    if not records:
        raise ValueError(f"No usable rows in {path}")

    t_offset = records[0][0] - 1.0  # compare-file t of model day 1, minus that day
    obs = []
    for t, v in records:
        obs.append({'name': f"t{int(t)}_q", 'value': v, 'group': 'q',
                    'node': SWI_Q_NODE, 'time': t - t_offset})
    return obs


def _read_node_series(path, node_names):
    """Read a whitespace-delimited 'time node1 node2 ...' file.

    A value of exactly 0 means "no measurement" in these files (confirmed:
    SWI's n176 column is 100% zeros, and other SWI/SP columns have scattered
    zeros) so those rows are dropped per node, matching the convention used
    in 01_SA_6par_Tiantians_models.py (`obs.replace(0, np.nan)`).

    Returns dict: node_name -> list of (time, value) with zeros removed.
    """
    df = pd.read_csv(path, sep=r'\s+', header=None, engine='python')
    df.columns = ['time'] + node_names
    out = {}
    for node in node_names:
        sub = df[['time', node]].replace(0, np.nan).dropna()
        out[node] = list(zip(sub['time'].astype(float), sub[node].astype(float)))
    return out


def build_observations(model_dir):
    """Dispatch to the right obs builder for the currently-selected model,
    detected by whether a COMPARE_Ly<N>_Qmeas*.txt is present in model_dir
    (lysimeter models Model_Ly1/Model_Ly2/...) or not (profile models'
    SWC/SWI/SP wiring)."""
    if is_swi_model(model_dir):
        return build_observations_swi(model_dir)
    return build_observations_profile(model_dir)


def build_observations_profile(model_dir):
    """Build the full, ordered observation list across SWC, SWI, and SP.

    Returns a list of dicts with keys: name, value, group, node, time.
    Naming: 'swc' keeps the original t{time}_{node} form (unchanged, so it
    matches obs names already baked into hydrus.pst); 'swi' appends 'c'
    (concentration) to disambiguate from swc at the same node; 'sp' has a
    single pseudo-node.
    """
    obs = []
    for group, fname, nodes in OBS_FILE_SPECS:
        path = os.path.join(model_dir, fname)
        node_series = _read_node_series(path, nodes)
        for node in nodes:
            for t, v in node_series[node]:
                if group == 'swc':
                    name = f"t{int(t)}_{node}"
                elif group == 'swi':
                    name = f"t{int(t)}_{node}c"
                else:  # sp
                    name = f"t{int(t)}_sp"
                obs.append({'name': name, 'value': v, 'group': group,
                            'node': node, 'time': t})

    allowed = _read_obs_group_filter(model_dir)
    if allowed is not None:
        obs = [o for o in obs if o['group'] in allowed]
    return obs
