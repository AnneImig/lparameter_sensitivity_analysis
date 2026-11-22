#!/usr/bin/env zsh
# Ly1 Morris, 10 levels (same ranges as ly1_morris_swp2, which used 4), SALib trajectories (r=2000, p=10; l [-2, 5], n [1.05, 2.3], alpha from 0.005, theta_s from 0.15, Ks from 4,000),
# run through pestpp-swp on Condor so every run's outflow and water-balance totals
# are kept -> remove this run's cluster -> Morris with and without the runs that take
# in more water than rained, five objectives, figures.
# Run inside tmux savgn with the sensitivity_analysis env active:
#     zsh 3.4_SA_SEN_PEST/scripts/morris_ly1/run_chain.sh (on the cluster)
# Writes under results/ly1_morris_p10/results/model_ly1/morris_swp/. Expect ~15-25 min
# for the 14,000 runs on 70 cores (7 wombats x 10), then a few minutes of analysis.
set -o pipefail

REPO=/Users/scratch/portega/SA_6VGN_parameters
ROOT=$REPO/results/ly1_morris_p10
STAGE=$ROOT/results/model_ly1/morris_swp
LOGS=$ROOT/logs
R=2000
mkdir -p $LOGS
cd $REPO

stamp() { echo "=== $(date '+%F %T') $*"; }

# --- pre-flight -------------------------------------------------------------
# Port 4004 is shared with rtm-dsivc's pestpp-ies masters on this machine.
if lsof -nP -iTCP:4004 -sTCP:LISTEN >/dev/null; then
    echo "Port 4004 is in use by another PEST++ master; not starting." >&2
    exit 1
fi

# sweep_out.csv: 14,000 runs x 1,221 outputs, ~200 MB.
if (( $(df -g $REPO | awk 'NR==2{print $4}') < 5 )); then
    echo "Less than 5 GB free on $REPO; not starting." >&2
    exit 1
fi

# --- rerun --------------------------------------------------------------------
stamp "pestpp-swp: 14,000 Ly1 Morris points, daily Q + water-balance totals, Condor, 70 workers"
python $REPO/3.4_SA_SEN_PEST/scripts/morris_ly1/run_morris.py --workers 70 --chunk 2000 2>&1 | tee $LOGS/morris.log
RUN_RC=$?

# Evicted-and-restarted workers never hear the master's shutdown (cluster 316).
# Remove this run's cluster only -- never condor_rm by user, rtm-dsivc shares the pool.
CLUSTER=$(grep -ao 'cluster [0-9]*' $LOGS/morris.log | head -1 | awk '{print $2}')
if [[ -n $CLUSTER ]] && condor_q -af ClusterId 2>/dev/null | grep -qx $CLUSTER; then
    stamp "removing leftover workers of cluster $CLUSTER"
    condor_rm $CLUSTER
fi

if (( RUN_RC != 0 )); then
    stamp "morris run exited $RUN_RC; stopping before the analysis"
    exit $RUN_RC
fi

# --- analysis and figures ---------------------------------------------------
stamp "Morris with and without non-physical runs, five objectives"
python $REPO/3.4_SA_SEN_PEST/scripts/morris_ly1/analyze.py $STAGE $R 2>&1 | tee $LOGS/analysis.log

stamp "done; figures in $STAGE/analysis/figures"
