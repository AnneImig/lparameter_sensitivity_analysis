#!/usr/bin/env zsh
# Record of how the Ly1 1M sweep was run (Pablo Ortega, HTCondor pool, Oct 2026). Needs the hydrus-sa
# package (branch stages-and-condor, see results/ly1_1M/run_manifest.json), which is not in this
# repository; the paths below are those of that machine. The post-mortem scripts it calls are in
# postmortem/ and run here on results/ly1_1M directly.
# Ly1 1,000,000-realization Condor sweep -> post-mortem scripts and figures -> GLUE -> report.html
# Run inside tmux savgn with the sensitivity_analysis env active:
#     zsh results/ly1_1M/run_chain.sh
# Writes under results/ly1_1M/results/model_ly1/sweep/; earlier runs are left untouched.
# Expect ~12-24 h for the sweep plus ~1-2 h of scoring and post-mortem.
set -o pipefail

REPO=/Users/scratch/portega/SA_6VGN_parameters
ROOT=$REPO/results/ly1_1M
STAGE=$ROOT/results/model_ly1/sweep
SRC=$ROOT/postmortem_scripts    # the ly1_5000 copies; chunked, scale with sample size
LOGS=$ROOT/logs
REALS=1000000
CHUNK=20000                     # pestpp-swp batch size (default 500 idles on stragglers)
MIN_FREE_GB=40                  # sweep_out.csv ~7 GB, copied once into the stage dir
mkdir -p $LOGS
cd $REPO

stamp() { echo "=== $(date '+%F %T') $*"; }
free_gb() { df -g $REPO | awk 'NR==2{print $4}'; }

# --- pre-flight -------------------------------------------------------------
# Port 4004 is shared with rtm-dsivc's pestpp-ies masters on this machine.
if lsof -nP -iTCP:4004 -sTCP:LISTEN >/dev/null; then
    echo "Port 4004 is in use by another PEST++ master; not starting." >&2
    exit 1
fi
# The Aug 28 1M attempt died at 83% on a full disk.
if (( $(free_gb) < MIN_FREE_GB )); then
    echo "Only $(free_gb) GB free on $REPO, need $MIN_FREE_GB; not starting." >&2
    exit 1
fi
# Orphans from an earlier cluster hold pool slots the new workers need.
if condor_q -af ClusterId 2>/dev/null | grep -qx 316; then
    echo "Cluster 316's orphaned workers are still queued; run condor_rm 316 first." >&2
    exit 1
fi

# Log free disk every 10 min for the life of the chain.
( while true; do echo "$(date '+%F %T') $(free_gb) GB free"; sleep 600; done ) >> $LOGS/disk.log &
DISKMON=$!
trap 'kill $DISKMON 2>/dev/null' EXIT

# --- sweep ------------------------------------------------------------------
stamp "sweep: model_ly1, $REALS reals, chunk $CHUNK, Condor, 70 workers"
hydrus-sa sweep --model model_ly1 --reals $REALS --chunk $CHUNK --allow-failures \
    --condor --workers 70 --root $ROOT 2>&1 | tee $LOGS/sweep.log
SWEEP_RC=$?

# Workers that Condor evicted and restarted never hear the master's shutdown and
# spin on "failed to connect" forever (cluster 316). Remove this run's cluster,
# and only this one -- never condor_rm by user, rtm-dsivc shares the pool.
CLUSTER=$(grep -ao 'cluster [0-9]*' $LOGS/sweep.log | head -1 | awk '{print $2}')
if [[ -n $CLUSTER ]] && condor_q -af ClusterId 2>/dev/null | grep -qx $CLUSTER; then
    stamp "removing leftover workers of cluster $CLUSTER"
    condor_rm $CLUSTER
fi

if (( SWEEP_RC != 0 )); then
    if [[ -f $STAGE/sweep_out.csv ]]; then
        stamp "sweep exited $SWEEP_RC but sweep_out.csv exists; carrying on"
    else
        stamp "sweep FAILED ($SWEEP_RC) with no sweep_out.csv, stopping"; exit 1
    fi
fi

# --- post-mortem ------------------------------------------------------------
stamp "post-mortem: copying scripts from $SRC"
cp $SRC/{score_partial,sentinel_split,partial_rescore,analyse_partial,calibration_plot,crash_surface}.py $STAGE/
cp $SRC/report.html.tmpl $STAGE/
mkdir -p $STAGE/figures
cd $STAGE

for step in score_partial sentinel_split partial_rescore analyse_partial calibration_plot crash_surface; do
    stamp "post-mortem: $step.py"
    if [[ $step == score_partial ]]; then
        python $step.py . 2>&1 | tee $STAGE/$step.log || stamp "$step.py FAILED"
    else
        python $step.py 2>&1 | tee $STAGE/$step.log || stamp "$step.py FAILED"
    fi
done

for f in glue_prior glue_behavioral; do
    [[ -f $f.csv ]] && mv $f.csv ${f}_top1pct_kge.csv
done

# --- GLUE -------------------------------------------------------------------
cd $REPO
stamp "glue: threshold 0.8"
hydrus-sa glue --model model_ly1 --threshold 0.8 --root $ROOT 2>&1 | tee $LOGS/glue.log \
    || stamp "glue FAILED"

# --- report figures ---------------------------------------------------------
stamp "report: embedding figures in report.html (prose rewritten afterwards for the artifact)"
cd $STAGE
python - <<'PY' 2>&1 | tee $LOGS/report.log || stamp "report FAILED"
import base64, pathlib
figs = {'SCORES': 'scores', 'CALIBRATION': 'calibration', 'LOSS': 'loss_by_parameter',
        'KGE_BY_PARAM': 'kge_by_parameter', 'SURFACE': 'response_surface',
        'BEST_ACHIEVABLE': 'best_achievable', 'PRIOR_TOP': 'param_prior_top'}
html = pathlib.Path('report.html.tmpl').read_text()
for key, name in figs.items():
    png = pathlib.Path(f'figures/{name}.png')
    if not png.exists():
        print(f'missing figures/{name}.png -- {{{{{key}}}}} left blank')
        html = html.replace('{{%s}}' % key, '')
        continue
    b64 = base64.b64encode(png.read_bytes()).decode()
    html = html.replace('{{%s}}' % key, 'data:image/png;base64,' + b64)
pathlib.Path('report.html').write_text(html)
print('wrote report.html')
PY

stamp "done"
