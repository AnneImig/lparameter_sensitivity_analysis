#!/usr/bin/env zsh
# Two 1,000,000-realization sweeps with l fixed at 0.5 (Ly1, then Ly2), each paired
# run for run with its l-free 1M sweep (same seed, same bounds, so only l differs).
# Per lysimeter: sweep with water-balance totals -> remove its cluster -> post-mortem
# -> paired comparison -> delete raw/ (PEST++ staging and master, ~8 GB) to make
# room for the next sweep.
# Run inside tmux savgn with the sensitivity_analysis env active:
#     zsh results/l05_1M/run_chain.sh
# Expect ~16-20 h for Ly1 and ~15 h for Ly2, plus ~1 h of scoring and analysis each.
set -o pipefail

REPO=/Users/scratch/portega/SA_6VGN_parameters
ROOT=$REPO/results/l05_1M
LOGS=$ROOT/logs
REALS=1000000
CHUNK=20000
MIN_FREE_GB=17                  # sweep_out.csv ~6.8 GB in raw/master plus its copy, plus staging
mkdir -p $LOGS
cd $REPO

stamp() { echo "=== $(date '+%F %T') $*"; }
free_gb() { df -g $REPO | awk 'NR==2{print $4}'; }

( while true; do echo "$(date '+%F %T') $(free_gb) GB free"; sleep 600; done ) >> $LOGS/disk.log &
DISKMON=$!
trap 'kill $DISKMON 2>/dev/null' EXIT

for LY in ly1 ly2; do
    MODEL=model_$LY
    STAGE=$ROOT/results/$MODEL/sweep
    BOUNDS=()
    # The Ly2 1M sweep ran with these overrides; the paired sample needs the same.
    [[ $LY == ly2 ]] && BOUNDS=(--bound alpha 1e-3 0.5 --bound n 1.05 4)

    # --- pre-flight (port 4004 is shared with rtm-dsivc's pestpp-ies masters) ---
    if lsof -nP -iTCP:4004 -sTCP:LISTEN >/dev/null; then
        stamp "port 4004 is in use by another PEST++ master; stopping before $MODEL"; exit 1
    fi
    if [[ -n $(condor_q -af ClusterId 2>/dev/null) ]]; then
        stamp "jobs already queued ($(condor_q -af ClusterId | sort -u | tr '\n' ' ')); stopping before $MODEL"; exit 1
    fi
    if (( $(free_gb) < MIN_FREE_GB )); then
        stamp "only $(free_gb) GB free, need $MIN_FREE_GB; stopping before $MODEL"; exit 1
    fi

    # --- sweep ------------------------------------------------------------------
    stamp "sweep: $MODEL, $REALS reals, l fixed at 0.5, water balance, chunk $CHUNK, Condor, 70 workers"
    hydrus-sa sweep --model $MODEL --reals $REALS --chunk $CHUNK --allow-failures \
        $BOUNDS --fix l 0.5 --water-balance \
        --condor --workers 70 --root $ROOT 2>&1 | tee $LOGS/sweep_$LY.log
    SWEEP_RC=$?

    # Evicted-and-restarted workers never hear the master's shutdown (cluster 316).
    CLUSTER=$(grep -ao 'cluster [0-9]*' $LOGS/sweep_$LY.log | head -1 | awk '{print $2}')
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

    # --- post-mortem --------------------------------------------------------------
    cp $ROOT/postmortem_$LY/{score_partial,sentinel_split,partial_rescore,analyse_partial,calibration_plot}.py $STAGE/
    mkdir -p $STAGE/figures
    cd $STAGE
    for step in score_partial sentinel_split partial_rescore analyse_partial calibration_plot; do
        stamp "post-mortem $LY: $step.py"
        if [[ $step == score_partial ]]; then
            python $step.py . 2>&1 | tee $STAGE/$step.log || stamp "$step.py FAILED"
        else
            python $step.py 2>&1 | tee $STAGE/$step.log || stamp "$step.py FAILED"
        fi
    done
    cd $REPO

    stamp "paired comparison with the l-free sweep: $MODEL"
    python $ROOT/scripts/compare_l.py $MODEL 2>&1 | tee $LOGS/compare_$LY.log
    COMPARE_RC=$?

    # --- make room for the next sweep -------------------------------------------
    # Only once everything that reads raw/ has run and left its output behind.
    if (( COMPARE_RC == 0 )) && [[ -f $STAGE/metrics_partial.csv && -f $STAGE/compare_summary.json ]]; then
        stamp "deleting $STAGE/raw ($(du -sh $STAGE/raw | cut -f1))"
        rm -rf $STAGE/raw
    else
        stamp "post-mortem or comparison incomplete; keeping raw/ and stopping"; exit 1
    fi
done

stamp "done"
