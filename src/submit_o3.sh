#!/bin/bash
# =============================================================================
# Submit O3 catalog runs to Isambard-AI.
#
# Usage:
#   bash submit_o3.sh [--batch-size N] [--catalog PATH] [--dry-run]
#
# Options:
#   --batch-size N   Events per job (default 3; use 1 for individual jobs)
#   --catalog PATH   Path to o3_catalog.npy (default ./o3_catalog.npy)
#   --dry-run        Print sbatch commands without submitting
#
# Build the catalog first:
#   python build_catalog.py --pastro 0.99 --out o3_catalog.npy
#
# Examples:
#   bash submit_o3.sh                      # 3 events/job → ~17 jobs of ~3.5h
#   bash submit_o3.sh --batch-size 1       # 1 event/job  → ~50 jobs of ~1.5h
#   bash submit_o3.sh --dry-run            # preview only
# =============================================================================

set -euo pipefail

SRC="$(cd "$(dirname "$0")" && pwd)"

# --- defaults ----------------------------------------------------------------
BATCH_SIZE=3
CATALOG="${SRC}/o3_catalog.npy"
DRY_RUN=false

# --- parse args --------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --batch-size) BATCH_SIZE="$2"; shift 2 ;;
        --catalog)    CATALOG="$2";    shift 2 ;;
        --dry-run)    DRY_RUN=true;    shift   ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# --- check catalog -----------------------------------------------------------
if [[ ! -f "$CATALOG" ]]; then
    echo "ERROR: catalog not found at $CATALOG"
    echo "Run:  python build_catalog.py --pastro 0.99 --out o3_catalog.npy"
    exit 1
fi

# --- count events ------------------------------------------------------------
N_EVENTS=$(python3 -c "
import numpy as np
cat = np.load('$CATALOG', allow_pickle=True).item()
print(len(cat))
")

N_TASKS=$(( (N_EVENTS + BATCH_SIZE - 1) / BATCH_SIZE ))

# time: ~60 min per event + 20 min headroom, rounded up to next 30 min
MINUTES=$(( BATCH_SIZE * 60 + 20 ))
MINUTES=$(( (MINUTES + 29) / 30 * 30 ))
TIME=$(printf "%02d:%02d:00" $(( MINUTES / 60 )) $(( MINUTES % 60 )))

mkdir -p "$SRC/logs"

echo "Catalog  : $CATALOG"
echo "Events   : $N_EVENTS"
echo "Batch    : $BATCH_SIZE events/job  →  $N_TASKS jobs"
echo "Time/job : $TIME"
echo ""

# --- submit ------------------------------------------------------------------
submit_job() {
    local start="$1"
    local name
    local end=$(( start + BATCH_SIZE - 1 ))
    [[ $end -ge $N_EVENTS ]] && end=$(( N_EVENTS - 1 ))
    name="o3_${start}-${end}"

    local cmd=(
        sbatch
        --partition=workq
        --gpus=1
        --job-name="$name"
        --time="$TIME"
        --output="$SRC/logs/${name}_%j.out"
        --error="$SRC/logs/${name}_%j.err"
        "$SRC/run.sh"
        bash "$SRC/run_o3_batch.sh" "$start" "$BATCH_SIZE" "$N_EVENTS" "$CATALOG"
    )

    if $DRY_RUN; then
        echo "  [DRY] ${cmd[*]}"
    else
        local jid
        jid=$("${cmd[@]}" | awk '{print $NF}')
        echo "  Submitted $name → job $jid"
    fi
}

for (( start=0; start<N_EVENTS; start+=BATCH_SIZE )); do
    submit_job "$start"
done

echo ""
if $DRY_RUN; then
    echo "Dry run complete — no jobs submitted."
else
    echo "All $N_TASKS jobs submitted."
    echo "Monitor: squeue -u \$USER"
    echo "Logs:    $SRC/logs/"
fi
