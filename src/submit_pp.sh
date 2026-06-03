#!/bin/bash
# Submit 500 PP-test array jobs to Isambard-AI.
#
# Usage (from src/):
#   bash submit_pp.sh [--dry-run]

set -euo pipefail

SRC="$(cd "$(dirname "$0")" && pwd)"
DRY_RUN=false
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=true

mkdir -p "$SRC/logs"

cmd=(
    sbatch
    --partition=workq
    --gpus=1
    --job-name=pp_test
    --array=0-499
    --time=04:00:00
    --output="$SRC/logs/pp_%A_%a.out"
    --error="$SRC/logs/pp_%A_%a.err"
    "$SRC/run_pp_array.sh"
)

if $DRY_RUN; then
    echo "[DRY] ${cmd[*]}"
else
    jid=$("${cmd[@]}" | awk '{print $NF}')
    echo "Submitted 500-job PP array → job $jid"
    echo "Monitor: squeue -u \$USER"
    echo "Logs:    $SRC/logs/pp_${jid}_*.out"
fi
