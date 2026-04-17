#!/bin/bash
# =============================================================================
# Submit all 16 nested sampling runs (4s injection + GW150914, fixed + ALCS,
# no notch + t20 + t10 + posterior mask).
#
# Usage:
#   cd src/
#   bash submit_all.sh
#
# Prerequisites (run once before submitting):
#   python compute_notch_mask.py                        # GW150914 t20 (already done)
#   python compute_notch_mask.py --threshold 10 --suffix _t10
#   python compute_notch_mask_4s.py                     # 4s t20
#   python compute_notch_mask_4s.py --threshold 10 --suffix _t10
#   python compute_posterior_notch_mask.py              # GW150914 posterior (uses existing CSV)
#
# The 4s posterior mask is computed as a SLURM job that depends on the
# 4s ALCS no-notch run completing first.
# =============================================================================

# --- Cluster settings (adjust to your system) --------------------------------
PARTITION="gpu"
GRES="gpu:1"
MEM="32G"
TIME_LONG="12:00:00"   # for the main sampling runs
TIME_SHORT="01:00:00"  # for mask computation
PYTHON="python"
# -----------------------------------------------------------------------------

SRC="$(cd "$(dirname "$0")" && pwd)"

submit_job() {
    local name="$1"
    local script="$2"
    local args="$3"
    local dep="$4"

    local dep_flag=""
    [[ -n "$dep" ]] && dep_flag="--dependency=afterok:${dep}"

    sbatch $dep_flag \
        --job-name="$name" \
        --partition="$PARTITION" \
        --gres="$GRES" \
        --mem="$MEM" \
        --time="$TIME_LONG" \
        --output="logs/${name}_%j.out" \
        --error="logs/${name}_%j.err" \
        --wrap="cd $SRC && $PYTHON $script $args" \
        | awk '{print $NF}'
}

submit_mask_job() {
    local name="$1"
    local script="$2"
    local args="$3"
    local dep="$4"

    local dep_flag=""
    [[ -n "$dep" ]] && dep_flag="--dependency=afterok:${dep}"

    sbatch $dep_flag \
        --job-name="$name" \
        --partition="$PARTITION" \
        --gres="$GRES" \
        --mem="16G" \
        --time="$TIME_SHORT" \
        --output="logs/${name}_%j.out" \
        --error="logs/${name}_%j.err" \
        --wrap="cd $SRC && $PYTHON $script $args" \
        | awk '{print $NF}'
}

mkdir -p "$SRC/logs"

echo "=== Phase 1: independent runs (12 jobs) ==="

# --- 4s injection, no notch ---
J_4S_NORM_NONOTCH=$(submit_job   "4s_norm_nonotch"   blackjax_4s_norm.py    "")
J_4S_ALCS_NONOTCH=$(submit_job   "4s_alcs_nonotch"   blackjax_alcs_4s.py    "")
echo "4s no-notch:   norm=$J_4S_NORM_NONOTCH  alcs=$J_4S_ALCS_NONOTCH"

# --- 4s injection, t20 notch ---
J_4S_NORM_T20=$(submit_job   "4s_norm_t20"   blackjax_4s_norm.py   "--notch")
J_4S_ALCS_T20=$(submit_job   "4s_alcs_t20"   blackjax_alcs_4s.py   "--notch")
echo "4s t20 notch:  norm=$J_4S_NORM_T20  alcs=$J_4S_ALCS_T20"

# --- 4s injection, t10 notch ---
J_4S_NORM_T10=$(submit_job   "4s_norm_t10"   blackjax_4s_norm.py   "--notch --mask-suffix _t10")
J_4S_ALCS_T10=$(submit_job   "4s_alcs_t10"   blackjax_alcs_4s.py   "--notch --mask-suffix _t10")
echo "4s t10 notch:  norm=$J_4S_NORM_T10  alcs=$J_4S_ALCS_T10"

# --- GW150914, no notch ---
J_GW_NORM_NONOTCH=$(submit_job   "gw_norm_nonotch"   blackjax_gw150914_norm.py   "")
J_GW_ALCS_NONOTCH=$(submit_job   "gw_alcs_nonotch"   blackjax_alcs_gw150914.py   "")
echo "GW no-notch:   norm=$J_GW_NORM_NONOTCH  alcs=$J_GW_ALCS_NONOTCH"

# --- GW150914, t20 notch ---
J_GW_NORM_T20=$(submit_job   "gw_norm_t20"   blackjax_gw150914_norm.py   "--notch")
J_GW_ALCS_T20=$(submit_job   "gw_alcs_t20"   blackjax_alcs_gw150914.py   "--notch")
echo "GW t20 notch:  norm=$J_GW_NORM_T20  alcs=$J_GW_ALCS_T20"

# --- GW150914, t10 notch ---
J_GW_NORM_T10=$(submit_job   "gw_norm_t10"   blackjax_gw150914_norm.py   "--notch --mask-suffix _t10")
J_GW_ALCS_T10=$(submit_job   "gw_alcs_t10"   blackjax_alcs_gw150914.py   "--notch --mask-suffix _t10")
echo "GW t10 notch:  norm=$J_GW_NORM_T10  alcs=$J_GW_ALCS_T10"

echo ""
echo "=== Phase 1.5: compute 4s posterior mask (depends on 4s ALCS no-notch) ==="

J_4S_POSTMASK=$(submit_mask_job \
    "4s_posterior_mask" \
    compute_posterior_notch_mask_4s.py \
    "" \
    "")
echo "4s posterior mask job: $J_4S_POSTMASK"

echo ""
echo "=== Phase 2: posterior-masked runs (4 jobs) ==="

J_GW_POSTMASK=$(submit_mask_job \
    "gw_posterior_mask" \
    compute_posterior_notch_mask.py \
    "--samples-file blackjaxns_gw150914_notched.csv" \
    "")
echo "GW posterior mask job: $J_GW_POSTMASK"

J_GW_NORM_POST=$(submit_job   "gw_norm_posterior"   blackjax_gw150914_norm.py   "--notch --mask-suffix _posterior"  "$J_GW_POSTMASK")
J_GW_ALCS_POST=$(submit_job   "gw_alcs_posterior"   blackjax_alcs_gw150914.py   "--notch --mask-suffix _posterior"  "$J_GW_POSTMASK")
echo "GW posterior:  norm=$J_GW_NORM_POST  alcs=$J_GW_ALCS_POST"

J_4S_NORM_POST=$(submit_job   "4s_norm_posterior"   blackjax_4s_norm.py   "--notch --mask-suffix _posterior"  "$J_4S_POSTMASK")
J_4S_ALCS_POST=$(submit_job   "4s_alcs_posterior"   blackjax_alcs_4s.py   "--notch --mask-suffix _posterior"  "$J_4S_POSTMASK")
echo "4s posterior:  norm=$J_4S_NORM_POST  alcs=$J_4S_ALCS_POST"

echo ""
echo "All jobs submitted. Total: 16 sampling + 1 mask job."
echo "Monitor with: squeue -u \$USER"
