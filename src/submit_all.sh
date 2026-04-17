#!/bin/bash
# =============================================================================
# Submit all 16 nested sampling runs (4s injection + GW150914, fixed + ALCS,
# no notch + t20 + t10 + posterior mask).
#
# Usage:
#   cd src/
#   bash submit_all.sh
#
# Prerequisites (already done locally, masks committed to repo):
#   compute_notch_mask.py, compute_notch_mask_4s.py,
#   compute_posterior_notch_mask.py, compute_posterior_notch_mask_4s.py
#
# The 4s and GW150914 posterior masks are pre-computed — no dependency needed.
# =============================================================================

# --- Isambard-AI settings ----------------------------------------------------
CONDA_ENV="lao"           # conda env with JAX, blackjax, jimgw etc.
TIME_LONG="12:00:00"      # main sampling runs
TIME_SHORT="01:00:00"     # posterior mask computation
# -----------------------------------------------------------------------------

SRC="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$SRC/logs"

submit_job() {
    local name="$1"
    local script="$2"
    local args="$3"
    local dep="$4"

    local dep_flag=""
    [[ -n "$dep" ]] && dep_flag="--dependency=afterok:${dep}"

    sbatch $dep_flag \
        --job-name="$name" \
        --time="$TIME_LONG" \
        --output="$SRC/logs/${name}_%j.out" \
        --error="$SRC/logs/${name}_%j.err" \
        "$SRC/run.sh" python "$script" $args \
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
        --time="$TIME_SHORT" \
        --output="$SRC/logs/${name}_%j.out" \
        --error="$SRC/logs/${name}_%j.err" \
        "$SRC/run.sh" python "$script" $args \
        | awk '{print $NF}'
}

echo "=== Phase 1: independent runs (12 jobs) ==="

J_4S_NORM_NONOTCH=$(submit_job  "4s_norm_nonotch"  blackjax_4s_norm.py  "")
J_4S_ALCS_NONOTCH=$(submit_job  "4s_alcs_nonotch"  blackjax_alcs_4s.py  "")
echo "4s no-notch:   norm=$J_4S_NORM_NONOTCH  alcs=$J_4S_ALCS_NONOTCH"

J_4S_NORM_T20=$(submit_job  "4s_norm_t20"  blackjax_4s_norm.py  "--notch")
J_4S_ALCS_T20=$(submit_job  "4s_alcs_t20"  blackjax_alcs_4s.py  "--notch")
echo "4s t20 notch:  norm=$J_4S_NORM_T20  alcs=$J_4S_ALCS_T20"

J_4S_NORM_T10=$(submit_job  "4s_norm_t10"  blackjax_4s_norm.py  "--notch --mask-suffix _t10")
J_4S_ALCS_T10=$(submit_job  "4s_alcs_t10"  blackjax_alcs_4s.py  "--notch --mask-suffix _t10")
echo "4s t10 notch:  norm=$J_4S_NORM_T10  alcs=$J_4S_ALCS_T10"

J_GW_NORM_NONOTCH=$(submit_job  "gw_norm_nonotch"  blackjax_gw150914_norm.py  "")
J_GW_ALCS_NONOTCH=$(submit_job  "gw_alcs_nonotch"  blackjax_alcs_gw150914.py  "")
echo "GW no-notch:   norm=$J_GW_NORM_NONOTCH  alcs=$J_GW_ALCS_NONOTCH"

J_GW_NORM_T20=$(submit_job  "gw_norm_t20"  blackjax_gw150914_norm.py  "--notch")
J_GW_ALCS_T20=$(submit_job  "gw_alcs_t20"  blackjax_alcs_gw150914.py  "--notch")
echo "GW t20 notch:  norm=$J_GW_NORM_T20  alcs=$J_GW_ALCS_T20"

J_GW_NORM_T10=$(submit_job  "gw_norm_t10"  blackjax_gw150914_norm.py  "--notch --mask-suffix _t10")
J_GW_ALCS_T10=$(submit_job  "gw_alcs_t10"  blackjax_alcs_gw150914.py  "--notch --mask-suffix _t10")
echo "GW t10 notch:  norm=$J_GW_NORM_T10  alcs=$J_GW_ALCS_T10"

echo ""
echo "=== Phase 2: posterior-masked runs (4 jobs, masks already exist) ==="

J_GW_NORM_POST=$(submit_job  "gw_norm_posterior"  blackjax_gw150914_norm.py  "--notch --mask-suffix _posterior")
J_GW_ALCS_POST=$(submit_job  "gw_alcs_posterior"  blackjax_alcs_gw150914.py  "--notch --mask-suffix _posterior")
echo "GW posterior:  norm=$J_GW_NORM_POST  alcs=$J_GW_ALCS_POST"

J_4S_NORM_POST=$(submit_job  "4s_norm_posterior"  blackjax_4s_norm.py  "--notch --mask-suffix _posterior")
J_4S_ALCS_POST=$(submit_job  "4s_alcs_posterior"  blackjax_alcs_4s.py  "--notch --mask-suffix _posterior")
echo "4s posterior:  norm=$J_4S_NORM_POST  alcs=$J_4S_ALCS_POST"

echo ""
echo "All 16 jobs submitted."
echo "Monitor with: squeue -u \$USER"
