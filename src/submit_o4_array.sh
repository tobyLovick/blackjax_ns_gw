#!/bin/bash
#SBATCH --partition=workq
#SBATCH --gpus=1
#SBATCH --time=08:00:00
#SBATCH --job-name=o4_alcs
#SBATCH --output=logs/o4_%A_%a.out
#SBATCH --error=logs/o4_%A_%a.err
# Array index set at submission time:
#   sbatch --array=0-166 submit_o4_array.sh
# or a subset:
#   sbatch --array=0-4 submit_o4_array.sh

CONDA_ENV="lao"

source ~/miniforge3/bin/activate
conda activate "${CONDA_ENV}"

SRC="${SLURM_SUBMIT_DIR}"

echo "========================================"
echo "Job ID:       ${SLURM_JOB_ID}"
echo "Array task:   ${SLURM_ARRAY_TASK_ID}"
echo "Node:         $(hostname)"
echo "Start time:   $(date)"
echo "Working dir:  ${SRC}"
echo "Python:       $(which python)"
echo "========================================"

cd "${SRC}"
mkdir -p logs

python blackjaxns_o4_analysis.py \
    --event-idx "${SLURM_ARRAY_TASK_ID}" \
    --event-list o4_event_list.json \
    --n-live 1400 \
    --n-delete 700

EXIT_CODE=$?

echo "========================================"
echo "End time:     $(date)"
echo "Exit code:    ${EXIT_CODE}"
echo "========================================"

exit ${EXIT_CODE}
