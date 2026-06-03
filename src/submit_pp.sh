#!/bin/bash
#SBATCH --partition=workq
#SBATCH --gpus=1
#SBATCH --time=04:00:00
#SBATCH --job-name=pp_test
#SBATCH --array=0-499
#SBATCH --output=slurm_logs/pp_%A_%a.out
#SBATCH --error=slurm_logs/pp_%A_%a.err

CONDA_ENV="lao"

source ~/miniforge3/bin/activate
conda activate "${CONDA_ENV}"

echo "========================================"
echo "Job ID:       ${SLURM_JOB_ID}"
echo "Array task:   ${SLURM_ARRAY_TASK_ID}"
echo "Node:         $(hostname)"
echo "Start time:   $(date)"
echo "Working dir:  $(pwd)"
echo "Python:       $(which python)"
echo "========================================"

mkdir -p slurm_logs

python blackjax_pp_injection.py --idx "${SLURM_ARRAY_TASK_ID}"

EXIT_CODE=$?

echo "========================================"
echo "End time:     $(date)"
echo "Exit code:    ${EXIT_CODE}"
echo "========================================"

exit ${EXIT_CODE}
