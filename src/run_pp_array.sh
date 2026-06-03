#!/bin/bash
# SLURM array job script for PP test.
# Submitted by submit_pp.sh — do not call directly.

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
python blackjax_pp_injection.py --idx "${SLURM_ARRAY_TASK_ID}"

EXIT_CODE=$?

echo "========================================"
echo "End time:     $(date)"
echo "Exit code:    ${EXIT_CODE}"
echo "========================================"

exit ${EXIT_CODE}
