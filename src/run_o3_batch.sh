#!/bin/bash
# Run a batch of consecutive catalog events in series.
# Called via run.sh so conda is already active.
#
# Usage: bash run_o3_batch.sh START BATCH_SIZE N_EVENTS CATALOG
START="$1"
BATCH_SIZE="$2"
N_EVENTS="$3"
CATALOG="$4"

END=$(( START + BATCH_SIZE - 1 ))
[[ $END -ge $N_EVENTS ]] && END=$(( N_EVENTS - 1 ))

echo "Batch: events ${START}–${END} of $(( N_EVENTS - 1 ))"

for (( idx=START; idx<=END; idx++ )); do
    echo ""
    echo ">>> Event ${idx}"
    python blackjax_gw_catalog.py --idx "$idx" --catalog "$CATALOG"
done
