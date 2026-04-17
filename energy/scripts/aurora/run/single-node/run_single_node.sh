#!/bin/bash

# Set the directory of the script as the baseline path
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

nodes=(1) # For quick testing with a single node
collectives=("a2a" "ar") # TUNABLE: you can add/remove collectives as needed
occl_paths=("sycl" "lz_kernel" "lz_copy") # TUNABLE: you can add/remove collectives as needed
PARTITION="capacity" # TUNABLE: set the partition name for your cluster

# Configuration paths using SCRIPT_DIR as a baseline
PBS_TEMPLATE="$SCRIPT_DIR/../../pbs/run_coll.sh"
ENV_FILE="$SCRIPT_DIR/../../env/occl_set_env.sh"

# Base directory for binaries and logs
EXE_BASE="/home/lcarpent/energy-workspace/Blink-GPU/energy/build/bin"
LOG_BASE="/home/lcarpent/energy-workspace/Blink-GPU/energy/logs"

for node in "${nodes[@]}"; do
    for coll in "${collectives[@]}"; do
        for occl_path in "${occl_paths[@]}"; do
            RUN_COLL_PY="$SCRIPT_DIR/run_coll.py"
            echo "===================================================="
            echo "Processing: Collective=$coll, Path=$occl_path"
            python3 "$RUN_COLL_PY" \
                --pbs "$PBS_TEMPLATE" \
                --exe "$EXE_BASE/${coll}_occl" \
                --log "$LOG_BASE/" \
                --env "$ENV_FILE" \
                --path "${occl_path}" \
                --coll "${coll}" \
                --partition "${PARTITION}" \
                --time "00:30:00" \
                --nodes "${node}"
        done
    done
done