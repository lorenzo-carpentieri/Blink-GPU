#!/bin/bash
################# oneCCL Multi-node #########################################################
# We are testing two collectives alltoall and allreduce.
# The multi-node alltoall does not support the SYCL path and fallback to the level zero one.
# Differently for the allreduce SYCL path is supported also for multi-node.
#############################################################################################

# Set the directory of the script as the baseline path
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# Arrays for nodes (1 to 128) and collectives
# nodes=(1 2 4 8 16 32 64 128) # TUNABLE: you can add/remove nodes as needed
nodes=(2) # For quick testing with a single node
collectives=("a2a") # TUNABLE: you can add/remove collectives as needed
# occl_paths=("sycl" "lz_kernel" "lz_copy") # TUNABLE: you can add/remove collectives as needed
occl_paths=("sycl_kernel") # TUNABLE: you can add/remove collectives as needed
a2a_algs=("scatter" "naive")
# ar_algs=("direct" "ring" "rabenseifner" "nreduce" "double_tree" "recursive_doubling")
ar_algs=("direct" "ring")
 
# collectives=("ar") # For quick testing with a single collective
PARTITION="debug-scaling" # TUNABLE: set the partition name for your cluster

# Configuration paths using SCRIPT_DIR as a baseline
PBS_TEMPLATE="$SCRIPT_DIR/../../pbs/run_coll.sh"
ENV_FILE="$SCRIPT_DIR/../../env/occl_set_env.sh"

# Base directory for binaries and logs
EXE_BASE="/home/lcarpent/energy-workspace/Blink-GPU/energy/build/bin"
LOG_BASE="/home/lcarpent/energy-workspace/Blink-GPU/energy/logs"

for node in "${nodes[@]}"; do
    for coll in "${collectives[@]}"; do
        for occl_path in "${occl_paths[@]}"; do
            if [[ "${coll}" == "ar" ]]; then
                algs=("${ar_algs[@]}")
            else
                algs=("${a2a_algs[@]}")
            fi
            for alg in "${algs[@]}"; do
                RUN_COLL_PY="$SCRIPT_DIR/run_coll.py"
                echo "===================================================="
                echo "Processing: Collective=$coll, Path=$occl_path, Alg=$alg, Nodes=$node"
                python3 "$RUN_COLL_PY" \
                    --pbs "$PBS_TEMPLATE" \
                    --exe "$EXE_BASE/${coll}_occl" \
                    --log "$LOG_BASE/" \
                    --env "$ENV_FILE" \
                    --path "${occl_path}" \
                    --coll "${coll}" \
                    --partition "${PARTITION}" \
                    --time "00:30:00" \
                    --nodes "${node}" \
                    --scaleout-alg "${alg}" 
            done
        done
    done
done