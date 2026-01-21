#!/bin/bash
#PBS -N a2a_occl
#PBS -A  EnergyOpt_PhaseFreq
#PBS -l select=1:ncpus=1:ngpus=3
#PBS -l walltime=00:10:00
#PBS -l filesystems=home
#PBS -o /home/lcarpent/energy-workspace/Blink-GPU/energy/logs/pbs-out/output_a2a_occl.txt
#PBS -e /home/lcarpent/energy-workspace/Blink-GPU/energy/logs/pbs-out/error_a2a_occl.txt
#PBS -q prod

echo "Executing a2a ..."  # Output: app_name
echo "Unique nodes allocated:"
sort -u $PBS_NODEFILE
source /home/lcarpent/energy-workspace/Blink-GPU/energy/scripts/aurora/env/set_env.sh

export  ZES_ENABLE_SYSMAN=1
export ONEAPI_DEVICE_SELECTOR=level_zero:gpu
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export CCL_LOG_LEVEL=error

# Set scaleup and scaleout mode for single node. Scaleout mode is not used in single node.
export CCL_ALLTOALLV_MONOLITHIC_KERNEL=${SCALEUP_MODE} # SCALE_UP mode can be 0 (use copy engines) or 1 (use GPU compute kernel to transfer data across GPUs). 
# export CCL_ALLTOALL_SCALEOUT="none" # Not used in single node but set for consistency.

power_file="${POWER_LOG_DIR}/single-node/occl/${COLLECTIVE}_${SCALEUP_MODE}_${ALG_NAME}_pow.csv"
csv_file="${CSV_LOG_DIR}/single-node/occl/${COLLECTIVE}_${SCALEUP_MODE}_${ALG_NAME}.csv"
mpiexec -n 3 -ppn 3  ${EXE_PATH} ${power_file} ${csv_file}