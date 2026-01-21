#!/bin/bash
#PBS -N a2a_occl_multinode
#PBS -A  EnergyOpt_PhaseFreq
#PBS -l select=2:ncpus=2:ngpus=12
#PBS -l place=excl
#PBS -l walltime=00:10:00
#PBS -l filesystems=home
#PBS -o /home/lcarpent/energy-workspace/Blink-GPU/energy/logs/pbs-out/output_a2a_occl.txt
#PBS -e /home/lcarpent/energy-workspace/Blink-GPU/energy/logs/pbs-out/error_a2a_occl.txt
#PBS -q debug-scaling

echo "Executing a2a ..."  # Output: app_name
echo "Unique nodes allocated:"
sort -u $PBS_NODEFILE
source /home/lcarpent/energy-workspace/Blink-GPU/energy/scripts/aurora/env/set_env.sh

export  ZES_ENABLE_SYSMAN=1
export ONEAPI_DEVICE_SELECTOR=level_zero:gpu
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export CCL_LOG_LEVEL=error

# Set scaleup and scaleout mode for multi node.
export CCL_ALLTOALLV_MONOLITHIC_KERNEL=${SCALEUP_MODE} # SCALE_UP mode can be 0 (use copy engines) or 1 (use GPU compute kernel to transfer data across GPUs). 
export CCL_ALLTOALL_SCALEOUT=${ALG_NAME} # Not used in single node but set for consistency.

power_file="${POWER_LOG_DIR}/multi-node/occl/${COLLECTIVE}_${SCALEUP_MODE}_${ALG_NAME}_pow.csv"
csv_file="${CSV_LOG_DIR}/multi-node/occl/${COLLECTIVE}_${SCALEUP_MODE}_${ALG_NAME}.csv"
mpiexec -n 24 -ppn 12  ${EXE_PATH} ${power_file} ${csv_file}