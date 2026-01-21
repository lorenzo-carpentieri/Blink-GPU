#!/bin/bash
#PBS -N a2a_occl
#PBS -A  EnergyOpt_PhaseFreq
#PBS -l select=1:ncpus=1:ngpus=3
#PBS -l walltime=00:20:00
#PBS -l filesystems=home
#PBS -o /home/lcarpent/energy-workspace/Blink-GPU/energy/logs/pbs-out/output_a2a_occl.txt
#PBS -e /home/lcarpent/energy-workspace/Blink-GPU/energy/logs/pbs-out/error_a2a_occl.txt
#PBS -q debug

echo "Executing a2a ..."  # Output: app_name
echo "Unique nodes allocated:"
sort -u $PBS_NODEFILE
source /home/lcarpent/energy-workspace/Blink-GPU/energy/scripts/aurora/env/set_env.sh

export  ZES_ENABLE_SYSMAN=1
export ONEAPI_DEVICE_SELECTOR=level_zero:gpu
export ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE
export CCL_LOG_LEVEL=error
# export CCL_LOCAL_RANK=3

exe="/home/lcarpent/energy-workspace/Blink-GPU/energy/install/bin/a2a_occl"
power_log_path="/home/lcarpent/energy-workspace/Blink-GPU/energy/logs/power/"
csv_path="/home/lcarpent/energy-workspace/Blink-GPU/energy/logs/perf/a2a.csv"
mpiexec -n 3 -ppn 3  ${exe} ${power_log_path}  ${csv_path}