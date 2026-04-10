#!/bin/bash
#PBS -N a2a_1
#PBS -A  EnergyOpt_PhaseFreq
#PBS -l select=1
#PBS -l walltime=00:10:00
#PBS -l filesystems=home
#PBS -o /home/lcarpent/energy-workspace/Blink-GPU/energy/logs/pbs-out/a2a_n1.out
#PBS -e /home/lcarpent/energy-workspace/Blink-GPU/energy/logs/pbs-out/a2a_n1.err
#PBS -q debug
#PBS -l place=excl
echo "Executing single node ${COLLECTIVE} ..."  # Output: app_name
echo "Unique nodes allocated:"
sort -u $PBS_NODEFILE
source ${SET_ENV_PATH}

export  ZES_ENABLE_SYSMAN=1
export ONEAPI_DEVICE_SELECTOR=level_zero:gpu
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export CCL_LOG_LEVEL=info

NNODES=`wc -l < $PBS_NODEFILE`

NRANKS_PER_NODE=12
NDEPTH=1
NTHREADS=1

NTOTRANKS=$(( NNODES * NRANKS_PER_NODE ))
echo "NUM_OF_NODES= ${NNODES} TOTAL_NUM_RANKS= ${NTOTRANKS} RANKS_PER_NODE= ${NRANKS_PER_NODE} THREADS_PER_RANK= ${NTHREADS}"

MPI_ARG="-n ${NTOTRANKS} --ppn ${NRANKS_PER_NODE} --depth=${NDEPTH} --cpu-bind depth "


############## VTUNE ARGUMENTS ##############
# # Vtune suggest to use aps for MPI program analysis
# VTUNE_LOG_DIR="/home/lcarpent/energy-workspace/Blink-GPU/energy/logs/vtune/"
# # VTUNE_ARG="vtune -collect gpu-hotspots -r ${VTUNE_LOG_DIR}"
# VTUNE_ARG="vtune -collect gpu-offload -r ${VTUNE_LOG_DIR}"

# export CCL_ITT_LEVEL=1

# Note Vtune: is not able to detect MPI info and it suggest to use APS
############## APS ARGUMENTS ###############

if [ "$NNODES" -gt 1 ]; then
    mkdir -p ${CSV_LOG_DIR}/multi-node/${COLLECTIVE}/
    csv_file="${CSV_LOG_DIR}/multi-node/${COLLECTIVE}/${COLLECTIVE}_n${NNODES}_scaleup${SCALEUP_MODE}_scaleout${SCALEOUT_MODE}_mnic${NIC_MODE}.csv"
else
    mkdir -p ${CSV_LOG_DIR}/single-node/${COLLECTIVE}/
    csv_file="${CSV_LOG_DIR}/single-node/${COLLECTIVE}/${COLLECTIVE}_n${NNODES}_scaleup${SCALEUP_MODE}_scaleoutDefault_mnicDefault.csv"
fi

# COMMAND="mpiexec ${MPI_ARG} ${VTUNE_ARG} ${EXE_PATH} ${csv_file}"
COMMAND="mpiexec ${MPI_ARG}  ${EXE_PATH} ${csv_file}"

$COMMAND