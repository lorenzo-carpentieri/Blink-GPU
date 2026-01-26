#!/bin/bash
#PBS -N send_recv_mpich_multinode
#PBS -A  EnergyOpt_PhaseFreq
#PBS -l select=2:ncpus=2:ngpus=2
#PBS -l walltime=00:10:00
#PBS -l place=scatter:excl
#PBS -l filesystems=home
#PBS -o /home/lcarpent/energy-workspace/Blink-GPU/energy/logs/pbs-out/output_send_recv_mpi.txt
#PBS -e /home/lcarpent/energy-workspace/Blink-GPU/energy/logs/pbs-out/error_send_recv_mpi.txt
#PBS -q prod

echo "Executing Send Recv ..."  # Output: app_name
echo "Unique nodes allocated:"
sort -u $PBS_NODEFILE
source /home/lcarpent/energy-workspace/Blink-GPU/energy/scripts/aurora/env/set_mpi_env.sh

export  ZES_ENABLE_SYSMAN=1
export ONEAPI_DEVICE_SELECTOR=level_zero:gpu
export ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE


# MPI example w/ 12 MPI ranks per node each with access to single GPU tile
NNODES=`wc -l < $PBS_NODEFILE`

NRANKS_PER_NODE=2 # use all gpus onthe node
NDEPTH=1
NTHREADS=1

NTOTRANKS=$(( NNODES * NRANKS_PER_NODE ))

echo "NUM_OF_NODES= ${NNODES} TOTAL_NUM_RANKS= ${NTOTRANKS} RANKS_PER_NODE= ${NRANKS_PER_NODE} THREADS_PER_RANK= ${NTHREADS}"


MPI_ARG="-n ${NTOTRANKS} --ppn ${NRANKS_PER_NODE} --depth=${NDEPTH} --cpu-bind depth ${CPU_BIND_SCHEME} ${GPU_BIND_SCHEME} ${MEM_BIND_SCHEME}"

EXE_PATH="/home/lcarpent/energy-workspace/Blink-GPU/energy/install/bin/send_recv_mpi"
COMMAND="mpiexec ${MPI_ARG} ${EXE_PATH}"

$COMMAND