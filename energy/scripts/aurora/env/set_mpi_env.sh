module restore
module load oneapi
module load mpich/opt/5.0.0.aurora_test.06f012a
module load cray-pals/1.8.0 cray-libpals/1.8.0 libfabric/1.22.0
# source /lus/flare/projects/catalyst/world_shared/mpich/setup.sh
# fi_info -p cxi
export MPICH_GPU_SUPPORT_ENABLED=1
export PALS_PMI=pmix

# export CPU_BIND_SCHEME="--cpu-bind=list:1:2"
export CPU_BIND_SCHEME=""
# export GPU_BIND_SCHEME="--gpu-bind=list:0:1"
export GPU_BIND_SCHEME=""
export MEM_BIND_SCHEME=""
# export MEM_BIND_SCHEME="--mem-bind=list:2:2"

########## NIC MPICH settings ##########
export MPIR_CVAR_CH4_OFI_ENABLE_MULTI_NIC_STRIPING=0 # Disable multi-NIC striping for more predictable performance on single NIC systems
export MPICH_OFI_NIC_MAPPING="0:0-1" # Map the first NIC (0) to all local ranks (0-1) on the node

export FI_MR_ZE_CACHE_MONITOR_ENABLED=0
export FI_MR_CACHE_MONITOR=disabled
export FI_CXI_RX_MATCH_MODE=hybrid
export FI_CXI_OFLOW_BUF_SIZE=8388608
export FI_CXI_DEFAULT_CQ_SIZE=1048576
export FI_CXI_CQ_FILL_PERCENT=30
export INTELGT_AUTO_ATTACH_DISABLE=1
export PALS_PING_PERIOD=240
export PALS_RPC_TIMEOUT=240
export MPIR_CVAR_GATHERV_INTER_SSEND_MIN_PROCS=-1 # to solve the sync send issue in Horovod seg fault
export MPI_PROVIDER=$FI_PROVIDER
echo "MPI provider: ${MPI_PROVIDER}" 
unset MPIR_CVAR_CH4_POSIX_COLL_SELECTION_TUNING_JSON_FILE
unset MPIR_CVAR_CH4_COLL_SELECTION_TUNING_JSON_FILE
unset MPIR_CVAR_COLL_SELECTION_TUNING_JSON_FILE
export MPIR_CVAR_ENABLE_GPU=1


# debug MPICH
export MPICH_OFI_VERBOSE=1
# export FI_LOG_LEVEL=debug
export MPICH_OFI_NIC_VERBOSE=1

export MPIR_CVAR_CH4_OFI_MAX_NICS=1
export MPIR_CVAR_OFI_USE_MIN_NICS=1