module restore
module load oneapi
module load mpich/opt/5.0.0.aurora_test.06f012a # Branch of MPICH with RDMA support in the 4.3 module of mpich the support for RDMA is not working
module load cray-pals/1.8.0 cray-libpals/1.8.0 libfabric/1.22.0
# source /opt/aurora/25.190.0/oneapi/ccl/latest/env/vars.sh --ccl-bundled-mpi=no
export CCL_SYCL_ENABLE_DIRECT_GPU_RDMA=1
export CCL_SYCL_ENABLE_PIPELINE_GPU_RDMA=1

# source /opt/aurora/25.190.0/oneapi/setvars.sh
export MPIR_CVAR_ENABLE_GPU=1
export CCL_PROCESS_LAUNCHER=pmix  
export CCL_ATL_TRANSPORT=mpi
export CCL_KVS_MODE=mpi
export CCL_CONFIGURATION_PATH=""
export CCL_CONFIGURATION=cpu_gpu_dpcpp
export CCL_KVS_CONNECTION_TIMEOUT=600 
export CCL_ZE_CACHE_OPEN_IPC_HANDLES_THRESHOLD=1024
export CCL_KVS_USE_MPI_RANKS=1
export MPI_PROVIDER=$FI_PROVIDER
echo "MPI provider: ${MPI_PROVIDER}" 
unset MPIR_CVAR_CH4_POSIX_COLL_SELECTION_TUNING_JSON_FILE
unset MPIR_CVAR_CH4_COLL_SELECTION_TUNING_JSON_FILE
unset MPIR_CVAR_COLL_SELECTION_TUNING_JSON_FILE

##### Set NIC count to only for Send/Recv test with two rank #####
export CCL_MNIC=local
export CCL_MNIC_COUNT=1
export CCL_MNIC_NAME=cxi0
##### mpich variables ####
export MPICH_GPU_SUPPORT_ENABLED=1
export PALS_PMI=pmix

ulimit -c unlimited
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
export CCL_ATL_SYNC_COLL=1 # to avoid potential hang at large scale
export CCL_OP_SYNC=1 # to avoid potential hang at large scale