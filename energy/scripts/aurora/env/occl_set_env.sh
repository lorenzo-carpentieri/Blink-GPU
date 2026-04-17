module reset 
module load cmake
module load frameworks/2025.3.1
# module load mpich/opt/5.0.0.aurora_test.3c70a61
# module load oneapi/release/2025.2.0  
# module load mpich
# source /opt/aurora/25.190.0/oneapi/setvars.sh --force

# module load mpich/opt/5.0.0.aurora_test.06f012a
# source /opt/aurora/26.26.0/oneapi/setvars.sh
# module load hdf5/1.14.6
# export PATH=/home/lcarpent/install/mpich/bin:$PATH
# export LD_LIBRARY_PATH=/home/lcarpent/install/mpich/lib:$LD_LIBRARY_PATH
# export MPICH_ROOT=/home/lcarpent/install/mpich/
# export MPICC=/home/lcarpent/install/mpich/bin/mpicc 
# export MPICXX=/home/lcarpent/install/mpich/bin/mpicxx
# export MPI_CXX=/home/lcarpent/install/mpich/bin/mpicxx

# export MPI_HOME=/home/lcarpent/install/mpich/
# export MPI_ROOT=/home/lcarpent/install/mpich/

module list
export CCL_PROCESS_LAUNCHER=pmix
export CCL_ATL_TRANSPORT=mpi
export FI_MR_CACHE_MONITOR=userfaultfd
export CCL_KVS_MODE=mpi
export PALS_PMI=pmix
export CCL_CONFIGURATION=cpu_gpu_dpcpp
export CCL_KVS_CONNECTION_TIMEOUT=600 

export CCL_ZE_CACHE_OPEN_IPC_HANDLES_THRESHOLD=1024
export CCL_KVS_USE_MPI_RANKS=1

# export CCL_WORKER_MEM_AFFINITY=auto
# export FI_PROVIDER=tcp
export MPI_PROVIDER=$FI_PROVIDER
unset MPIR_CVAR_CH4_POSIX_COLL_SELECTION_TUNING_JSON_FILE
unset MPIR_CVAR_CH4_COLL_SELECTION_TUNING_JSON_FILE
unset MPIR_CVAR_COLL_SELECTION_TUNING_JSON_FILE

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
# export CPU_BIND_SCHEME="list:4-7:8-11:12-15:16-19:20-23:24-27:56-59:60-63:64-67:68-71:72-75:76-79"
# export CCL_WORKER_AFFINITY="42,43,44,45,46,47,94,95,96,97,98,99"
# export CCL_ATL_HMEM=1
# export CCL_USE_HMEM=1

export MPIR_CVAR_ENABLE_GPU=1
export MPICH_GPU_SUPPORT_ENABLED=1
# export CCL_SYCL_ENABLE_DIRECT_GPU_RDMA=1
# export CCL_SYCL_ENABLE_PIPELINE_GPU_RDMA=1
# export CCL_SYCL_PIPELINE_GPU_RDMA=1

export CCL_MNIC=local
export CCL_MNIC_COUNT=8
