module --force purge
# module load LUMI/25.03
# module load partition/G
# module load PrgEnv-gnu
# module load craype-accel-amd-gfx90a
# module load rocm/6.3.4
# export EBU_USER_PREFIX=/project/project_465002469/EasyBuild
# module load EasyBuild-user
# module load aws-ofi-rccl/17d41cb-cpeGNU-25.03
# module load gcc-native


module load LUMI/25.03 partition/G
# module load PrgEnv-cray
# module load craype-accel-amd-gfx90a
# module load rocm
module load craype-accel-amd-gfx90a   # for MI250x GPUs
module load PrgEnv-gnu                 # GNU toolchain
module load cray-mpich                  # MPICH with libfabric support
module load rocm/6.3.4
export MPICH_GPU_SUPPORT_ENABLED=1     # sometimes also required

export MPICH_DBG_OUTPUT=stdout   # or file
export MPICH_DBG_CLASS=ALL
export MPICH_DBG_LEVEL=VERBOSE
export FI_LOG_LEVEL=debug
export FI_LOG_PROV=cxi