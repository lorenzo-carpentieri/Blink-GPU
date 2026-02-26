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

export EBU_USER_PREFIX=/project/project_465002469/EasyBuild
module load LUMI/25.03 partition/G
module load aws-ofi-rccl/
module load rocm/6.3.4

export NCCL_NET_GDR_LEVEL=3
export FI_CXI_ATS=0
export NCCL_BUFFSIZE=33554432

export CXX=/opt/rocm-6.3.4/lib/llvm/bin/clang++
export CC=gcc
export LOG_INFO=info