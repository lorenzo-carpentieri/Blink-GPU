module purge
module load LUMI/25.03 # LUMI/23.09
module load PrgEnv-cray
module load craype-accel-amd-gfx90a
module load rocm
module load gcc-native
export CXX=g++
export CC=gcc
