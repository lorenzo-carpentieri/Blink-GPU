
PROJ_DIR=$1
BUILD_DIR=${PROJ_DIR}/build
SOURCE_DIR=${PROJ_DIR}

source ${SOURCE_DIR}/scripts/leonardo/env/set_nccl_env.sh
cmake -S ${SOURCE_DIR} -B ${BUILD_DIR} \
    -DENABLE_NCCL=ON \
    -DCMAKE_BUILD_TYPE=Release \
    -DUSE_NVML=ON \
    -DUSE_RAPL=ON \
    -DNCCL_INCLUDE_DIR=${NCCL_INCLUDE} \
    -DNCCL_LIB_DIR=${NCCL_LIB} \
    -DCMAKE_CUDA_ARCHITECTURES=80 \
    -DNCCL_VERSION=2.22.3 \
    -DCMAKE_INSTALL_PREFIX=${BUILD_DIR}

cmake --build ${BUILD_DIR} -j --target install