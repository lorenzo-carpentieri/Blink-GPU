
PROJ_DIR=$1
BUILD_DIR=${PROJ_DIR}/build
SOURCE_DIR=${PROJ_DIR}

source ${SOURCE_DIR}/scripts/enea/env/occl_set_env.sh
cmake -S ${SOURCE_DIR} -B ${BUILD_DIR} \
    -DENABLE_ONECCL=ON \
    -DCMAKE_CXX_COMPILER=icpx \
    -DCMAKE_CXX_FLAGS="-fsycl -fsycl-targets=intel_gpu_pvc" \
    -DCMAKE_BUILD_TYPE=Release \
    -DUSE_NVML=OFF \
    -DUSE_LEVEL_ZERO=ON \
    -DUSE_RAPL=OFF \
    -DCMAKE_INSTALL_PREFIX=${BUILD_DIR} 

cmake --build ${BUILD_DIR} -j --target install