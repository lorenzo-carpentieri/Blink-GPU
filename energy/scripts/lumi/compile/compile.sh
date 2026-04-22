
PROJ_DIR=$1
BUILD_DIR=${PROJ_DIR}/build
SOURCE_DIR=${PROJ_DIR}

source ${SOURCE_DIR}/scripts/lumi/env/rccl_env.sh
cmake -S ${SOURCE_DIR} -B ${BUILD_DIR} \
    -DCMAKE_CXX_COMPILER=g++ \
    -DENABLE_RCCL=ON \
    -DCMAKE_BUILD_TYPE=Release \
    -DAMDGPU_TARGETS=gfx90a \
    -DUSE_ROCM=ON \
    -DUSE_RAPL=OFF \
    -DCMAKE_INSTALL_PREFIX=${BUILD_DIR} \
    -DCMAKE_CXX_STANDARD=17 

cmake --build ${BUILD_DIR} -j --target install