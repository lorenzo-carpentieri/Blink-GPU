cmake_minimum_required(VERSION 3.20)
project(collective-profiler LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(ENABLE_DEBUG_INFO OFF CACHE BOOL "Enable debug information")

# ------------------------------------------------------------
# Backend selection
# ------------------------------------------------------------
option(ENABLE_NCCL  "Build NVIDIA NCCL backend"  OFF)
option(ENABLE_RCCL  "Build AMD RCCL backend"     OFF)
option(ENABLE_ONECCL "Build Intel oneCCL backend" OFF)
option(ENABLE_MPI "Build MPI backend" OFF)


set(BACKEND_COUNT 0)

if (ENABLE_NCCL)
    math(EXPR BACKEND_COUNT "${BACKEND_COUNT} + 1")
endif()

if (ENABLE_RCCL)
    math(EXPR BACKEND_COUNT "${BACKEND_COUNT} + 1")
endif()


if (ENABLE_ONECCL)
    math(EXPR BACKEND_COUNT "${BACKEND_COUNT} + 1")
endif()

if (ENABLE_MPI)
    math(EXPR BACKEND_COUNT "${BACKEND_COUNT} + 1")
endif()

if (NOT BACKEND_COUNT EQUAL 1)
    message(FATAL_ERROR
        "Exactly one backend must be enabled:
         -DENABLE_NCCL=ON
         -DENABLE_RCCL=ON
         -DENABLE_ONECCL=ON")
endif()
# ------------------------------------------------------------
# Subprojects
# ------------------------------------------------------------
add_subdirectory(energy-profiler)
add_subdirectory(src)

# ------------------------------------------------------------
# Header-only common utilities
# ------------------------------------------------------------
add_library(common_headers INTERFACE)

target_include_directories(common_headers INTERFACE
    ${PROJECT_SOURCE_DIR}/include
)


if (ENABLE_ONECCL)
    target_compile_definitions(common_headers INTERFACE "USE_ONECCL")
endif()


if( ENABLE_DEBUG_INFO )
    message(STATUS "Debug information enabled")
    target_compile_definitions(common_headers INTERFACE "DEBUG")
endif()
