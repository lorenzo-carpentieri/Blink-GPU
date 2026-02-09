
set(NCCL_COLLECTIVES
    a2a_nccl
    ar_nccl
)


list(APPEND CMAKE_MODULE_PATH
     "${CMAKE_CURRENT_SOURCE_DIR}/../../cmake")

find_package(MPI REQUIRED)
find_package(NCCL REQUIRED)

set(CMAKE_CUDA_ARCHITECTURES "" CACHE STRING
    "CUDA architectures to build for. Example: 80;86 or 'native'"
)

if(CMAKE_CUDA_ARCHITECTURES STREQUAL "")
    message(FATAL_ERROR
        "You must set CMAKE_CUDA_ARCHITECTURES.\n"
        "For example: -DCMAKE_CUDA_ARCHITECTURES=80 or -DCMAKE_CUDA_ARCHITECTURES=80;86"
    )
endif()

foreach(target ${NCCL_COLLECTIVES})

    add_executable(${target}
        ${target}.cu
    )

    target_include_directories(${target} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/utils
    )

    target_link_libraries(${target}
        PRIVATE
            MPI::MPI_CXX
            nccl
            power_profiler
            common_headers
    )

    target_compile_definitions(${target} PRIVATE USE_NCCL)

    if(ENABLE_DEBUG_INFO)
        target_compile_definitions(${target} PRIVATE DEBUG)
    endif()

    install(TARGETS ${target}
            RUNTIME DESTINATION bin)

endforeach()