
set(RCCL_COLLECTIVES
    a2a_rccl
    ar_rccl
)

find_path(RCCL_INCLUDE_DIR
    NAMES rccl.h
    PATHS
        /opt/rocm/include
        /opt/rocm/rccl/include
        /opt/rocm/include/rccl/
    REQUIRED
)

list(APPEND CMAKE_MODULE_PATH
     "${CMAKE_CURRENT_SOURCE_DIR}/../../cmake")

find_package(MPI REQUIRED)
find_package(rccl REQUIRED)

set(CMAKE_HIP_ARCHITECTURES "" CACHE STRING
    "HIP architectures to build for. Example: gfx908;86 or 'native'"
)

if(CMAKE_HIP_ARCHITECTURES STREQUAL "")
    message(FATAL_ERROR
        "You must set CMAKE_HIP_ARCHITECTURES.\n"
    )
endif()

foreach(target ${RCCL_COLLECTIVES})

    add_executable(${target}
        ${target}.cpp
    )
    set_source_files_properties(
        ${target}.cpp
        PROPERTIES LANGUAGE HIP
    )
    target_include_directories(${target} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/utils
        ${RCCL_INCLUDE_DIR}
    )

    target_link_libraries(${target}
        PRIVATE
            MPI::MPI_CXX
            rccl
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