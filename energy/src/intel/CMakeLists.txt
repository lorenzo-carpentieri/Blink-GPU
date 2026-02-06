find_package(MPI REQUIRED)
find_package(oneCCL REQUIRED)
set(OCCL_COLLECTIVES
    a2a_occl
    ar_occl
    send_recv_occl
)

foreach(target ${OCCL_COLLECTIVES})

    add_executable(${target}
        ${target}.cpp
    )

    target_include_directories(${target} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/utils
    )

    target_link_libraries(${target}
        PRIVATE
            MPI::MPI_CXX
            oneCCL
            power_profiler
            common_headers
    )

    target_compile_definitions(${target} PRIVATE USE_ONECCL)

    if(ENABLE_DEBUG_INFO)
        target_compile_definitions(${target} PRIVATE DEBUG)
    endif()

    install(TARGETS ${target}
            RUNTIME DESTINATION bin)

endforeach()