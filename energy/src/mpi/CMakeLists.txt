find_package(MPI REQUIRED)

set(MPI_COLLECTIVES
    send_recv_mpi
)

foreach(target ${MPI_COLLECTIVES})

    add_executable(${target}
        ${target}.cpp
    )

    target_include_directories(${target} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/utils
    )

    target_link_libraries(${target}
        PRIVATE
            MPI::MPI_CXX
            common_headers
    )


    if(ENABLE_DEBUG_INFO)
        target_compile_definitions(${target} PRIVATE DEBUG)
    endif()

    install(TARGETS ${target}
            RUNTIME DESTINATION bin)

endforeach()