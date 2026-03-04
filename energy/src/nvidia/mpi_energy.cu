#include <mpi.h>
#include <cuda_runtime.h>
#include "utils/nccl_data_type.hpp"
#include <chrono>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include "../../energy-profiler/include/profiler/power_profiler.hpp"
#include "./utils/nccl_ctx.hpp"

#define MAX_RUN 3
#define WARM_UP_RUN 1
#define TIME_TO_ACHIEVE_MS 5000
#define POWER_SAMPLING_RATE_MS 40
#define MAX_BUF 100

namespace prof_data_types = profiler::data_types; // define profiler::data_types namespace abbreviation
namespace clog = common::logger;

template<typename T>
void run(nvidia::utils::ncclContext& ctx){ 

    int N=1024*1024*1024; // number of elements to send, we want to send 100MB of data
    int M=1024*1024; // number of elements to send in the warm up phase, we want to send 1MB of data

    ncclDataType_t dtype = nccl_type_traits<T>::type; // define the mapping for T and nccl data type used in collectives

    cudaStream_t stream = std::move(*ctx.stream); // ccl stream
    ncclComm_t comm = std::move(*ctx.comm);
    clog::Logger csv_log = std::move(*ctx.logger);

    int rank = ctx.global_rank;
    int numGPUs = ctx.local_rank_size;

    if (rank >= 2){
        throw std::runtime_error("This test is designed to run with 2 ranks but more ranks were detected. Please run with 2 ranks.");
        return; 
    }

    profiler::PowerProfiler powerProf(rank % numGPUs, 0, POWER_SAMPLING_RATE_MS);
    powerProf.start();
    float a = 1.0f, b = 2.0f, c = 3.0f;
    if(rank==0){
        for(int i = 0; i < N; ++i) {
            c = a * b + c;
        }
    }else{
        for(int i = 0; i < M; ++i) {
            c = a * b + c;
        }
        
    }
    
    powerProf.stop();
    
    std::cout << "Rank " << rank << " - Host Energy (uj): " << powerProf.get_host_energy() << std::endl;
    std::cout << "Rank: " << rank << " Results c: " << c << std::endl;

}

int main(int argc, char *argv[]) {

    std::string csv_path;
    
    if (argc != 2)
        return -1;
    else{   
        csv_path = argv[1];
    }


    nvidia::utils::ncclContext ctx = nvidia::utils::init_nccl(csv_path, "cpu_energy"); // initialize oneCCL and MPI
 
    run<float>(ctx);

    ncclCommDestroy(std::move(*ctx.comm));
    
    MPI_Finalize();

    return 0;
}



