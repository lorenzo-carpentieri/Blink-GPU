#include <mpi.h>
#include <cuda_runtime.h>
#include "utils/nccl_data_type.hpp"
#include <chrono>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include "../../energy-profiler/include/profiler/power_profiler.hpp"
#include "./utils/nccl_ctx.hpp"

#define MAX_RUN 5
#define WARM_UP_RUN 1
#define TIME_TO_ACHIEVE_MS 10000
#define POWER_SAMPLING_RATE_MS 20
#define MAX_BUF 100
#define MESSAGE_SIZE_FACTOR 4

namespace prof_data_types = profiler::data_types; // define profiler::data_types namespace abbreviation
namespace clog = common::logger;

template<typename T>
void run(nvidia::utils::ncclContext& ctx){ 
    ncclDataType_t dtype = nccl_type_traits<T>::type; // define the mapping for T and nccl data type used in collectives

    cudaStream_t stream = std::move(*ctx.stream); // ccl stream
    ncclComm_t comm = std::move(*ctx.comm);
    clog::Logger csv_log = std::move(*ctx.logger);

    int rank = ctx.global_rank;
    int numGPUs = ctx.local_rank_size;

    

    constexpr size_t ONE_GB = 1024 * 1024 * 1024;
    size_t *buff_size_byte = (size_t *)malloc(sizeof(size_t) * MAX_BUF);
    size_t num_elements=1;

    int i=0;
    while(num_elements * sizeof(T) <= ONE_GB ){
        buff_size_byte[i] = num_elements * sizeof(T);
        num_elements *= MESSAGE_SIZE_FACTOR;
        i++;
    }

    const int num_iters = i;
    T *d_sendbuf, *d_recvbuf;
    cudaMalloc((void **)&d_sendbuf, buff_size_byte[num_iters - 1]);
    cudaMalloc((void **)&d_recvbuf, buff_size_byte[num_iters - 1]); // each rank sendbuff size bytes

    T *h_sendbuf = (T *)malloc(buff_size_byte[num_iters - 1]);
    T *h_recvbuf = (T *)malloc(buff_size_byte[num_iters - 1]);

    cudaStreamCreate(&stream);

    for (int i = 0; i < WARM_UP_RUN; i++) {
        size_t count_el = buff_size_byte[num_iters-1] / sizeof(T);
        
        cudaMemcpy(h_sendbuf, d_sendbuf, buff_size_byte[num_iters-1], cudaMemcpyDeviceToHost);
        auto start = std::chrono::high_resolution_clock::now();
        ncclGroupStart();
        ncclAllReduce(d_sendbuf, d_recvbuf, count_el, dtype, ncclSum, comm, stream);
        ncclGroupEnd();

        cudaStreamSynchronize(stream);
        MPI_Barrier(MPI_COMM_WORLD);
        auto end = std::chrono::high_resolution_clock::now();
        auto time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        cudaMemcpy(d_recvbuf, h_recvbuf, buff_size_byte[num_iters-1], cudaMemcpyHostToDevice);
    }


    cudaMemset(d_sendbuf, rank, buff_size_byte[num_iters-1]);
    auto mem_cpy_t_start = std::chrono::high_resolution_clock::now();
    cudaMemcpy(h_sendbuf, d_sendbuf, buff_size_byte[num_iters-1], cudaMemcpyDeviceToHost);
    auto mem_cpy_t_end = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < num_iters; i++) {
        size_t count_el = buff_size_byte[i] / sizeof(T);
        int chain_size = 0;
        for (int run = 0; run < MAX_RUN; run++) {
            size_t ar_time = 0;
            size_t ar_time_per_rank = 0;
            chain_size = 0;
            profiler::PowerProfiler powerProf(rank % numGPUs, POWER_SAMPLING_RATE_MS);
            powerProf.start();
            while (ar_time < (TIME_TO_ACHIEVE_MS * 1000)) { // ar_time in microseconds
                auto start_s = std::chrono::high_resolution_clock::now();
                ncclGroupStart();
                ncclAllReduce(d_sendbuf, d_recvbuf, count_el, dtype, ncclSum, comm, stream);
                ncclGroupEnd();
                cudaStreamSynchronize(stream);
                auto end_s = std::chrono::high_resolution_clock::now();
                ar_time_per_rank += std::chrono::duration_cast<std::chrono::microseconds>(end_s - start_s).count();
                ar_time += std::chrono::duration_cast<std::chrono::microseconds>(end_s - start_s).count();

                MPI_Allreduce(MPI_IN_PLACE, &ar_time, 1, MPI_LONG_LONG, MPI_MAX, MPI_COMM_WORLD);
                chain_size++;
            }
            powerProf.stop();
            prof_data_types::energy_t dev_energy_uj = powerProf.get_device_energy(); //device energy in uj for one collective run for the current rank
            double time_ms = static_cast<double>(ar_time_per_rank) / 1000.0; // time in ms for the current rank
            double dev_energy_mj = static_cast<double>(dev_energy_uj) / 1000.0; 
            
            // host energy is 0 now
            clog::Logger::ProfilingInfo<T> prof_info{time_ms, buff_size_byte[i], dev_energy_mj, 0.0, ctx.global_rank, ctx.local_rank, ctx.global_rank_size, run, true, chain_size, "composite"};  
            prof_data_types::power_trace_t power_trace = powerProf.get_power_execution_data(); // get power trace data and store it internally in the power_prof object
            std::string power_trace_str = prof_data_types::power_trace_to_string(power_trace);
            clog::CsvField power_trace_field{"power_trace", power_trace_str};
            csv_log.log_result<T>(prof_info, power_trace_field); 
            

        }
    }
    cudaMemcpy(d_recvbuf, h_recvbuf, buff_size_byte[num_iters-1] , cudaMemcpyHostToDevice);


    cudaStreamDestroy(stream);
    cudaFree(d_sendbuf);
    cudaFree(d_recvbuf);
    free(h_sendbuf);
    free(h_recvbuf);


}

int main(int argc, char *argv[]) {

    std::string csv_path;
    
    if (argc != 2)
        return -1;
    else{   
        csv_path = argv[1];
    }


    nvidia::utils::ncclContext ctx = nvidia::utils::init_nccl(csv_path, "ar"); // initialize oneCCL and MPI
 
    // Run with different data type
    // run<uint8_t>(comm, rank, size, numGPUs, log_path, csv_path);
    // run<int>(comm, rank, size, numGPUs, log_path, csv_path);
    run<float>(ctx);
    // run<double>(comm, rank, size, numGPUs, log_path, csv_path);

    ncclCommDestroy(std::move(*ctx.comm));
    
    MPI_Finalize();

    return 0;
}



