#include <mpi.h>
#include <cuda_runtime.h>
#include "utils/nccl_data_type.hpp"
#include <string>
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

template <typename T>
void print_prof_info(profiler::PowerProfiler& powerProf, size_t time_per_rank, size_t buff_size_byte, int chain_size, clog::Logger& csv_log, nvidia::utils::ncclContext& ctx, int run, std::string step_name) {
    prof_data_types::energy_t dev_energy_uj = powerProf.get_device_energy(); //device energy in uj for one collective run for the current rank
    double dev_energy_mj = static_cast<double>(dev_energy_uj) / 1000.0; 
    
    prof_data_types::energy_t host_energy_uj = powerProf.get_host_energy(); //host energy in uj for one collective run for the current rank
    double host_energy_mj = static_cast<double>(host_energy_uj) / 1000.0; 
    
    double time_ms = static_cast<double>(time_per_rank) / 1000.0; // time in ms for the current rank
    
    clog::Logger::ProfilingInfo<T> prof_info{time_ms, buff_size_byte, dev_energy_mj, host_energy_mj, ctx.global_rank, ctx.local_rank, ctx.global_rank_size, run, true, chain_size, "composite"};  
    prof_data_types::power_trace_t power_trace = powerProf.get_power_execution_data(); // get power trace data and store it internally in the power_prof object
    std::string power_trace_str = prof_data_types::power_trace_to_string(power_trace);
    clog::CsvField power_trace_field{"power_trace", power_trace_str};
    clog::CsvField allreduce_step= {"allreduce_step", step_name};

    csv_log.log_result<T>(prof_info, power_trace_field, allreduce_step); // log the profiling info and the power trace in the csv file
    return;
}

template<typename T>
void run(nvidia::utils::ncclContext& ctx){ 
    ncclDataType_t dtype = nccl_type_traits<T>::type; // define the mapping for T and nccl data type used in collectives

    cudaStream_t stream = std::move(*ctx.stream); // ccl stream
    ncclComm_t comm = std::move(*ctx.comm);
    clog::Logger csv_log = std::move(*ctx.logger);

    int rank = ctx.global_rank;
    int numGPUs = ctx.local_rank_size;

    

    // size_t buff_size_byte[] = {
    //     4,
    //     32, 64, 512, 4096, 32768, 262144,
    //     2097152, 16777216, 134217728, 1073741824
    // };
    size_t buff_size_byte[] = {
        // 4,
        // 32, 64, 512, 4096, 32768, 262144,
        // 2097152, 16777216, 134217728, 
        1073741824
    };

    const int num_iters = std::size(buff_size_byte);

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

            size_t reduce_time = 0, scatter_time = 0, allgather_time = 0;
            size_t reduce_time_per_rank = 0, allgather_time_per_rank = 0;
            chain_size = 0;
            // Profile the three different step of allreduce: the reduce, the scatter and the allgather.
            profiler::PowerProfiler powerProf_reduce(rank % numGPUs, 0, POWER_SAMPLING_RATE_MS);
            profiler::PowerProfiler powerProf_scatter(rank % numGPUs, 0, POWER_SAMPLING_RATE_MS);
            profiler::PowerProfiler powerProf_allgather(rank % numGPUs, 0, POWER_SAMPLING_RATE_MS);
            // Step 1: Reduce + Scatter
            std::cout << "STEP 1 -> ReduceScatter, " << "DATA SIZE -> " << buff_size_byte[i] << ", RUN -> "<< run << std::endl;
            size_t reduceScatter_el = count_el / ctx.global_rank_size; // number of elements to reduce+scatter for each rank
            powerProf_reduce.start();
            while (reduce_time < (TIME_TO_ACHIEVE_MS * 1000)) { // reduce_time in microseconds
                auto start_s = std::chrono::high_resolution_clock::now();
                ncclGroupStart();
                ncclReduceScatter(d_sendbuf, d_recvbuf, reduceScatter_el, dtype, ncclSum, comm, stream); 
                ncclGroupEnd();
                cudaStreamSynchronize(stream);
                auto end_s = std::chrono::high_resolution_clock::now();
                reduce_time_per_rank += std::chrono::duration_cast<std::chrono::microseconds>(end_s - start_s).count();
                reduce_time += std::chrono::duration_cast<std::chrono::microseconds>(end_s - start_s).count();
                MPI_Allreduce(MPI_IN_PLACE, &reduce_time, 1, MPI_LONG_LONG, MPI_MAX, MPI_COMM_WORLD);
                cudaMemset(d_sendbuf, rank, buff_size_byte[num_iters-1]);
                chain_size++;
            }
            powerProf_reduce.stop();
            std::string step_name_reduce = "reduceScatter";
            print_prof_info<T>(powerProf_reduce, reduce_time_per_rank, buff_size_byte[i], chain_size, csv_log, ctx, run, step_name_reduce);
            chain_size = 0; // reset the chain size for the next step
            std::cout << "STEP 2 -> AllGather, " << "DATA SIZE -> " << buff_size_byte[i] << ", RUN -> "<< run << std::endl;
            
            // Step 2: AllGather operation
            powerProf_allgather.start();
            int allgather_el = count_el / ctx.global_rank_size; // number of elements to allgather for each rank
            while (allgather_time < (TIME_TO_ACHIEVE_MS * 1000)) { // allgather_time in microseconds
                auto start_s = std::chrono::high_resolution_clock::now();
                ncclGroupStart();
                ncclAllGather(d_sendbuf, d_recvbuf, allgather_el, dtype, comm, stream); 
                ncclGroupEnd();
                cudaStreamSynchronize(stream);
                auto end_s = std::chrono::high_resolution_clock::now();
                allgather_time_per_rank += std::chrono::duration_cast<std::chrono::microseconds>(end_s - start_s).count();
                allgather_time += std::chrono::duration_cast<std::chrono::microseconds>(end_s - start_s).count();
                MPI_Allreduce(MPI_IN_PLACE, &allgather_time, 1, MPI_LONG_LONG, MPI_MAX, MPI_COMM_WORLD);
                chain_size++;
            }
            powerProf_allgather.stop();
            std::string step_name_allgather = "allgather";
            print_prof_info<T>(powerProf_allgather, allgather_time_per_rank, buff_size_byte[i], chain_size, csv_log, ctx, run, step_name_allgather);
            chain_size = 0; // reset the chain size for the next step
            


        }
        cudaMemcpy(d_recvbuf, h_recvbuf, buff_size_byte[num_iters-1] , cudaMemcpyHostToDevice);
        
    }


    // Scatter operation

    // All gather operation


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


    nvidia::utils::ncclContext ctx = nvidia::utils::init_nccl(csv_path, "ar_manual"); // initialize oneCCL and MPI
 
    // Run with different data type
    // run<uint8_t>(comm, rank, size, numGPUs, log_path, csv_path);
    // run<int>(comm, rank, size, numGPUs, log_path, csv_path);
    run<float>(ctx);
    // run<double>(comm, rank, size, numGPUs, log_path, csv_path);

    ncclCommDestroy(std::move(*ctx.comm));
    
    MPI_Finalize();

    return 0;
}



