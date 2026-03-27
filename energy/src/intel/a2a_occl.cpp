
// You can use template paramter for specilize the funcition that run the collective with different data types
//TODO: Test multi nodo per vedere se cambia qualcosa dal punto di vista delle performance e dell'energy
#include <mpi.h>
#include "../utils/occl_utils.hpp"
#include "../../include/utils/ccl_data_types.hpp"  
#include <oneapi/ccl.hpp> // oneCCL main header

#include <chrono>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include "../../energy-profiler/include/profiler/power_profiler.hpp"
#include <vector>

#define MAX_RUN 5
#define WARM_UP_RUN 5
#define TIME_TO_ACHIEVE_MS 5000
#define POWER_SAMPLING_RATE_MS 40
#define MAX_BUF 100

namespace clog = common::logger;
namespace data_types = common::utils::data_types; // define data_types namespace abbreviation
namespace prof_data_types = profiler::data_types; // define profiler::data_types namespace abbreviation
using Comm = data_types::CommWrapper<>::type;  // automaticall select ccl::communicator at compile time accroding to the defined macro USE_ONECCL

template<typename T>
void run(intel::utils::OneCCLContext& ctx){ 

    sycl::queue q = ctx.q; // sycl queue

   
    ccl::stream stream =std::move(*ctx.stream); // ccl stream
    Comm comm = std::move(*ctx.comm);
    clog::Logger csv_log = std::move(*ctx.logger);

    int rank = ctx.global_rank;
    int num_ranks = ctx.global_rank_size;

    int numGPUs = ctx.local_rank_size;
    
    size_t buff_size_byte[] = {
        4,
        32, 64, 512, 4096, 32768, 262144,
        2097152, 16777216, 134217728, 1073741824
    };
  
    const int num_iters = std::size(buff_size_byte);

    T* d_sendbuf = sycl::malloc_device<T>( buff_size_byte[num_iters - 1], q);
    T* d_recvbuf = sycl::malloc_device<T>(buff_size_byte[num_iters - 1] * num_ranks, q);

    /* init device side buffer */
    q.memset(d_sendbuf, rank, buff_size_byte[num_iters - 1]);
    q.memset(d_recvbuf, -1, buff_size_byte[num_iters - 1] * num_ranks);
     

    // create dependencies vector: can be used in the collective to ensure that consecutive call to the collective are exectude in order
    std::vector<ccl::event> deps;
    auto attr = ccl::create_operation_attr<ccl::alltoall_attr>();


    // allocate host pointer
    T *h_sendbuf = (T *)malloc(buff_size_byte[num_iters - 1]);
    T *h_recvbuf = (T *)malloc(buff_size_byte[num_iters - 1] * num_ranks);

    if (rank == 0) {
        std::cout<< "[INFO] Warm up run for alltoall" << std::endl;    
    }
    
    for (int i = 0; i < WARM_UP_RUN; i++) {
        // cpy dev -> host
        q.memcpy(h_sendbuf, d_sendbuf, buff_size_byte[num_iters-1]).wait();       

        auto start = std::chrono::high_resolution_clock::now();
        ccl::alltoall(d_sendbuf, d_recvbuf, buff_size_byte[num_iters-1] / sizeof(T), ccl::datatype::float32, comm, stream, attr, deps).wait();

        auto end = std::chrono::high_resolution_clock::now();
        auto time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        // copy dev -> host
        q.memcpy(h_recvbuf, d_recvbuf, buff_size_byte[num_iters-1]*num_ranks).wait();       
    }

    for (int i = 0; i < num_iters; i++) {
        int count=buff_size_byte[i] / sizeof(T); // total num of elemenent per buffer
        int chain_size = 0; //  num of times that a collective is executed 
        for (int run = 0; run < MAX_RUN; run++) {
            if (rank == 0) {
                std::cout<< "[INFO] Run " << run << " for alltoall with message size "<< buff_size_byte[i] << " B" << std::endl;    
            }
            size_t a2a_time = 0;
            size_t a2a_time_per_rank = 0;
            chain_size = 0;
            
            profiler::PowerProfiler powerProf(rank % numGPUs,0, POWER_SAMPLING_RATE_MS);
            powerProf.start();
            while (a2a_time < (TIME_TO_ACHIEVE_MS * 1000)) {
                auto start_s = std::chrono::high_resolution_clock::now();
                if(buff_size_byte[i] <= 256){ // reduce the number of times that MPIAll reduce is called for small message sizes to avoid that the time to solution is dominated by the overhead of MPI_Allreduce
                    for(int repeat=0; repeat < 1000; repeat++){ 
                        ccl::alltoall(d_sendbuf, d_recvbuf, count , ccl::datatype::float32, comm, stream, attr, deps).wait();
                    }
                    chain_size+=1000; // for small message sizes we repeat the collective 1000 times for each measurement
                }else{
                    ccl::alltoall(d_sendbuf, d_recvbuf, count , ccl::datatype::float32, comm, stream, attr, deps).wait();
                    chain_size++;
                }
                auto end_s = std::chrono::high_resolution_clock::now();
                MPI_Barrier(MPI_COMM_WORLD);
                // Time to solution of the current rank
                auto elapsed_time = std::chrono::duration_cast<std::chrono::microseconds>(end_s - start_s).count();
                a2a_time_per_rank += elapsed_time;
                a2a_time += elapsed_time;

                // Time to solution to complete the collective 
                MPI_Allreduce(MPI_IN_PLACE, &a2a_time, 1, MPI_LONG_LONG, MPI_MAX, MPI_COMM_WORLD);
            }
           
            powerProf.stop();
            prof_data_types::energy_t dev_energy_uj = powerProf.get_device_energy(); //device energy in uj for one collective run for the current rank
            double time_ms = static_cast<double>(a2a_time_per_rank) / 1000.0; // time in ms for the current rank
            double dev_energy_mj = static_cast<double>(dev_energy_uj) / 1000.0; 
            prof_data_types::energy_t host_energy_uj = powerProf.get_host_energy(); //host energy in uj for one collective run for the current rank
            double host_energy_mj = static_cast<double>(host_energy_uj) / 1000.0; 
            
            // host energy is 0 now
            clog::Logger::ProfilingInfo<T> prof_info{time_ms, buff_size_byte[i], dev_energy_mj, host_energy_mj, ctx.global_rank, ctx.local_rank, ctx.global_rank_size, run, true, chain_size, "composite"};  
            prof_data_types::power_trace_t power_trace = powerProf.get_power_execution_data(); // get power trace data and store it internally in the power_prof object
            std::string power_trace_str = prof_data_types::power_trace_to_string(power_trace);
            clog::CsvField power_trace_field{"power_trace", power_trace_str};
            csv_log.log_result<T>(prof_info, power_trace_field); 
        }
    }
   
    q.memcpy(d_recvbuf, h_recvbuf, buff_size_byte[num_iters-1]).wait();

    sycl::free(d_sendbuf, q);
    sycl::free(d_recvbuf, q);
    free(h_sendbuf);
    free(h_recvbuf);
}

int main(int argc, char *argv[]) {
    int rank, size;
    std::string csv_log_path;
    
    if (argc != 2)
        return -1;
    else{   
        csv_log_path = argv[1];
    }

    
    intel::utils::OneCCLContext ctx = intel::utils::init_oneccl(csv_log_path,"a2a", intel::utils::GPUMode::Composite); // initialize oneCCL and MPI

    // Run with different data type
    // run<uint8_t>(comm, rank, numGPUs, power_log_path, csv_log_path);
    // run<int>(comm, rank, numGPUs, power_log_path, csv_log_path);
    run<float>(ctx);
    // run<double>(comm, rank, numGPUs, power_log_path, csv_log_path);

    
    MPI_Finalize();

    return 0;
}