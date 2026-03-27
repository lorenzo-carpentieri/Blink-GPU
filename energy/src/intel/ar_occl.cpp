
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

namespace clog = common::logger; // define logger namespace abbreviation
namespace data_types = common::utils::data_types; // define data_types namespace abbreviation
namespace prof_data_types = profiler::data_types; // define profiler::data_types namespace abbreviation
using Comm = data_types::CommWrapper<>::type;  // automaticall select ccl::communicator at compile time accroding to the defined macro USE_ONECCL

template<typename T>
void run(intel::utils::OneCCLContext& ctx){ 

    sycl::queue q = ctx.q; // sycl queue
    ccl::stream stream =std::move(*ctx.stream); // ccl stream
    Comm comm = std::move(*ctx.comm);
    log::Logger csv_log = std::move(*ctx.logger);

    int rank = ctx.global_rank;
    int numGPUs = ctx.local_rank_size;
    

    size_t buff_size_byte[] = {
        4,
        32, 64, 512, 4096, 32768, 262144,
        2097152, 16777216, 134217728, 1073741824
    };
  
    const int num_iters = std::size(buff_size_byte);

    // allocate device pointer
    T *d_sendbuf = sycl::malloc_device<T>(buff_size_byte[num_iters - 1], q); 
    T *d_recvbuf = sycl::malloc_device<T>(buff_size_byte[num_iters - 1], q); 

    /* init device side buffer */
    q.memset(d_sendbuf, rank, buff_size_byte[num_iters - 1]);
    q.memset(d_recvbuf, 0, buff_size_byte[num_iters - 1]);


    // create dependencies vector: can be used in the collective to ensure that consecutive call to the collective are exectude in order
    std::vector<ccl::event> deps;
    auto attr = ccl::create_operation_attr<ccl::allreduce_attr>();


    // allocate host pointer
    T *h_sendbuf = (T *)malloc(buff_size_byte[num_iters - 1]);
    T *h_recvbuf = (T *)malloc(buff_size_byte[num_iters - 1]);
    if (rank == 0) {
        std::cout<< "Warm up run for allreduce" << std::endl;    
    }
    
    for (int i = 0; i < WARM_UP_RUN; i++) {
        q.memcpy(h_sendbuf, d_sendbuf, buff_size_byte[num_iters-1]).wait();

        auto start = std::chrono::high_resolution_clock::now();
        ccl::allreduce(d_sendbuf, d_recvbuf, buff_size_byte[num_iters-1] / sizeof(T), ccl::datatype::float32,  ccl::reduction::sum, comm, stream, attr, deps).wait();

        auto end = std::chrono::high_resolution_clock::now();
        auto time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        
        q.memcpy(h_recvbuf, d_recvbuf, buff_size_byte[num_iters-1]).wait();       
    }

 

    
    
    for (int i = 0; i < num_iters; i++) {
        int chain_size = 0; //  num of times that a collective is executed 
        int count = buff_size_byte[i] / sizeof(T); // total num of elemenent per buffer
        for (int run = 0; run < MAX_RUN; run++) {
            if (rank == 0) {
                std::cout<< "Run " << run << " for allreduce with message size "<< buff_size_byte[i] << " B" << std::endl;    
            }
            size_t ar_time = 0;
            size_t ar_time_per_rank = 0;
            chain_size = 0;
            
            profiler::PowerProfiler powerProf(rank % numGPUs, 0, POWER_SAMPLING_RATE_MS);
            powerProf.start();
            while (ar_time < (TIME_TO_ACHIEVE_MS * 1000)) {
                auto start_s = std::chrono::high_resolution_clock::now();
                
                ccl::allreduce(d_sendbuf, d_recvbuf,  buff_size_byte[i] / sizeof(T),ccl::datatype::float32, ccl::reduction::sum, comm, stream, attr, deps).wait();

                auto end_s = std::chrono::high_resolution_clock::now();
                MPI_Barrier(MPI_COMM_WORLD);
                // Time to solution of the current rank
                auto elapsed_time = std::chrono::duration_cast<std::chrono::microseconds>(end_s - start_s).count();
                ar_time_per_rank += elapsed_time;
                ar_time += elapsed_time;

                // Time to solution to complete the collective 
                MPI_Allreduce(MPI_IN_PLACE, &ar_time, 1, MPI_LONG_LONG, MPI_MAX, MPI_COMM_WORLD);
                chain_size++;
            }
           
            powerProf.stop();
            
            prof_data_types::energy_t dev_energy_uj = powerProf.get_device_energy(); //device energy in uj for one collective run for the current rank
            double time_ms = static_cast<double>(ar_time_per_rank) / 1000.0; // time in ms for the current rank
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

    std::cerr << "[DEBUG] CSV log path: " << csv_log_path << std::endl;
    std::cerr << "[DEBUG] Initialize MPI with oneCCL ...: " << std::endl;

    intel::utils::OneCCLContext ctx = intel::utils::init_oneccl(csv_log_path,"ar", intel::utils::GPUMode::Composite); // initialize oneCCL and MPI

    run<float>(ctx);
    // run<double>(comm, rank, numGPUs, power_log_path, csv_log_path);

    
    MPI_Finalize();

    return 0;
}