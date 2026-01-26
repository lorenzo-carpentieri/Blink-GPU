
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

#define MAX_RUN 10

namespace log = common::logger; // define logger namespace abbreviation
namespace data_types = common::utils::data_types; // define data_types namespace abbreviation
namespace prof_data_types = profiler::data_types; // define profiler::data_types namespace abbreviation
using Comm = data_types::CommWrapper<>::type;  // automaticall select ccl::communicator at compile time accroding to the defined macro USE_ONECCL

template<typename T>
void run(intel::utils::OneCCLContext& ctx){ 

    sycl::queue& q = ctx.q; // sycl queue

   
    ccl::stream stream =std::move(*ctx.stream); // ccl stream
    Comm comm = std::move(*ctx.comm);
    log::Logger csv_log = std::move(*ctx.logger);

    int rank = ctx.global_rank;
    int num_ranks = ctx.global_rank_size;

    int numGPUs = ctx.local_rank_size;
    
    constexpr size_t ONE_GB = 1024 * 1024 * 1024 ; // in bytes
    size_t buff_size = ONE_GB;
    size_t num_el = ONE_GB / sizeof(T);

    char hostname[256];
    gethostname(hostname, sizeof(hostname));

    printf("Rank %d running on host %s\n", rank, hostname);


    T* d_sendbuf = sycl::malloc_device<T>( num_el, q);
    T* d_recvbuf = sycl::malloc_device<T>( num_el, q);

    /* init device side buffer */
    sycl::event init_event = q.submit([&](auto& h) {
        h.parallel_for(num_el, [=](auto id) {
            d_sendbuf[id] = rank;
            d_recvbuf[id] = -1;
        });
    });

    // create dependencies vector: can be used in the collective to ensure that consecutive call to the collective are exectude in order
    std::vector<ccl::event> deps;


    // allocate host pointer
    T *h_sendbuf = (T *)malloc( buff_size);
    T *h_recvbuf = (T *)malloc( buff_size);
    if (rank == 0) {
        std::cout<< "[INFO] Warm up run for send recv" << std::endl;    
    }
    int peer = (rank + numGPUs) % num_ranks; // The number of GPUs per node is equal to the number of ranks per node
    std::cout << "Num gpus per rank: " << numGPUs << " Rank: " << rank << " Peer: "<< peer << std::endl;
    MPI_Barrier(MPI_COMM_WORLD);

    std::vector<double> times;
    for (int i = 0; i < MAX_RUN; i++) {
        double t1 = MPI_Wtime();

        // Launch send/recv in parallel (non-blocking)
        ccl::event e_send, e_recv;
        
        if (rank == 0 || rank==1) { 
            e_send = ccl::send(d_sendbuf, num_el, peer, comm, stream);
            e_recv = ccl::recv(d_recvbuf, num_el, peer, comm, stream); 
        } 
        else {
            e_recv = ccl::recv(d_recvbuf, num_el, peer, comm, stream); 
            e_send = ccl::send(d_sendbuf, num_el, peer, comm, stream); 
        } // Wait for both to complete using move semantics e_send.wait(); e_recv.wait();
     
        // Wait for both to complete using move semantics
        e_send.wait();
        e_recv.wait();
        double elapsed_time = MPI_Wtime() - t1;
        MPI_Allreduce(MPI_IN_PLACE, &elapsed_time, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        if (rank == 0) {
            double gbps = (buff_size * 8.0) / (elapsed_time * 1e9);
            std::cout << "One-way goodput: " << gbps << " Gb/s\n";
        }       
        times.push_back(elapsed_time);
    }
    
    q.memcpy(h_recvbuf, d_recvbuf , buff_size).wait();       



    for(int k=0; k<100; k++){
        if(h_recvbuf[k] != peer){
            std::cout << "[ERROR] Validation failed at index " << k << " expected " << peer << " got " << h_recvbuf[k] << std::endl;
            break;
        }
    }

    std::cout << "[INFO] Rank " << rank << " Send/Recv completed successfully." << std::endl;
    double avg_time = 0.0;
    for(auto t : times){
        avg_time += t;
    }
    avg_time /= times.size();
    if (rank == 0) {
        double gbps = (buff_size * 8.0) / (avg_time * 1e9);
        std::cout << "One-way goodput avg: " << gbps << " Gb/s\n";
    }    

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

    run<float>(ctx);
    // run<double>(comm, rank, numGPUs, power_log_path, csv_log_path);

    
    MPI_Finalize();

    return 0;
}