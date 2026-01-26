
// You can use template paramter for specilize the funcition that run the collective with different data types
//TODO: Test multi nodo per vedere se cambia qualcosa dal punto di vista delle performance e dell'energy
#include <mpi.h>
#include <sycl/sycl.hpp>
#include <chrono>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

#define T float 
#define BYTES_SIZE 13

int main(int argc, char *argv[]) {
    int rank, size,local_rank, local_rank_size;
    
    constexpr std::array<size_t, BYTES_SIZE> bytes = [] {
        std::array<size_t, BYTES_SIZE> a{};
        size_t value = 1024;
        for (size_t i = 0; i < a.size(); ++i) {
            a[i] = value;
            value <<= 1; // multiply by 2
        }
        return a;
    }(); 
    
    constexpr int MAX_RUN = 5;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    MPI_Comm local_comm; // create a local communicator for each node
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local_comm);
    MPI_Comm_rank(local_comm, &local_rank); // index of the rank inside the local communicator
    MPI_Comm_size(local_comm, &local_rank_size); // size of the local communicator (i.e., number of ranks on the same node)
            
    std::vector<sycl::device> candidates; // Available devices for each rank
            
    for (const auto& plat : sycl::platform::get_platforms()) { // Handle only Level-Zero platforms: oneCCL only supports Intel GPUs 
        auto name = plat.get_info<sycl::info::platform::name>();
        if (name.find("Level-Zero") == std::string::npos) continue;

        for (const auto& root : plat.get_devices()) {
            if (!root.is_gpu()) continue;

            candidates.push_back(root);
        }
    }
    std::cout << "Rank " << rank << " found " << candidates.size() << " devices\n";

    sycl::queue q(candidates[rank % local_rank_size]); // Simple round-robin assignment

    for (const auto& byte : bytes) {
        // Allocate device buffers
        size_t num_elements = byte / sizeof(T);
        T* d_sendbuf = sycl::malloc_device<T>(num_elements, q);
        T* d_recvbuf = sycl::malloc_device<T>(num_elements, q);

        // Initialize device buffers
        q.parallel_for(num_elements, [=](auto i) {
            d_sendbuf[i] = rank;
            d_recvbuf[i] = -1;
        }).wait();

        // Allocate host buffers for MPI
        std::vector<T> h_sendbuf(num_elements);
        std::vector<T> h_recvbuf(num_elements);


        int peer = (rank + local_rank_size) % size; // The number of GPUs per node is equal to the number of ranks per node
        for (int run = 0; run < MAX_RUN; run++) {
            double t1 = MPI_Wtime();

            MPI_Sendrecv(d_sendbuf, num_elements, MPI_FLOAT, peer, 0, d_recvbuf, num_elements, MPI_FLOAT, peer, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            double elapsed = MPI_Wtime() - t1;
            double max_elapsed;
            MPI_Allreduce(&elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

            if (rank == 0) {
                // Each rank send "byte" bytes to another rank so the number of bytes sent is multiplies for the number of local rank that is equal to number of GPUs allocated
                double gbps = (byte * local_rank_size * 8.0) / (max_elapsed * 1e9); 
                std::cout << "Run " << run << " Byte size: " << byte  << " One-way bandwidth = " << gbps << " Gb/s\n";
            }
        }

        // Copy back result to device (optional)
        q.memcpy(h_recvbuf.data(), d_recvbuf, byte).wait();
        sycl::free(d_sendbuf, q);
        sycl::free(d_recvbuf, q);
         // Validate first 10 elements
        bool ok = true;
        for (int i = 0; i < 10; i++) {
            if (h_recvbuf[i] != peer) {
                ok = false;
                std::cerr << "Validation failed at index " << i << ": got " << h_recvbuf[i] << " expected " << peer << "\n";
                break;
            }
        }
        if (ok && rank == 0) std::cout << "Validation passed!\n";
    }
  
   

   

    MPI_Finalize();

    

    return 0;
}