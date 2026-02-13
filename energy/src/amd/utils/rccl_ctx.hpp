#pragma once
#include <mpi.h>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>
#include <hip/hip_runtime.h>
#include <rccl/rccl.h>
#include "../../../include/logger/logger.hpp" 
#include "./amd_check.hpp"
namespace clog = common::logger;

namespace amd {
    namespace utils {
            
        struct rcclContext {
            int global_rank_size; // total number of ranks on all the nodes (i.e., on two nodes with 4 GPUs each and 8 ranks, global_rank_size is 8)
            int local_rank_size; // number of ranks on the same node (i.e., on two nodes with 4 GPUs each and 8 ranks, local_rank_size is 4)
            int global_rank; // from 0 to global_rank_size-1
            int local_rank; // from 0 to local_rank_size-1
            std::unique_ptr<hipStream_t> stream; // cuda stream associated to the rank
            std::unique_ptr<ncclComm_t> comm;
            std::unique_ptr<clog::Logger> logger;
        };

        
       
        
     

        // local_size_rank is the number of rank running on the same node (i.e., local communicator size)
        // if we have 2 nodes with 4 GPUs each, and we run 8 ranks, local_size_rank is 4
        // Each rank will see 4 GPUs numbered from 0 to 3
       int pick_device_for_rank(amd::utils::rcclContext& ctx) {
            int numGPUs;
            int rank = ctx.global_rank;

            CHECK_HIP(hipGetDeviceCount(&numGPUs));
            if (numGPUs == 0) {
                std::cerr << "No GPU devices available!" << std::endl;
                MPI_Abort(MPI_COMM_WORLD, 1);
            }

            // Local rank for each node
            int local_rank, local_size;
            MPI_Comm local_comm;
            MPI_Comm_split_type(
                MPI_COMM_WORLD,
                MPI_COMM_TYPE_SHARED,
                rank,
                MPI_INFO_NULL,
                &local_comm
            );
            MPI_Comm_rank(local_comm, &local_rank);
            MPI_Comm_size(local_comm, &local_size);

            // Bind each local process to a GPU
            int device = local_rank % numGPUs;
            CHECK_HIP(hipSetDevice(device));

            return device;
        }

        // GPU mode can be "gpu" or "tile". An Intel Mac GPU 1550 can be programmed as two GPU tiles or one single GPU.
        // If the GPU is considered as two GPU tile, each tile is exposed as a single device.
        // Differenty if the GPU is considered as a single GPU, only one device is exposed.
        inline amd::utils::rcclContext init_rccl(const std::string& output_dir,
                                        const std::string& collective_name) { 
            MPI_Init(nullptr, nullptr); // Initialize MPI
            
            int global_rank_size, global_rank, local_rank, local_rank_size; // define global and local rank and size variables
            
            // Global Communicator
            MPI_Comm_size(MPI_COMM_WORLD, &global_rank_size);
            MPI_Comm_rank(MPI_COMM_WORLD, &global_rank);
            
            // Local Communicator
            MPI_Comm local_comm; // create a local communicator for each node
            MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, global_rank, MPI_INFO_NULL, &local_comm);
            MPI_Comm_rank(local_comm, &local_rank); // index of the rank inside the local communicator
            MPI_Comm_size(local_comm, &local_rank_size); // size of the local communicator (i.e., number of ranks on the same node)
            
            // Init rcclContext with rank info                                    
            amd::utils::rcclContext ctx;
            ctx.global_rank_size = global_rank_size;
            ctx.local_rank_size = local_rank_size;
            ctx.global_rank = global_rank;
            ctx.local_rank = local_rank;

            int cuda_dev = pick_device_for_rank(ctx);
						// Create CUDA strem
            hipStream_t stream;
            CHECK_HIP(hipStreamCreate(&stream));
						
            // Communicator
            ncclComm_t comm;
            ncclUniqueId id;
            if (global_rank == 0) ncclGetUniqueId(&id);
            MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
            ncclCommInitRank(&comm, global_rank_size, id, global_rank);
						
            ctx.comm = std::make_unique<ncclComm_t>(comm);
			ctx.stream = std::make_unique<hipStream_t>(stream);
            clog::Logger logger(output_dir, "rccl", collective_name, "Composite"); // create logger instance 

            ctx.logger = std::make_unique<clog::Logger>(std::move(logger)); // move logger instance
            return ctx;
        }
    } // namespace utils
} // namespace intel
