#pragma once

#include <mpi.h>
#include <oneapi/ccl.hpp>
#include <sycl/sycl.hpp>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>
#include "../../../include/logger/logger.hpp" 

namespace log = common::logger;

namespace utils {
        
    struct OneCCLContext {
        int global_rank_size; // total number of ranks on all the nodes (i.e., on two nodes with 4 GPUs each and 8 ranks, global_rank_size is 8)
        int local_rank_size; // number of ranks on the same node (i.e., on two nodes with 4 GPUs each and 8 ranks, local_rank_size is 4)
        int global_rank; // from 0 to global_rank_size-1
        int local_rank; // from 0 to local_rank_size-1
        sycl::queue q; // sycl queue associated to the rank
        ccl::communicator comm;
        ccl::stream stream;
        log::Logger logger;
    };

    
    enum GPUMode {
        Composite, // Intel GPU as a whole
        Flat // Intel GPU as individual tiles
    };

    // local_size_rank is the number of rank running on the same node (i.e., local communicator size)
    // if we have 2 nodes with 4 GPUs each, and we run 8 ranks, local_size_rank is 4
    // Each rank will see 4 GPUs numbered from 0 to 3
    sycl::device pick_device_for_rank(utils::OneCCLContext& ctx,onst utils::GPUMode& gpu_mode) {
        std::vector<sycl::device> candidates; // Available devices for each rank
        
        for (const auto& plat : sycl::platform::get_platforms()) { // Handle only Level-Zero platforms: oneCCL only supports Intel GPUs 
            auto name = plat.get_info<sycl::info::platform::name>();
            if (name.find("Level-Zero") == std::string::npos) continue;

            for (const auto& root : plat.get_devices()) {
                if (!root.is_gpu()) continue;

                if (utils::GPUMode::Composite == gpu_mode) {
                    candidates.push_back(root);
                }
                else if (utils::GPUMode::Flat == gpu_mode) {
                    // each tile is considered as a single device
                    auto tiles = root.create_sub_devices<
                        sycl::info::partition_property::partition_by_affinity_domain>(
                        sycl::info::partition_affinity_domain::next_partitionable);
                    candidates.insert(candidates.end(), tiles.begin(), tiles.end());
                }
            }
        }

        if (ctx.local_rank >= (int)candidates.size()) { // Check if there are enough (sub)devices for all local ranks
            throw std::runtime_error("Not enough (sub)devices for ranks");
        }
        return candidates[ctx.local_rank];
    }

    // GPU mode can be "gpu" or "tile". An Intel Mac GPU 1550 can be programmed as two GPU tiles or one single GPU.
    // If the GPU is considered as two GPU tile, each tile is exposed as a single device.
    // Differenty if the GPU is considered as a single GPU, only one device is exposed.
    inline utils::OneCCLContext init_oneccl(const std::string& output_dir,
                                    const std::string& collective_name,
                                    const utils::GPUMode& gpu_mode = utils::GPUMode::Composite) { 
        ccl::init(); // Initialize oneCCL
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
        
        // Init OneCCLContext with rank info                                    
        utils::OneCCLContext ctx;
        ctx.global_rank_size = global_rank_size;
        ctx.local_rank_size = local_rank_size;
        ctx.global_rank = global_rank;
        ctx.local_rank = local_rank;
        // Scegli il (sub)device per questo rank
        sycl::device dev = pick_device_for_rank(ctx, gpu_mode);

        // Context e queue SOLO su quel (sub)device (best practice)
        sycl::context ctx(dev);
        sycl::queue q(ctx, dev, { sycl::property::queue::in_order() }); 

        // KVS + communicator + stream
        ccl::shared_ptr_class<ccl::kvs> kvs;
        ccl::kvs::address_type addr;
        if (global_rank == 0) { 
            kvs = ccl::create_main_kvs(); 
            addr = kvs->get_address(); 
        }
        MPI_Bcast(addr.data(), addr.size(), MPI_BYTE, 0, MPI_COMM_WORLD);
        if (global_rank != 0) kvs = ccl::create_kvs(addr);

        auto ccl_dev = ccl::create_device(q.get_device());  // create ccl device from sycl device
        auto ccl_ctx = ccl::create_context(q.get_context()); // crea ccl context from sycl context
        auto comm    = ccl::create_communicator(global_rank_size, global_rank, ccl_dev, ccl_ctx, kvs); // create communicator
        auto stream  = ccl::create_stream(q); // create ccl stream from sycl queue

        log::Logger logger(output_dir, "occl", collective_name); // create logger instance 
        return OneCCLContext{global_rank_size, local_rank_size, global_rank, local_rank, std::move(q), std::move(comm), std::move(stream), std::move(logger)};
    }
} // namespace utils