#include <mpi.h>
#include <cuda_runtime.h>
#include "../utils/nccl_data_type.hpp"
#include <chrono>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include "../../../energy-profiler/include/profiler/power_profiler.hpp"
#include "../utils/nccl_ctx.hpp"

#define MAX_RUN 3
#define WARM_UP_RUN 1
#define TIME_TO_ACHIEVE_MS 5000
#define POWER_SAMPLING_RATE_MS 40

namespace prof_data_types = profiler::data_types;
namespace clog = common::logger;

template<typename T>
void run(nvidia::utils::ncclContext& ctx){ 
    ncclDataType_t dtype = nccl_type_traits<T>::type;

    cudaStream_t stream = std::move(*ctx.stream);
    ncclComm_t comm = std::move(*ctx.comm);
    clog::Logger csv_log = std::move(*ctx.logger);

    int rank = ctx.global_rank;
    int numGPUs = ctx.local_rank_size;
    int size = ctx.global_rank_size;

    size_t buff_size_byte[] = { 2 * 1024 * 1024 };
    const int num_iters = std::size(buff_size_byte);

    T *d_sendbuf, *d_recvbuf;
    cudaMalloc((void **)&d_sendbuf, buff_size_byte[num_iters - 1]);
    cudaMalloc((void **)&d_recvbuf, buff_size_byte[num_iters - 1] * size);

    T *h_sendbuf = (T *)malloc(buff_size_byte[num_iters - 1]);
    T *h_recvbuf = (T *)malloc(buff_size_byte[num_iters - 1] * size);

    cudaStreamCreate(&stream);

    // 🔹 Warm-up
    for (int i = 0; i < WARM_UP_RUN; i++) {
        size_t count_el = buff_size_byte[num_iters-1] / sizeof(T);

        cudaMemcpy(h_sendbuf, d_sendbuf, buff_size_byte[num_iters-1], cudaMemcpyDeviceToHost);

        ncclGroupStart();
        for (int r = 0; r < size; r++) {
            ncclSend(d_sendbuf, count_el, dtype, r, comm, stream);
            ncclRecv(d_recvbuf + (r * count_el), count_el, dtype, r, comm, stream);
        }
        ncclGroupEnd();

        cudaStreamSynchronize(stream);
        MPI_Barrier(MPI_COMM_WORLD);

        cudaMemcpy(d_recvbuf, h_recvbuf, buff_size_byte[num_iters-1] * size, cudaMemcpyHostToDevice);
    }

    // Initialize buffer
    cudaMemset(d_sendbuf, rank, buff_size_byte[num_iters-1]);

    for (int i = 0; i < num_iters; i++) {
        size_t count_el = buff_size_byte[i] / sizeof(T);

        for (int run = 0; run < MAX_RUN; run++) {
            size_t a2a_time = 0;
            size_t a2a_time_per_rank = 0;
            int chain_size = 0;

            profiler::PowerProfiler powerProf(rank % numGPUs, 0, POWER_SAMPLING_RATE_MS);
            powerProf.start();

            while (a2a_time < (TIME_TO_ACHIEVE_MS * 1000)) {
                auto start_s = std::chrono::high_resolution_clock::now();

                ncclGroupStart();
                for (int r = 0; r < size; r++) {
                    ncclSend(d_sendbuf, count_el, dtype, r, comm, stream);
                    ncclRecv(d_recvbuf + (r * count_el), count_el, dtype, r, comm, stream);
                }
                ncclGroupEnd();

                cudaStreamSynchronize(stream);

                auto end_s = std::chrono::high_resolution_clock::now();
                auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end_s - start_s).count();

                a2a_time += elapsed;
                a2a_time_per_rank += elapsed;

                MPI_Allreduce(MPI_IN_PLACE, &a2a_time, 1, MPI_LONG_LONG, MPI_MAX, MPI_COMM_WORLD);

                chain_size++;
            }

            powerProf.stop();

            prof_data_types::energy_t dev_energy_uj = powerProf.get_device_energy();
            double time_ms = static_cast<double>(a2a_time_per_rank) / 1000.0;
            double dev_energy_mj = static_cast<double>(dev_energy_uj) / 1000.0;

            prof_data_types::energy_t host_energy_uj = powerProf.get_host_energy();
            double host_energy_mj = static_cast<double>(host_energy_uj) / 1000.0;

            clog::Logger::ProfilingInfo<T> prof_info{
                time_ms,
                buff_size_byte[i],
                dev_energy_mj,
                host_energy_mj,
                ctx.global_rank,
                ctx.local_rank,
                ctx.global_rank_size,
                run,
                true,
                chain_size,
                "composite"
            };

            prof_data_types::power_trace_t power_trace = powerProf.get_power_execution_data();
            std::string power_trace_str = prof_data_types::power_trace_to_string(power_trace);
            clog::CsvField power_trace_field{"power_trace", power_trace_str};

            csv_log.log_result<T>(prof_info, power_trace_field);
        }
    }

    cudaStreamDestroy(stream);
    cudaFree(d_sendbuf);
    cudaFree(d_recvbuf);
    free(h_sendbuf);
    free(h_recvbuf);
}

int main(int argc, char *argv[]) {

    if (argc != 2)
        return -1;

    std::string csv_path = argv[1];

    nvidia::utils::ncclContext ctx = nvidia::utils::init_nccl(csv_path, "a2a");

    run<float>(ctx);

    ncclCommDestroy(std::move(*ctx.comm));
    MPI_Finalize();

    return 0;
}