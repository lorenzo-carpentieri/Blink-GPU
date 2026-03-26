// #include <stdio.h>
// #include <stdlib.h>
// #include <mpi.h>
// #include <hip/hip_runtime.h>
// #include <rccl/rccl.h>
// #include <chrono>

// #define CHECK_HIP(cmd) do { \
//     hipError_t e = cmd; \
//     if (e != hipSuccess) { \
//         printf("HIP error %s:%d: %s\n", __FILE__, __LINE__, hipGetErrorString(e)); \
//         MPI_Abort(MPI_COMM_WORLD, -1); \
//     } \
// } while(0)

// #define CHECK_RCCL(cmd) do { \
//     ncclResult_t r = cmd; \
//     if (r != ncclSuccess) { \
//         printf("RCCL error %s:%d: %s\n", __FILE__, __LINE__, ncclGetErrorString(r)); \
//         MPI_Abort(MPI_COMM_WORLD, -1); \
//     } \
// } while(0)

// int main(int argc, char* argv[]) {
//     MPI_Init(&argc, &argv);

//     int rank, nranks;
//     MPI_Comm_rank(MPI_COMM_WORLD, &rank);
//     MPI_Comm_size(MPI_COMM_WORLD, &nranks);

//     if (nranks != 2) {
//         if (rank == 0) printf("This benchmark requires exactly 2 ranks\n");
//         MPI_Finalize();
//         return 0;
//     }

//     if (argc < 2) {
//         if (rank == 0) printf("Usage: %s <bytes>\n", argv[0]);
//         MPI_Finalize();
//         return 0;
//     }

//     size_t bytes = strtoull(argv[1], NULL, 10);
//     size_t count = bytes; // ncclInt8 elements

//     CHECK_HIP(hipSetDevice(0));

//     ncclUniqueId id;
//     if (rank == 0) CHECK_RCCL(ncclGetUniqueId(&id));
//     MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);

//     ncclComm_t comm;
//     CHECK_RCCL(ncclCommInitRank(&comm, 2, id, rank));

//     void* sendbuff;
//     void* recvbuff;
//     CHECK_HIP(hipMalloc(&sendbuff, bytes));
//     CHECK_HIP(hipMalloc(&recvbuff, bytes));

//     hipStream_t stream;
//     CHECK_HIP(hipStreamCreate(&stream));

//     // Warmup
//     for (int i = 0; i < 5; i++) {
//         ncclGroupStart();
//         CHECK_RCCL(ncclSend(sendbuff, count, ncclInt8, 1-rank, comm, stream));
//         CHECK_RCCL(ncclRecv(recvbuff, count, ncclInt8, 1-rank, comm, stream));
//         ncclGroupEnd();
//         CHECK_HIP(hipStreamSynchronize(stream));
//     }

//     MPI_Barrier(MPI_COMM_WORLD);

//     const int iterations = 50;
//     auto t_start = std::chrono::high_resolution_clock::now();

//     for (int i = 0; i < iterations; i++) {
//         ncclGroupStart();
//         // Bidirectional: each rank sends to the other and receives from the other
//         CHECK_RCCL(ncclSend(sendbuff, count, ncclInt8, 1-rank, comm, stream));
//         CHECK_RCCL(ncclRecv(recvbuff, count, ncclInt8, 1-rank, comm, stream));
//         ncclGroupEnd();
//     }

//     CHECK_HIP(hipStreamSynchronize(stream));

//     auto t_stop = std::chrono::high_resolution_clock::now();
//     std::chrono::duration<double> elapsed = t_stop - t_start;
//     double seconds = elapsed.count();

//     // Compute bidirectional throughput
//     if (rank == 0) {
//         double total_gb = (double)bytes * iterations * 2 / 1e9; // 2 transfers per iteration
//         double gbps = total_gb / seconds;
//         printf("Message size: %zu bytes, Iterations: %d\n", bytes, iterations);
//         printf("Bidirectional goodput: %.2f GB/s\n", gbps);
//     }

//     CHECK_HIP(hipFree(sendbuff));
//     CHECK_HIP(hipFree(recvbuff));
//     CHECK_RCCL(ncclCommDestroy(comm));
//     CHECK_HIP(hipStreamDestroy(stream));

//     MPI_Finalize();
//     return 0;
// }
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <hip/hip_runtime.h>
#include <chrono>

#define CHECK_HIP(cmd) do { \
    hipError_t e = cmd; \
    if (e != hipSuccess) { \
        printf("HIP error %s:%d: %s\n", __FILE__, __LINE__, hipGetErrorString(e)); \
        MPI_Abort(MPI_COMM_WORLD, -1); \
    } \
} while(0)

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, nranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);

    if (nranks != 2) {
        if (rank == 0) printf("This benchmark requires exactly 2 ranks\n");
        MPI_Finalize();
        return 0;
    }

    if (argc < 2) {
        if (rank == 0) printf("Usage: %s <bytes>\n", argv[0]);
        MPI_Finalize();
        return 0;
    }

    size_t bytes = strtoull(argv[1], NULL, 10);

    // Use GPU 0 per rank
    CHECK_HIP(hipSetDevice(0));

    void *sendbuff, *recvbuff;
    CHECK_HIP(hipMalloc(&sendbuff, bytes));
    CHECK_HIP(hipMalloc(&recvbuff, bytes));

    // Initialize buffer with some values
    CHECK_HIP(hipMemset(sendbuff, rank, bytes));
    CHECK_HIP(hipMemset(recvbuff, 0, bytes));

    MPI_Barrier(MPI_COMM_WORLD);

    const int iterations = 50;
    auto t_start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < iterations; i++) {
        MPI_Sendrecv(
            sendbuff, bytes, MPI_BYTE, 1 - rank, 0,   // send
            recvbuff, bytes, MPI_BYTE, 1 - rank, 0,   // recv
            MPI_COMM_WORLD, MPI_STATUS_IGNORE
        );
    }
    MPI_Barrier(MPI_COMM_WORLD);
    auto t_stop = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed = t_stop - t_start;
    double seconds = elapsed.count();

    if (rank == 0) {
        double total_gb = (double)bytes * iterations * 2 / 1e9; // bidirectional
        double gbps = total_gb / seconds;
        printf("Message size: %zu bytes, Iterations: %d\n", bytes, iterations);
        printf("Bidirectional GPU-aware MPI goodput: %.2f GB/s\n", gbps);
    }

    CHECK_HIP(hipFree(sendbuff));
    CHECK_HIP(hipFree(recvbuff));

    MPI_Finalize();
    return 0;
}