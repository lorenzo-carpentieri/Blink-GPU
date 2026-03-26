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
        MPI_Request reqs[2];

        // Bidirectional GPU-aware send/recv
        MPI_Isend(sendbuff, bytes, MPI_BYTE, 1-rank, 0, MPI_COMM_WORLD, &reqs[0]);
        MPI_Irecv(recvbuff, bytes, MPI_BYTE, 1-rank, 0, MPI_COMM_WORLD, &reqs[1]);

        MPI_Waitall(2, reqs, MPI_STATUSES_IGNORE);
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