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

    // Initialize buffers
    CHECK_HIP(hipMemset(sendbuff, rank, bytes));
    CHECK_HIP(hipMemset(recvbuff, 0, bytes));

    MPI_Barrier(MPI_COMM_WORLD);

    const int iterations = 50;
    char ack = 1;

    auto t_start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < iterations; i++) {
        if (rank == 0) {
            // Send large message (measured direction)
            MPI_Send(sendbuff, bytes, MPI_BYTE, 1, 0, MPI_COMM_WORLD);

            // Receive tiny ACK
            MPI_Recv(&ack, 1, MPI_BYTE, 1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        } else {
            // Receive large message
            MPI_Recv(recvbuff, bytes, MPI_BYTE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            // Send tiny ACK
            MPI_Send(&ack, 1, MPI_BYTE, 0, 1, MPI_COMM_WORLD);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto t_stop = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed = t_stop - t_start;
    double seconds = elapsed.count();

    if (rank == 0) {
        // Only count one direction (true unidirectional)
        double total_gb = (double)bytes * iterations / 1e9;
        double gbps = total_gb / seconds;

        printf("Message size: %zu bytes, Iterations: %d\n", bytes, iterations);
        printf("Unidirectional GPU-aware MPI goodput: %.2f GB/s\n", gbps);
    }

    CHECK_HIP(hipFree(sendbuff));
    CHECK_HIP(hipFree(recvbuff));

    MPI_Finalize();
    return 0;
}