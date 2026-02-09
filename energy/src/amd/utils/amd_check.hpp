#pragma once
#define CHECK_HIP(cmd) do {                               \
  hipError_t e = cmd;                                     \
  if (e != hipSuccess) {                                  \
    printf("HIP error %s:%d: %s\n",                       \
           __FILE__, __LINE__, hipGetErrorString(e));     \
    MPI_Abort(MPI_COMM_WORLD, -1);                        \
  }                                                       \
} while(0)

#define CHECK_RCCL(cmd) do {                              \
  ncclResult_t r = cmd;                                   \
  if (r != ncclSuccess) {                                 \
    printf("RCCL error %s:%d: %s\n",                      \
           __FILE__, __LINE__, ncclGetErrorString(r));    \
    MPI_Abort(MPI_COMM_WORLD, -1);                        \
  }                                                       \
} while(0)
