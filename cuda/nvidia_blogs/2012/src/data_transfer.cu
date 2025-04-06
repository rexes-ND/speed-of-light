/*
    Link: https://developer.nvidia.com/blog/how-optimize-data-transfers-cuda-cc/
*/

#include <cassert>
#include <cstdio>
#include <cstring>

#include <cuda_runtime_api.h>

inline cudaError_t checkCuda(cudaError_t result) {
  if (result != cudaSuccess) {
    std::fprintf(stderr, "CUDA Runtime Error: %s\n",
                 cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
  return result;
}

void profileCopies(float *h_a, float *h_b, float *d, unsigned int n,
                   const char *desc) {
  /*
    Profiles `h_a` -> `d` and `d` -> `h_b`.
  */
  std::printf("\n%s transfer\n", desc);
  const size_t bytes{n * sizeof(float)};

  cudaEvent_t startEvent, stopEvent;
  checkCuda(cudaEventCreate(&startEvent));
  checkCuda(cudaEventCreate(&stopEvent));

  checkCuda(cudaEventRecord(startEvent, 0));
  checkCuda(cudaMemcpy(d, h_a, bytes, cudaMemcpyHostToDevice));
  checkCuda(cudaEventRecord(stopEvent, 0));
  checkCuda(cudaEventSynchronize(stopEvent));

  float time{};
  checkCuda(cudaEventElapsedTime(&time, startEvent, stopEvent));
  std::printf("\tHost to Device bandwidth (GB/s): %f\n", bytes * 1e-6 / time);

  checkCuda(cudaEventRecord(startEvent, 0));
  checkCuda(cudaMemcpy(h_b, d, bytes, cudaMemcpyDeviceToHost));
  checkCuda(cudaEventRecord(stopEvent, 0));
  checkCuda(cudaEventSynchronize(stopEvent));

  checkCuda(cudaEventElapsedTime(&time, startEvent, stopEvent));
  std::printf("\tDevice to Host bandwidth (GB/s): %f\n", bytes * 1e-6 / time);

  for (int i = 0; i < n; ++i) {
    if (h_a[i] != h_b[i]) {
      std::printf("*** %s transfers failed ***\n", desc);
      break;
    }
  }
}

int main() {
  /*
      CMD: nsys profile --stats=true -o /dev/null ./build/data_transfer
  */
  //   constexpr unsigned int N{1 << 20};
  //   int *h_a{new int[N]};
  //   int *d_a;
  //   const size_t bytes{N * sizeof(int)};
  //   cudaMalloc(&d_a, bytes);

  //   std::memset(h_a, 0, bytes);
  //   cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
  //   cudaMemcpy(h_a, d_a, bytes, cudaMemcpyDeviceToHost);

  /*
    Device: NVIDIA GeForce RTX 4090
    Transfer size (MB): 16

    Pageable transfer
        Host to Device bandwidth (GB/s): 6.159978
        Device to Host bandwidth (GB/s): 5.782758

    Pinned transfer
        Host to Device bandwidth (GB/s): 15.100895
        Device to Host bandwidth (GB/s): 12.688480
  */

  constexpr unsigned int nElements{4 << 20};
  const size_t bytes{nElements * sizeof(float)};

  float *h_aPageable{new float[nElements]};
  float *h_bPageable{new float[nElements]};

  float *h_aPinned, *h_bPinned;
  float *d_a;
  checkCuda(cudaMallocHost(&h_aPinned, bytes));
  checkCuda(cudaMallocHost(&h_bPinned, bytes));
  checkCuda(cudaMalloc(&d_a, bytes));

  for (int i = 0; i < nElements; ++i)
    h_aPageable[i] = i;

  std::memcpy(h_aPinned, h_aPageable, bytes);
  std::memset(h_bPageable, 0, bytes);
  std::memset(h_bPinned, 0, bytes);

  cudaDeviceProp prop;
  checkCuda(cudaGetDeviceProperties(&prop, 0));

  std::printf("\nDevice: %s\n", prop.name);
  std::printf("Transfer size (MB): %lu\n", bytes >> 20);

  profileCopies(h_aPageable, h_bPageable, d_a, nElements, "Pageable");
  profileCopies(h_aPinned, h_bPinned, d_a, nElements, "Pinned");
  std::printf("\n");

  cudaFree(d_a);
  cudaFreeHost(h_aPinned);
  cudaFreeHost(h_bPinned);
  delete[] h_aPageable;
  delete[] h_bPageable;

  return 0;
}
