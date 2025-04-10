/*
  Link:
  https://developer.nvidia.com/blog/how-access-global-memory-efficiently-cuda-c-kernels/
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

template <typename T> __global__ void offset(T *a, int s) {
  unsigned int i{blockDim.x * blockIdx.x + threadIdx.x + s};
  a[i] = a[i] + 1; // L/S
}

template <typename T> __global__ void stride(T *a, int s) {
  unsigned int i{(blockDim.x * blockIdx.x + threadIdx.x) * s};
  a[i] = a[i] + 1;
}

template <typename T> void runTest(int deviceId, int nMB) {
  constexpr int blockSize{256};

  T *d_a{};
  const size_t n{nMB * 1024 * 1024 / sizeof(T)};
  checkCuda(cudaMalloc(&d_a, n * 33 * sizeof(T)));

  float ms{};
  cudaEvent_t startEvent, stopEvent;
  checkCuda(cudaEventCreate(&startEvent));
  checkCuda(cudaEventCreate(&stopEvent));

  std::printf("Offset, Bandwidth (GB/s):\n");

  offset<<<n / blockSize, blockSize>>>(d_a, 0); // warm up
  for (int i = 0; i <= 32; ++i) {
    checkCuda(cudaMemset(d_a, 0, n * sizeof(T)));
    checkCuda(cudaEventRecord(startEvent, 0));
    offset<<<n / blockSize, blockSize>>>(d_a, i);
    checkCuda(cudaEventRecord(stopEvent, 0));
    checkCuda(cudaEventSynchronize(stopEvent));

    checkCuda(cudaEventElapsedTime(&ms, startEvent, stopEvent));
    std::printf("%d, %f\n", i, 2 * nMB / ms);
  }

  std::printf("\n");
  std::printf("Stride, Bandwidth (GB/s):\n");

  stride<<<n / blockSize, blockSize>>>(d_a, 1);
  for (int i = 1; i <= 32; ++i) {
    checkCuda(cudaMemset(d_a, 0, n * sizeof(T)));
    checkCuda(cudaEventRecord(startEvent, 0));
    stride<<<n / blockSize, blockSize>>>(d_a, i);
    checkCuda(cudaEventRecord(stopEvent, 0));
    checkCuda(cudaEventSynchronize(stopEvent));

    checkCuda(cudaEventElapsedTime(&ms, startEvent, stopEvent));
    std::printf("%d, %f\n", i, 2 * nMB / ms);
  }

  checkCuda(cudaEventDestroy(startEvent));
  checkCuda(cudaEventDestroy(stopEvent));
  cudaFree(d_a);
}

int main(int argc, char *argv[]) {
  constexpr int nMB{256};
  int deviceId{0};
  bool bFp64{false};

  for (int i = 1; i < argc; ++i) {
    if (!strncmp(argv[i], "dev=", 4))
      deviceId = atoi(&argv[i][4]);
    else if (!strcmp(argv[i], "fp64"))
      bFp64 = true;
  }

  cudaDeviceProp prop;
  checkCuda(cudaSetDevice(deviceId));
  checkCuda(cudaGetDeviceProperties(&prop, deviceId));
  std::printf("Device: %s\n", prop.name);
  std::printf("Transfer size (MB): %d\n", nMB);
  std::printf("%s Precision\n", bFp64 ? "Double" : "Single");

  if (bFp64)
    runTest<double>(deviceId, nMB);
  else
    runTest<float>(deviceId, nMB);

  return 0;
}
