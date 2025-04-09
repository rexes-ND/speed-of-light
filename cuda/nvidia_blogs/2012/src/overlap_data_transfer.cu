/*
  Link: https://developer.nvidia.com/blog/how-overlap-data-transfers-cuda-cc/
*/

#include <array>
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

__global__ void kernel(float *a, int offset) {
  auto i{offset + threadIdx.x + blockIdx.x * blockDim.x};
  auto x{static_cast<float>(i)};
  auto s{sinf(x)};
  auto c{cosf(x)};
  a[i] += sqrtf(s * s + c * c);
}

float maxError(float *a, int n) {
  float maxE{0};
  for (int i = 0; i < n; ++i) {
    float error{fabs(a[i] - 1.0f)};
    if (error > maxE)
      maxE = error;
  }
  return maxE;
}

int main(int argc, char **argv) {
  constexpr int blockSize{256};
  constexpr int nStreams{4};
  constexpr int n{4 * 1024 * blockSize * nStreams};
  constexpr int streamSize{n / nStreams};
  constexpr int bytes{n * sizeof(float)};
  constexpr int streamBytes{streamSize * sizeof(float)};

  int devId{(argc > 1) ? atoi(argv[1]) : 0};

  cudaDeviceProp prop;
  checkCuda(cudaGetDeviceProperties(&prop, devId));
  printf("Device : %s\n", prop.name);
  checkCuda(cudaSetDevice(devId));

  float *a, *d_a;
  checkCuda(cudaMallocHost(&a, bytes));
  checkCuda(cudaMalloc(&d_a, bytes));

  cudaEvent_t startEvent, stopEvent, dummyEvent;
  std::array<cudaStream_t, nStreams> streams;
  checkCuda(cudaEventCreate(&startEvent));
  checkCuda(cudaEventCreate(&stopEvent));
  checkCuda(cudaEventCreate(&dummyEvent));
  for (int i = 0; i < nStreams; ++i)
    checkCuda(cudaStreamCreate(&streams[i]));

  // baseline case - sequential transfer and execute
  std::memset(a, 0, bytes);
  checkCuda(cudaEventRecord(startEvent, 0));
  checkCuda(cudaMemcpy(d_a, a, bytes, cudaMemcpyHostToDevice));
  kernel<<<n / blockSize, blockSize>>>(d_a, 0);
  checkCuda(cudaMemcpy(a, d_a, bytes, cudaMemcpyDeviceToHost));
  checkCuda(cudaEventRecord(stopEvent, 0));
  checkCuda(cudaEventSynchronize(stopEvent));
  float ms;
  checkCuda(cudaEventElapsedTime(&ms, startEvent, stopEvent));
  std::printf("Time for sequential transfer and execute (ms): %f\n", ms);
  std::printf("\tmax error: %e\n", maxError(a, n));

  // async v1: loop over {copy, kernel, copy}
  std::memset(a, 0, bytes);
  checkCuda(cudaEventRecord(startEvent, 0));
  for (int i = 0; i < nStreams; ++i) {
    const int offset{i * streamSize};
    checkCuda(cudaMemcpyAsync(&d_a[offset], &a[offset], streamBytes,
                              cudaMemcpyHostToDevice, streams[i]));
    kernel<<<streamSize / blockSize, blockSize, 0, streams[i]>>>(d_a, offset);
    checkCuda(cudaMemcpyAsync(&a[offset], &d_a[offset], streamBytes,
                              cudaMemcpyDeviceToHost, streams[i]));
  }
  checkCuda(cudaEventRecord(stopEvent, 0));
  checkCuda(cudaEventSynchronize(stopEvent));
  checkCuda(cudaEventElapsedTime(&ms, startEvent, stopEvent));
  std::printf("Time for asynchronous V1 transfer and execute (ms): %f\n", ms);
  std::printf("\tmax error: %e\n", maxError(a, n));

  // async v2: loop over copy, loop over kernel, loop over copy
  std::memset(a, 0, bytes);
  checkCuda(cudaEventRecord(startEvent, 0));
  for (int i = 0; i < nStreams; ++i) {
    const int offset{i * streamSize};
    checkCuda(cudaMemcpyAsync(&d_a[offset], &a[offset], streamBytes,
                              cudaMemcpyHostToDevice, streams[i]));
  }
  for (int i = 0; i < nStreams; ++i) {
    const int offset{i * streamSize};
    kernel<<<streamSize / blockSize, blockSize, 0, streams[i]>>>(d_a, offset);
  }
  for (int i = 0; i < nStreams; ++i) {
    const int offset{i * streamSize};
    checkCuda(cudaMemcpyAsync(&a[offset], &d_a[offset], streamBytes,
                              cudaMemcpyDeviceToHost, streams[i]));
  }
  checkCuda(cudaEventRecord(stopEvent, 0));
  checkCuda(cudaEventSynchronize(stopEvent));
  checkCuda(cudaEventElapsedTime(&ms, startEvent, stopEvent));
  std::printf("Time for asynchronouse V2 transfer and execute (ms): %f\n", ms);
  std::printf("\tmax error: %e\n", maxError(a, n));

  // Cleanup
  checkCuda(cudaEventDestroy(startEvent));
  checkCuda(cudaEventDestroy(stopEvent));
  checkCuda(cudaEventDestroy(dummyEvent));
  for (int i = 0; i < nStreams; ++i)
    checkCuda(cudaStreamDestroy(streams[i]));
  cudaFree(d_a);
  cudaFreeHost(a);

  return 0;
}
