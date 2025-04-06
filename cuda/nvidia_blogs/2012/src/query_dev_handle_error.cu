/*
    Link:
   https://developer.nvidia.com/blog/how-query-device-properties-and-handle-errors-cuda-cc/
*/

#include <cstdio>

#include <cuda_runtime_api.h>

int main() {
  int nDevices{};
  cudaError_t err = cudaGetDeviceCount(&nDevices);
  if (err != cudaSuccess) {
    std::printf("%s\n", cudaGetErrorString(err));
    return 1;
  }

  /*
    Device number: 0
      Device name: NVIDIA GeForce RTX 4090
      Memory Clock Rate (KHz): 10501000
      Memory Bus Width (bits): 384
      Peak Memory Bandwidth (GB/s): 1008.096000
  */

  for (int i = 0; i < nDevices; ++i) {
    cudaDeviceProp prop;
    cudaError_t err = cudaGetDeviceProperties(&prop, i);
    if (err != cudaSuccess) {
      std::printf("%s\n", cudaGetErrorString(err));
      return 1;
    }
    std::printf("Device number: %d\n", i);
    std::printf("\tDevice name: %s\n", prop.name);
    std::printf("\tMemory Clock Rate (KHz): %d\n",
                prop.memoryClockRate); // NOTE(rexes): Deprecated
    std::printf("\tMemory Bus Width (bits): %d\n", prop.memoryBusWidth);
    std::printf("\tPeak Memory Bandwidth (GB/s): %f\n\n",
                2.0 * (prop.memoryClockRate / 1e6) * (prop.memoryBusWidth / 8));
  }

  return 0;
}
