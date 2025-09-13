#include <iostream>

#include <cuda_runtime_api.h>

#define CHECK_CUDA_ERROR(err) __check_cuda_error(err, __FILE__, __LINE__)
static void __check_cuda_error(cudaError_t err, const char *filename,
                               int line) {
  if (cudaSuccess != err) {
    std::cerr << "CUDA API error: " << cudaGetErrorString(err) << " from file "
              << filename << ", line " << line << std::endl;
    exit(err);
  }
}

int main() {
  int num_devices;
  CHECK_CUDA_ERROR(cudaGetDeviceCount(&num_devices));
  for (int dev_id = 0; dev_id < num_devices; ++dev_id) {
    cudaDeviceProp dev_prop;
    CHECK_CUDA_ERROR(cudaGetDeviceProperties(&dev_prop, dev_id));

    int memoryClockRate;
    CHECK_CUDA_ERROR(cudaDeviceGetAttribute(
        &memoryClockRate, cudaDevAttrMemoryClockRate, dev_id));

    const auto memory_bandwidth = 2.0 * (memoryClockRate / 1e6) *
                                  static_cast<int>(dev_prop.memoryBusWidth / 8);
    std::cout << "Device number: " << dev_id << std::endl;
    std::cout << "\tDevice name: " << dev_prop.name << std::endl;
    std::cout << "\tMemory Clock Rate (KHz): " << memoryClockRate << std::endl;
    std::cout << "\tMemory Bus Width (bits): " << dev_prop.memoryBusWidth
              << std::endl;
    std::cout << "\tPeak Memory Bandwidth (GB/s): " << memory_bandwidth
              << std::endl;
    std::cout << std::endl;
  }

  return 0;
}
