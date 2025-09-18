#include <iostream>
#include <string>

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

template <typename T> __global__ void offset(T *a, int s) {
  const auto i = blockDim.x * blockIdx.x + threadIdx.x + s;
  a[i] = a[i] + 1;
}

template <typename T> __global__ void stride(T *a, int s) {
  const auto i = (blockDim.x * blockIdx.x + threadIdx.x) * s;
  a[i] = a[i] + 1;
}

template <typename T> void run_test(int dev_id, unsigned int nMB) {
  constexpr auto block_size = 256U;

  T *d_a;
  const auto N = (nMB << 20) / sizeof(T);
  CHECK_CUDA_ERROR(cudaMalloc(&d_a, N * 33 * sizeof(T)));

  float ms;
  cudaEvent_t start_event, stop_event;
  CHECK_CUDA_ERROR(cudaEventCreate(&start_event));
  CHECK_CUDA_ERROR(cudaEventCreate(&stop_event));

  std::cout << "Offset, Bandwidth (GB/s):" << std::endl;

  offset<<<N / block_size, block_size>>>(d_a, 0); // warm up
  for (int i = 0; i <= 32; ++i) {
    CHECK_CUDA_ERROR(cudaMemset(d_a, 0, N * sizeof(T)));
    CHECK_CUDA_ERROR(cudaEventRecord(start_event));
    offset<<<N / block_size, block_size>>>(d_a, i);
    CHECK_CUDA_ERROR(cudaEventRecord(stop_event));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop_event));

    CHECK_CUDA_ERROR(cudaEventElapsedTime(&ms, start_event, stop_event));
    const auto bandwidth = 2 * nMB / ms;
    std::cout << i << ", " << bandwidth << std::endl;
  }

  std::cout << std::endl;
  std::cout << "Stride, Bandwidth (GB/s):" << std::endl;

  stride<<<N / block_size, block_size>>>(d_a, 1);
  for (int i = 1; i <= 32; ++i) {
    CHECK_CUDA_ERROR(cudaMemset(d_a, 0, N * sizeof(T)));
    CHECK_CUDA_ERROR(cudaEventRecord(start_event));
    stride<<<N / block_size, block_size>>>(d_a, i);
    CHECK_CUDA_ERROR(cudaEventRecord(stop_event));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop_event));

    CHECK_CUDA_ERROR(cudaEventElapsedTime(&ms, start_event, stop_event));
    std::cout << i << ", " << 2 * nMB / ms << std::endl;
  }

  CHECK_CUDA_ERROR(cudaEventDestroy(start_event));
  CHECK_CUDA_ERROR(cudaEventDestroy(stop_event));

  cudaFree(d_a);
}

int main(int argc, char *argv[]) {
  constexpr auto nMB = 256U;
  int dev_id = 0;
  bool is_fp64 = false;

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg.size() >= 4 && arg.substr(0, 4) == "dev=")
      dev_id = std::stoi(arg.substr(4));
    else if (arg == "fp64")
      is_fp64 = true;
  }

  cudaDeviceProp prop;
  CHECK_CUDA_ERROR(cudaSetDevice(dev_id));
  CHECK_CUDA_ERROR(cudaGetDeviceProperties(&prop, dev_id));
  std::cout << "Device: " << prop.name << std::endl;
  std::cout << "Transfer size (MB): " << nMB << std::endl;
  std::cout << (is_fp64 ? "Double" : "Single") << " Precision" << std::endl;

  if (is_fp64)
    run_test<double>(dev_id, nMB);
  else
    run_test<float>(dev_id, nMB);

  return 0;
}
