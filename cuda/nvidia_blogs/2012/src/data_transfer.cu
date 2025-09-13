#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

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

void profile_copy(float *h_a, float *h_b, float *d, unsigned int n,
                  const std::string &desc) {
  // 1. copy `h_a` to `d`
  // 2. copy `d`   to `h_b`
  std::cout << std::endl << desc << " transfer" << std::endl;
  const auto bytes = n * sizeof(float);

  cudaEvent_t start_event, stop_event;
  CHECK_CUDA_ERROR(cudaEventCreate(&start_event));
  CHECK_CUDA_ERROR(cudaEventCreate(&stop_event));

  CHECK_CUDA_ERROR(cudaEventRecord(start_event));
  CHECK_CUDA_ERROR(cudaMemcpy(d, h_a, bytes, cudaMemcpyDefault));
  CHECK_CUDA_ERROR(cudaEventRecord(stop_event));
  CHECK_CUDA_ERROR(cudaEventSynchronize(stop_event));

  float time;
  CHECK_CUDA_ERROR(cudaEventElapsedTime(&time, start_event, stop_event));
  std::cout << "\tHost to Device bandwidth (GB/s): " << bytes * 1e-6 / time
            << std::endl;

  CHECK_CUDA_ERROR(cudaEventRecord(start_event));
  CHECK_CUDA_ERROR(cudaMemcpy(h_b, d, bytes, cudaMemcpyDefault));
  CHECK_CUDA_ERROR(cudaEventRecord(stop_event));
  CHECK_CUDA_ERROR(cudaEventSynchronize(stop_event));

  CHECK_CUDA_ERROR(cudaEventElapsedTime(&time, start_event, stop_event));
  std::cout << "\tDevice to Host bandwidth (GB/s): " << bytes * 1e-6 / time
            << std::endl;

  for (int i = 0; i < n; ++i) {
    if (h_a[i] != h_b[i]) {
      std::cout << "*** " << desc << " transfers failed ***" << std::endl;
      break;
    }
  }
}

int main() {
  /*
    // nsys profile --stats=true -o /dev/null <executable>

    constexpr auto N = 1U << 20;
    std::vector<int> h_a(N);

    int *d_a;
    const auto bytes = N * sizeof(int);
    cudaMalloc(&d_a, bytes);

    cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyDefault);
    cudaMemcpy(h_a.data(), d_a, bytes, cudaMemcpyDefault);
  */

  constexpr auto N = 4U << 20;
  constexpr auto bytes = N * sizeof(float);

  std::vector<float> h_a(N);
  std::vector<float> h_b(N);

  float *h_a_pinned, *h_b_pinned;
  CHECK_CUDA_ERROR(cudaMallocHost(&h_a_pinned, bytes));
  CHECK_CUDA_ERROR(cudaMallocHost(&h_b_pinned, bytes));

  float *d_a;
  CHECK_CUDA_ERROR(cudaMalloc(&d_a, bytes));

  for (int i = 0; i < N; ++i)
    h_a[i] = i;

  std::copy_n(h_a.data(), N, h_a_pinned);
  std::fill(h_b_pinned, h_b_pinned + N, 0);

  cudaDeviceProp prop;
  CHECK_CUDA_ERROR(cudaGetDeviceProperties(&prop, 0));

  std::cout << std::endl << "Device: " << prop.name << std::endl;
  std::cout << "Transfer size (MB): " << (bytes >> 20) << std::endl;

  profile_copy(h_a.data(), h_b.data(), d_a, N, "Pageable");
  profile_copy(h_a_pinned, h_b_pinned, d_a, N, "Pinned");
  std::cout << std::endl;

  cudaFree(d_a);
  cudaFreeHost(h_a_pinned);
  cudaFreeHost(h_b_pinned);

  return 0;
}
