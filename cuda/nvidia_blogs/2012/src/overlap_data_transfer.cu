#include <algorithm>
#include <array>
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

__global__ void kernel(float *a, int offset) {
  const auto i = offset + threadIdx.x + blockIdx.x * blockDim.x;
  const auto x = static_cast<float>(i);
  const auto s = sinf(x);
  const auto c = cosf(x);
  a[i] += sqrtf(s * s + c * c);
}

float maxError(float *a, int n) {
  float max_error = 0;
  for (int i = 0; i < n; ++i)
    max_error = std::max(max_error, std::abs(a[i] - 1.0f));
  return max_error;
}

int main(int argc, char **argv) {
  constexpr auto block_size = 256U;
  constexpr auto num_streams = 4;
  constexpr auto N = 4 * 1024 * block_size * num_streams;
  constexpr auto stream_size = N / num_streams;
  constexpr auto bytes = N * sizeof(float);
  constexpr auto stream_bytes = stream_size * sizeof(float);

  const auto dev_id = (argc > 1) ? atoi(argv[1]) : 0;

  cudaDeviceProp prop;
  CHECK_CUDA_ERROR(cudaGetDeviceProperties(&prop, dev_id));
  std::cout << "Device : " << prop.name << std::endl;
  CHECK_CUDA_ERROR(cudaSetDevice(dev_id));

  float *a, *d_a;
  CHECK_CUDA_ERROR(cudaMallocHost(&a, bytes));
  CHECK_CUDA_ERROR(cudaMalloc(&d_a, bytes));

  cudaEvent_t startEvent, stopEvent, dummyEvent;
  std::array<cudaStream_t, num_streams> streams;
  CHECK_CUDA_ERROR(cudaEventCreate(&startEvent));
  CHECK_CUDA_ERROR(cudaEventCreate(&stopEvent));
  CHECK_CUDA_ERROR(cudaEventCreate(&dummyEvent));
  for (int i = 0; i < num_streams; ++i)
    CHECK_CUDA_ERROR(cudaStreamCreate(&streams[i]));

  // baseline case - sequential transfer and execute
  std::fill_n(a, N, 0);
  CHECK_CUDA_ERROR(cudaEventRecord(startEvent));
  CHECK_CUDA_ERROR(cudaMemcpy(d_a, a, bytes, cudaMemcpyDefault));
  kernel<<<N / block_size, block_size>>>(d_a, 0);
  CHECK_CUDA_ERROR(cudaMemcpy(a, d_a, bytes, cudaMemcpyDefault));
  CHECK_CUDA_ERROR(cudaEventRecord(stopEvent));
  CHECK_CUDA_ERROR(cudaEventSynchronize(stopEvent));
  float ms;
  CHECK_CUDA_ERROR(cudaEventElapsedTime(&ms, startEvent, stopEvent));
  std::cout << "Time for sequential transfer and execute (ms): " << ms
            << std::endl;
  std::cout << "\tmax error: " << maxError(a, N) << std::endl;

  // async v1: loop over {copy, kernel, copy}
  std::fill_n(a, N, 0);
  CHECK_CUDA_ERROR(cudaEventRecord(startEvent, 0));
  for (int i = 0; i < num_streams; ++i) {
    const auto offset = i * stream_size;
    CHECK_CUDA_ERROR(cudaMemcpyAsync(&d_a[offset], &a[offset], stream_bytes,
                                     cudaMemcpyDefault, streams[i]));
    kernel<<<stream_size / block_size, block_size, 0, streams[i]>>>(d_a,
                                                                    offset);
    CHECK_CUDA_ERROR(cudaMemcpyAsync(&a[offset], &d_a[offset], stream_bytes,
                                     cudaMemcpyDefault, streams[i]));
  }
  CHECK_CUDA_ERROR(cudaEventRecord(stopEvent, 0));
  CHECK_CUDA_ERROR(cudaEventSynchronize(stopEvent));
  CHECK_CUDA_ERROR(cudaEventElapsedTime(&ms, startEvent, stopEvent));
  std::cout << "Time for asynchronous V1 transfer and execute (ms): " << ms
            << std::endl;
  std::cout << "\tmax error: " << maxError(a, N) << std::endl;

  // async v2: loop over copy, loop over kernel, loop over copy
  // std::memset(a, 0, bytes);
  std::fill_n(a, N, 0);
  CHECK_CUDA_ERROR(cudaEventRecord(startEvent, 0));
  for (int i = 0; i < num_streams; ++i) {
    const auto offset = i * stream_size;
    CHECK_CUDA_ERROR(cudaMemcpyAsync(&d_a[offset], &a[offset], stream_bytes,
                                     cudaMemcpyDefault, streams[i]));
  }
  for (int i = 0; i < num_streams; ++i) {
    const auto offset = i * stream_size;
    kernel<<<stream_size / block_size, block_size, 0, streams[i]>>>(d_a,
                                                                    offset);
  }
  for (int i = 0; i < num_streams; ++i) {
    const auto offset = i * stream_size;
    CHECK_CUDA_ERROR(cudaMemcpyAsync(&a[offset], &d_a[offset], stream_bytes,
                                     cudaMemcpyDeviceToHost, streams[i]));
  }
  CHECK_CUDA_ERROR(cudaEventRecord(stopEvent, 0));
  CHECK_CUDA_ERROR(cudaEventSynchronize(stopEvent));
  CHECK_CUDA_ERROR(cudaEventElapsedTime(&ms, startEvent, stopEvent));
  std::cout << "Time for asynchronouse V2 transfer and execute (ms): " << ms
            << std::endl;
  std::cout << "\tmax error: " << maxError(a, N) << std::endl;

  // Cleanup
  CHECK_CUDA_ERROR(cudaEventDestroy(startEvent));
  CHECK_CUDA_ERROR(cudaEventDestroy(stopEvent));
  CHECK_CUDA_ERROR(cudaEventDestroy(dummyEvent));
  for (int i = 0; i < num_streams; ++i)
    CHECK_CUDA_ERROR(cudaStreamDestroy(streams[i]));
  cudaFree(d_a);
  cudaFreeHost(a);

  return 0;
}
