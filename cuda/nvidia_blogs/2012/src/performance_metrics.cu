#include <iostream>
#include <vector>

#include <cuda_runtime_api.h>

#define CEIL_DIV(N, M) (((N) + (M) - 1) / (M))

__global__ void saxpy(int n, float a, float *x, float *y) {
  const auto i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    y[i] = a * x[i] + y[i];
}

int main(int argc, char *argv[]) {
  constexpr auto N = 80 * (1U << 20); // 80 M
  float *d_x;
  float *d_y;
  std::vector<float> x(N);
  std::vector<float> y(N);

  const auto bytes = N * sizeof(float);
  cudaMalloc(&d_x, bytes);
  cudaMalloc(&d_y, bytes);

  for (int i = 0; i < N; ++i) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }

  cudaMemcpy(d_x, x.data(), bytes, cudaMemcpyDefault);
  cudaMemcpy(d_y, y.data(), bytes, cudaMemcpyDefault);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);

  constexpr auto block_size = 512U;
  saxpy<<<CEIL_DIV(N, block_size), block_size>>>(N, 2.0f, d_x, d_y);

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  cudaMemcpy(y.data(), d_y, bytes, cudaMemcpyDefault);

  auto milliseconds = 0.0f;
  cudaEventElapsedTime(&milliseconds, start, stop);

  auto max_error = 0.0f;
  for (int i = 0; i < N; ++i)
    max_error = std::max(max_error, std::abs(y[i] - 4.0f));
  std::cout << "Max error: " << max_error << std::endl;

  const auto effective_bandwidth = 3 * bytes / (milliseconds * 1e6);
  std::cout << "Effective Bandwidth (GB/s): " << effective_bandwidth
            << std::endl;

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaFree(d_x);
  cudaFree(d_y);

  return 0;
}
