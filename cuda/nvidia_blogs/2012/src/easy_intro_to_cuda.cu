/*
    Link: https://developer.nvidia.com/blog/easy-introduction-cuda-c-and-c/
*/

#include <cstdio>
#include <vector>

#include <cuda_runtime_api.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>

#define CEIL_DIV(N, M) (((N) + (M) - 1) / (M))

__global__ void saxpy(int n, float a, float *x, float *y) {
  const auto i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    y[i] = a * x[i] + y[i];
}

int main() {
  constexpr unsigned int N = 1 << 20;
  thrust::device_vector<float> d_x(N);
  thrust::device_vector<float> d_y(N);
  std::vector<float> x(N);
  std::vector<float> y(N);

  for (int i = 0; i < N; ++i) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }

  thrust::copy(x.begin(), x.end(), d_x.begin());
  thrust::copy(y.begin(), y.end(), d_y.begin());

  constexpr unsigned int block_size = 256;
  saxpy<<<CEIL_DIV(N, block_size), block_size>>>(
      N, 2.0f, thrust::raw_pointer_cast(d_x.data()),
      thrust::raw_pointer_cast(d_y.data()));

  thrust::copy(d_y.begin(), d_y.end(), y.begin());

  float max_error = 0.0f;
  for (int i = 0; i < N; ++i)
    max_error = std::max(max_error, std::abs(y[i] - 4.0f));
  std::printf("Max error: %f\n", max_error);

  return 0;
}
