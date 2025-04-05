/*
    Link: https://developer.nvidia.com/blog/easy-introduction-cuda-c-and-c/
*/

#include <cstdio>

#include <cuda_runtime_api.h>

__global__ void saxpy(int n, float a, float *x, float *y) {
  unsigned int i{blockIdx.x * blockDim.x + threadIdx.x};
  if (i < n)
    y[i] = a * x[i] + y[i];
}

int main() {
  constexpr int N{1 << 20};
  float *d_x, *d_y;
  float *x{new float[N]};
  float *y{new float[N]};

  const size_t bytes{N * sizeof(float)};
  cudaMalloc(&d_x, bytes);
  cudaMalloc(&d_y, bytes);

  for (int i = 0; i < N; ++i) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }

  cudaMemcpy(d_x, x, bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, y, bytes, cudaMemcpyHostToDevice);

  constexpr int block_size{256};
  saxpy<<<(N + block_size - 1) / block_size, block_size>>>(N, 2.0f, d_x, d_y);

  cudaMemcpy(y, d_y, bytes, cudaMemcpyDeviceToHost);

  float max_error{0.0f};
  for (int i = 0; i < N; ++i)
    max_error = max(max_error, std::abs(y[i] - 4.0f));
  std::printf("Max error: %f\n", max_error);

  cudaFree(d_x);
  cudaFree(d_y);
  delete[] x;
  delete[] y;

  return 0;
}
