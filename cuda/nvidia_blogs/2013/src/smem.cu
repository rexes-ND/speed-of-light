/*
    Link: https://developer.nvidia.com/blog/using-shared-memory-cuda-cc/
*/

#include <iostream>

#include <cuda_runtime_api.h>

constexpr unsigned int N{64};

__global__ void staticReverse(int *d) {
  __shared__ int s[N];
  unsigned int t{threadIdx.x};
  unsigned int trev{N - t - 1};
  s[t] = d[t];
  __syncthreads();
  d[t] = s[trev];
}

__global__ void dynamicReverse(int *d) {
  extern __shared__ int s[];
  unsigned int t{threadIdx.x};
  unsigned int trev{N - t - 1};
  s[t] = d[t];
  __syncthreads();
  d[t] = s[trev];
}

int main() {
  int a[N], arev[N], d[N];

  for (int i = 0; i < N; ++i) {
    a[i] = i;
    arev[i] = N - i - 1;
    d[i] = 0;
  }

  int *d_d{};
  cudaMalloc(&d_d, N * sizeof(int));

  cudaMemcpy(d_d, a, N * sizeof(int), cudaMemcpyHostToDevice);
  staticReverse<<<1, N>>>(d_d);
  cudaMemcpy(d, d_d, N * sizeof(int), cudaMemcpyDeviceToHost);

  for (int i = 0; i < N; ++i)
    if (d[i] != arev[i])
      std::printf("Error: d[%d] != arev[%d] (%d, %d)\n", i, i, d[i], arev[i]);

  cudaMemcpy(d_d, a, N * sizeof(int), cudaMemcpyHostToDevice);
  dynamicReverse<<<1, N, N * sizeof(int)>>>(d_d);
  cudaMemcpy(d, d_d, N * sizeof(int), cudaMemcpyDeviceToHost);
  for (int i = 0; i < N; ++i)
    if (d[i] != arev[i])
      std::printf("Error: d[%d] != arev[%d] (%d, %d)\n", i, i, d[i], arev[i]);

  return 0;
}
