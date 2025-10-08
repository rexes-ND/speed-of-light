#include <iostream>
#include <vector>

#include <cuda_runtime_api.h>

constexpr auto N = 64U;

__global__ void static_reverse(int *d) {
  __shared__ int s[N];

  const auto i = threadIdx.x;
  const auto irev = N - 1 - i;

  s[i] = d[i];
  __syncthreads();
  d[i] = s[irev];
}

__global__ void dynamic_reverse(int *d) {
  extern __shared__ int s[];

  const auto i = threadIdx.x;
  const auto irev = N - 1 - i;

  s[i] = d[i];
  __syncthreads();
  d[i] = s[irev];
}

int main() {
  std::vector<int> v(N);
  std::vector<int> vrev(N);
  std::vector<int> tmp_host(N);

  for (int i = 0; i < N; ++i) {
    v[i] = i;
    vrev[i] = N - 1 - i;
  }

  int *tmp_dev;
  cudaMalloc(&tmp_dev, N * sizeof(int));

  cudaMemcpy(tmp_dev, v.data(), N * sizeof(int), cudaMemcpyDefault);
  static_reverse<<<1, N>>>(tmp_dev);
  cudaMemcpy(tmp_host.data(), tmp_dev, N * sizeof(int), cudaMemcpyDefault);

  for (int i = 0; i < N; ++i) {
    if (tmp_host[i] != vrev[i]) {
      std::printf("Error: d[%d] != arev[%d] (%d, %d)\n", i, i, tmp_host[i],
                  vrev[i]);
    }
  }

  cudaMemcpy(tmp_dev, v.data(), N * sizeof(int), cudaMemcpyDefault);
  dynamic_reverse<<<1, N, N * sizeof(int)>>>(tmp_dev);
  cudaMemcpy(tmp_host.data(), tmp_dev, N * sizeof(int), cudaMemcpyDefault);

  for (int i = 0; i < N; ++i) {
    if (tmp_host[i] != vrev[i]) {
      std::printf("Error: d[%d] != arev[%d] (%d, %d)\n", i, i, tmp_host[i],
                  vrev[i]);
    }
  }

  cudaFree(tmp_dev);

  return 0;
}
