#include <iostream>
#include <vector>

#include <cuda_runtime_api.h>

__global__ void kernel_shfl_down(int *warp_data) {
  int i = threadIdx.x;
  int j = __shfl_down_sync(0xFFFFFFFF, i, 2);
  warp_data[i] = j;
}

void test_kernel_shfl_down() {
  int *warp_data_dev;
  cudaMalloc(&warp_data_dev, 32 * sizeof(int));
  kernel_shfl_down<<<1, 32>>>(warp_data_dev);

  std::vector<int> warp_data_host(32);
  cudaMemcpy(warp_data_host.data(), warp_data_dev, 32 * sizeof(int),
             cudaMemcpyDefault);
  for (int i = 0; i < 32; ++i) {
    if (i + 2 < 32 && warp_data_host[i] != i + 2)
      std::cout << "warp_data_host[" << i << "] != " << i + 2 << std::endl;
    else if (i + 2 >= 32 && warp_data_host[i] != i)
      std::cout << "warp_data_host[" << i << "] != " << i << std::endl;
  }
}

__global__ void kernel_warp_reduce_sum(int *warp_data) {
  int i = threadIdx.x;
  for (int offset = warpSize / 2; offset > 0; offset /= 2)
    i += __shfl_down_sync(0xFFFFFFFF, i, offset);
  warp_data[threadIdx.x] = i;
}

void test_kernel_warp_reduce_sum() {
  int *warp_data_dev;
  cudaMalloc(&warp_data_dev, 32 * sizeof(int));
  kernel_warp_reduce_sum<<<1, 32>>>(warp_data_dev);

  std::vector<int> warp_data_host(32);
  std::vector<int> warp_data_host_sol(32);
  cudaMemcpy(warp_data_host.data(), warp_data_dev, 32 * sizeof(int),
             cudaMemcpyDefault);
  for (int i = 0; i < 32; ++i)
    warp_data_host_sol[i] = i;
  for (int offset = 16; offset > 0; offset /= 2) {
    for (int i = 0; i < 32; ++i) {
      if (i + offset < 32)
        warp_data_host_sol[i] += warp_data_host_sol[i + offset];
      else
        warp_data_host_sol[i] += warp_data_host_sol[i];
    }
  }
  for (int i = 0; i < 32; ++i) {
    if (warp_data_host[i] != warp_data_host_sol[i]) {
      std::cout << "warp_data_host and warp_data_host_sol mismatched at " << i
                << std::endl;
    }
  }
}

__global__ void kernel_warp_reduce_xor_sum(int *warp_data) {
  int i = threadIdx.x;
  for (int offset = warpSize / 2; offset > 0; offset /= 2)
    i += __shfl_xor_sync(0xFFFFFFFF, i, offset);
  warp_data[threadIdx.x] = i;
}

void test_kernel_warp_reduce_xor_sum() {
  int *warp_data_dev;
  cudaMalloc(&warp_data_dev, 32 * sizeof(int));
  kernel_warp_reduce_xor_sum<<<1, 32>>>(warp_data_dev);

  std::vector<int> warp_data_host(32);
  std::vector<int> warp_data_host_sol(32);
  cudaMemcpy(warp_data_host.data(), warp_data_dev, 32 * sizeof(int),
             cudaMemcpyDefault);
  for (int i = 0; i < 32; ++i)
    warp_data_host_sol[i] = i;
  for (int offset = 16; offset > 0; offset /= 2) {
    std::vector<int> tmp(32);
    for (int i = 0; i < 32; ++i)
      tmp[i] = warp_data_host_sol[i] + warp_data_host_sol[i ^ offset];
    warp_data_host_sol = tmp;
  }
  for (int i = 0; i < 32; ++i) {
    if (warp_data_host[i] != warp_data_host_sol[i]) {
      std::cout << "warp_data_host and warp_data_host_sol mismatched at " << i
                << std::endl;
    }
  }
}

__inline__ __device__ int warp_reduce_sum(int val) {
  for (int offset = warpSize / 2; offset > 0; offset /= 2)
    val += __shfl_down_sync(0xFFFFFFFF, val, offset);
  return val;
}

__inline__ __device__ int block_reduce_sum(int val) {
  static __shared__ int smem[32];
  int lane_idx = threadIdx.x % warpSize;
  int warp_idx = threadIdx.x / warpSize;

  val = warp_reduce_sum(val);

  if (lane_idx == 0)
    smem[warp_idx] = val;

  __syncthreads();

  val = (threadIdx.x < blockDim.x / warpSize) ? smem[lane_idx] : 0;

  if (warp_idx == 0)
    val = warp_reduce_sum(val);

  return val;
}

__global__ void device_reduce_kernel(const int *in, int *out, int N) {
  int sum = 0;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N;
       i += blockDim.x * gridDim.x)
    sum += in[i];
  sum = block_reduce_sum(sum);
  if (threadIdx.x == 0)
    out[blockIdx.x] = sum;
}

void test_device_reduce_kernel() {
  int *in_dev, *out_dev;
  cudaMalloc(&in_dev, 1024 * 1024 * sizeof(int));
  cudaMalloc(&out_dev, 1024 * sizeof(int));

  std::vector<int> in_host(1024 * 1024, 1);
  cudaMemcpy(in_dev, in_host.data(), 1024 * 1024 * sizeof(int),
             cudaMemcpyDefault);
  device_reduce_kernel<<<1024, 512>>>(in_dev, out_dev, 1024 * 1024);
  device_reduce_kernel<<<1, 1024>>>(out_dev, out_dev, 1024);
  int out_host;
  cudaMemcpy(&out_host, out_dev, sizeof(int), cudaMemcpyDefault);
  if (out_host != 1024 * 1024)
    std::cout << "sum (" << out_host << ") != 1024 * 1024" << std::endl;
}

__global__ void device_reduce_warp_atomic_kernel(const int *in, int *out,
                                                 int N) {
  int sum = 0;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N;
       i += blockDim.x * gridDim.x)
    sum += in[i];
  sum = warp_reduce_sum(sum);
  if (threadIdx.x % warpSize == 0)
    atomicAdd(out, sum);
}

void test_device_reduce_warp_atomic_kernel() {
  int *in_dev, *out_dev;
  cudaMalloc(&in_dev, 1024 * 1024 * sizeof(int));
  cudaMalloc(&out_dev, sizeof(int));
  cudaMemset(out_dev, 0, sizeof(int));

  std::vector<int> in_host(1024 * 1024, 1);
  cudaMemcpy(in_dev, in_host.data(), 1024 * 1024 * sizeof(int),
             cudaMemcpyDefault);
  device_reduce_warp_atomic_kernel<<<1024, 512>>>(in_dev, out_dev, 1024 * 1024);
  int out_host;
  cudaMemcpy(&out_host, out_dev, sizeof(int), cudaMemcpyDefault);
  if (out_host != 1024 * 1024)
    std::cout << "sum (" << out_host << ") != 1024 * 1024" << std::endl;
}

__global__ void device_reduce_block_atomic_kernel(const int *in, int *out,
                                                  int N) {
  int sum = 0;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N;
       i += blockDim.x * gridDim.x)
    sum += in[i];
  sum = block_reduce_sum(sum);
  if (threadIdx.x == 0)
    atomicAdd(out, sum);
}

void test_device_reduce_block_atomic_kernel() {
  int *in_dev, *out_dev;
  cudaMalloc(&in_dev, 1024 * 1024 * sizeof(int));
  cudaMalloc(&out_dev, sizeof(int));
  cudaMemset(out_dev, 0, sizeof(int));

  std::vector<int> in_host(1024 * 1024, 1);
  cudaMemcpy(in_dev, in_host.data(), 1024 * 1024 * sizeof(int),
             cudaMemcpyDefault);
  device_reduce_block_atomic_kernel<<<1024, 512>>>(in_dev, out_dev,
                                                   1024 * 1024);
  int out_host;
  cudaMemcpy(&out_host, out_dev, sizeof(int), cudaMemcpyDefault);
  if (out_host != 1024 * 1024)
    std::cout << "sum (" << out_host << ") != 1024 * 1024" << std::endl;
}

int main() {
  test_kernel_shfl_down();
  test_kernel_warp_reduce_sum();
  test_kernel_warp_reduce_xor_sum();
  test_device_reduce_kernel();
  test_device_reduce_warp_atomic_kernel();
  test_device_reduce_block_atomic_kernel();

  return 0;
}
