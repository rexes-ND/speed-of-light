#include <iostream>

__global__ void square(int *data, int len) {
  const auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  const auto val = data[idx];
  if (idx < len)
    data[idx] = val * val;
}

void launch_square(int *data, int len) {
  /*
    1. Choose a reasonable block size
    2. Calculates a theoretical maximum occupancy
  */

  int block_size, min_grid_size;

  // blockSizeLimit = 0 means no limit.
  cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, square, 0, 0);

  const auto grid_size = (len + block_size - 1) / block_size;
  square<<<grid_size, block_size>>>(data, len);
  cudaDeviceSynchronize();

  int max_active_blocks;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_active_blocks, square,
                                                block_size, 0);

  int device;
  cudaDeviceProp props;
  cudaGetDevice(&device);
  cudaGetDeviceProperties(&props, device);

  const auto occupancy =
      static_cast<double>(max_active_blocks * block_size / props.warpSize) /
      (props.maxThreadsPerMultiProcessor / props.warpSize);

  std::cout << "Launched blocks of size " << block_size
            << ". Theoretical occupancy: " << occupancy << std::endl;
}

int main() {
  int *data;
  const int len = 1024 * 1024;
  cudaMalloc(&data, len * sizeof(int));
  launch_square(data, len);

  return 0;
}
