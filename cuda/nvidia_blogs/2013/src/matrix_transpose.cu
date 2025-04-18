/*
  Link: https://developer.nvidia.com/blog/efficient-matrix-transpose-cuda-cc/
*/

#include <cassert>
#include <iostream>

#include <cuda_runtime_api.h>

using uint = unsigned int;

inline cudaError_t checkCuda(cudaError_t result) {
  if (result != cudaSuccess) {
    std::fprintf(stderr, "CUDA Runtime Error: %s\n",
                 cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
  return result;
}

constexpr uint TILE_DIM{32};
constexpr uint BLOCK_ROWS{8};
constexpr uint NUM_REPS{100};

// Check errors and print GB/s
void postprocess(const float *ref, const float *res, uint n, float ms) {
  bool passed{true};
  for (int i = 0; i < n; ++i) {
    if (res[i] != ref[i]) {
      std::printf("%d %f %f\n", i, res[i], ref[i]);
      std::printf("%25s\n", "*** FAILED ***");
      passed = false;
      break;
    }
  }
  if (passed)
    std::printf("%20.2f\n", 2 * n * sizeof(float) * 1e-6 * NUM_REPS / ms);
}

// simple copy kernel
// Used as reference case representing best effective bandwidth
__global__ void copy(float *odata, const float *idata) {
  const uint x{blockIdx.x * TILE_DIM + threadIdx.x};
  const uint y{blockIdx.y * TILE_DIM + threadIdx.y};
  const uint width{gridDim.x * TILE_DIM};

  for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
    odata[(y + j) * width + x] = idata[(y + j) * width + x];
}

// copy kernel using shared memory
// Also used as reference case, demonstrating effect of using shared memory.
__global__ void copySharedMem(float *odata, const float *idata) {
  __shared__ float tile[TILE_DIM * TILE_DIM];

  const uint x{blockIdx.x * TILE_DIM + threadIdx.x};
  const uint y{blockIdx.y * TILE_DIM + threadIdx.y};
  const uint width{gridDim.x * TILE_DIM};

  for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
    tile[(threadIdx.y + j) * TILE_DIM + threadIdx.x] =
        idata[(y + j) * width + x];

  // Included to mimic the transpose behavior.
  __syncthreads();

  for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
    odata[(y + j) * width + x] =
        tile[(threadIdx.y + j) * TILE_DIM + threadIdx.x];
}

// naive transpose
// Simplest transpose; doesn't use shared memory.
// Global memory reads are coalesced but writes are not.
__global__ void transposeNaive(float *odata, const float *idata) {
  const uint x{blockIdx.x * TILE_DIM + threadIdx.x};
  const uint y{blockIdx.y * TILE_DIM + threadIdx.y};
  const uint width{gridDim.x * TILE_DIM};

  for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
    odata[x * width + (y + j)] = idata[(y + j) * width + x];
}

// coalesced transpose
// Uses shared memory to achieve coalescing in both reads and writes
// Tile width == #banks causes shared memory bank conflicts.
__global__ void transposeCoalesced(float *odata, const float *idata) {
  __shared__ float tile[TILE_DIM][TILE_DIM];

  uint x{blockIdx.x * TILE_DIM + threadIdx.x};
  uint y{blockIdx.y * TILE_DIM + threadIdx.y};
  const uint width{gridDim.x * TILE_DIM};

  for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
    tile[threadIdx.y + j][threadIdx.x] = idata[(y + j) * width + x];

  __syncthreads();

  x = blockIdx.y * TILE_DIM + threadIdx.x;
  y = blockIdx.x * TILE_DIM + threadIdx.y;

  for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
    odata[(y + j) * width + x] = tile[threadIdx.x][threadIdx.y + j];
}

// No bank-conflict transpose
// Same as transposeCoalesced except the first tile dimension is padded
// to avoid shared memory bank conflicts.
__global__ void transposeNoBankConflicts(float *odata, const float *idata) {
  __shared__ float tile[TILE_DIM][TILE_DIM + 1];

  uint x{blockIdx.x * TILE_DIM + threadIdx.x};
  uint y{blockIdx.y * TILE_DIM + threadIdx.y};
  const uint width{gridDim.x * TILE_DIM};

  for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
    tile[threadIdx.y + j][threadIdx.x] = idata[(y + j) * width + x];

  __syncthreads();

  x = blockIdx.y * TILE_DIM + threadIdx.x;
  y = blockIdx.x * TILE_DIM + threadIdx.y;

  for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
    odata[(y + j) * width + x] = tile[threadIdx.x][threadIdx.y + j];
}

int main(int argc, char *argv[]) {
  constexpr uint N{1024};
  constexpr uint mem_size{N * N * sizeof(float)};

  const dim3 dimGrid(N / TILE_DIM, N / TILE_DIM);
  const dim3 dimBlock(TILE_DIM, BLOCK_ROWS);

  const int devId{argc > 1 ? atoi(argv[1]) : 0};

  cudaDeviceProp prop;
  checkCuda(cudaGetDeviceProperties(&prop, devId));
  std::printf("\nDevice : %s\n", prop.name);
  std::printf("Matrix size: %d %d, Block size: %d %d, Tile size: %d %d\n", N, N,
              TILE_DIM, BLOCK_ROWS, TILE_DIM, TILE_DIM);
  std::printf("dimGrid: %d %d %d. dimBlock %d %d %d\n", dimGrid.x, dimGrid.y,
              dimGrid.z, dimBlock.x, dimBlock.y, dimBlock.z);
  checkCuda(cudaSetDevice(devId));

  float *host_input_data{new float[N * N]};
  float *host_copy_data{new float[N * N]};
  float *host_transpose_data{new float[N * N]};
  float *gold{new float[N * N]};

  float *dev_input_data, *dev_copy_data, *dev_transpose_data;
  checkCuda(cudaMalloc(&dev_input_data, mem_size));
  checkCuda(cudaMalloc(&dev_copy_data, mem_size));
  checkCuda(cudaMalloc(&dev_transpose_data, mem_size));

  // check parameters and calculate execution configuration
  static_assert(N % TILE_DIM == 0);
  static_assert(TILE_DIM % BLOCK_ROWS == 0);

  for (int i = 0; i < N; ++i)
    for (int j = 0; j < N; ++j) {
      const auto tmp{i * N + j};
      host_input_data[i * N + j] = tmp;
      gold[j * N + i] = tmp;
    }

  checkCuda(cudaMemcpy(dev_input_data, host_input_data, mem_size,
                       cudaMemcpyHostToDevice));

  float ms{};
  cudaEvent_t startEvent, stopEvent;
  checkCuda(cudaEventCreate(&startEvent));
  checkCuda(cudaEventCreate(&stopEvent));

  // copy
  std::printf("%25s", "copy");
  checkCuda(cudaMemset(dev_copy_data, 0, mem_size));
  copy<<<dimGrid, dimBlock>>>(dev_copy_data, dev_input_data);
  checkCuda(cudaEventRecord(startEvent, 0));
  for (int i = 0; i < NUM_REPS; ++i)
    copy<<<dimGrid, dimBlock>>>(dev_copy_data, dev_input_data);
  checkCuda(cudaEventRecord(stopEvent, 0));
  checkCuda(cudaEventSynchronize(stopEvent));
  checkCuda(cudaEventElapsedTime(&ms, startEvent, stopEvent));
  checkCuda(cudaMemcpy(host_copy_data, dev_copy_data, mem_size,
                       cudaMemcpyDeviceToHost));
  postprocess(host_input_data, host_copy_data, N * N, ms);

  // copySharedMem
  std::printf("%25s", "shared memory copy");
  checkCuda(cudaMemset(dev_copy_data, 0, mem_size));
  copySharedMem<<<dimGrid, dimBlock>>>(dev_copy_data, dev_input_data);
  checkCuda(cudaEventRecord(startEvent, 0));
  for (int i = 0; i < NUM_REPS; ++i)
    copySharedMem<<<dimGrid, dimBlock>>>(dev_copy_data, dev_input_data);
  checkCuda(cudaEventRecord(stopEvent, 0));
  checkCuda(cudaEventSynchronize(stopEvent));
  checkCuda(cudaEventElapsedTime(&ms, startEvent, stopEvent));
  checkCuda(cudaMemcpy(host_copy_data, dev_copy_data, mem_size,
                       cudaMemcpyDeviceToHost));
  postprocess(host_input_data, host_copy_data, N * N, ms);

  // transposeNaive
  std::printf("%25s", "naive transpose");
  checkCuda(cudaMemset(dev_transpose_data, 0, mem_size));
  transposeNaive<<<dimGrid, dimBlock>>>(dev_transpose_data, dev_input_data);
  checkCuda(cudaEventRecord(startEvent, 0));
  for (int i = 0; i < NUM_REPS; ++i)
    transposeNaive<<<dimGrid, dimBlock>>>(dev_transpose_data, dev_input_data);
  checkCuda(cudaEventRecord(stopEvent, 0));
  checkCuda(cudaEventSynchronize(stopEvent));
  checkCuda(cudaEventElapsedTime(&ms, startEvent, stopEvent));
  checkCuda(cudaMemcpy(host_transpose_data, dev_transpose_data, mem_size,
                       cudaMemcpyDeviceToHost));
  postprocess(gold, host_transpose_data, N * N, ms);

  // transposeCoalesced
  std::printf("%25s", "coalesced tranpose");
  checkCuda(cudaMemset(dev_transpose_data, 0, mem_size));
  transposeCoalesced<<<dimGrid, dimBlock>>>(dev_transpose_data, dev_input_data);
  checkCuda(cudaEventRecord(startEvent, 0));
  for (int i = 0; i < NUM_REPS; ++i)
    transposeCoalesced<<<dimGrid, dimBlock>>>(dev_transpose_data,
                                              dev_input_data);
  checkCuda(cudaEventRecord(stopEvent, 0));
  checkCuda(cudaEventSynchronize(stopEvent));
  checkCuda(cudaEventElapsedTime(&ms, startEvent, stopEvent));
  checkCuda(cudaMemcpy(host_transpose_data, dev_transpose_data, mem_size,
                       cudaMemcpyDeviceToHost));
  postprocess(gold, host_transpose_data, N * N, ms);

  // transposeNoBankConflicts
  std::printf("%25s", "conflict-free transpose");
  checkCuda(cudaMemset(dev_transpose_data, 0, mem_size));
  transposeNoBankConflicts<<<dimGrid, dimBlock>>>(dev_transpose_data,
                                                  dev_input_data);
  checkCuda(cudaEventRecord(startEvent, 0));
  for (int i = 0; i < NUM_REPS; ++i)
    transposeNoBankConflicts<<<dimGrid, dimBlock>>>(dev_transpose_data,
                                                    dev_input_data);
  checkCuda(cudaEventRecord(stopEvent, 0));
  checkCuda(cudaEventSynchronize(stopEvent));
  checkCuda(cudaEventElapsedTime(&ms, startEvent, stopEvent));
  checkCuda(cudaMemcpy(host_transpose_data, dev_transpose_data, mem_size,
                       cudaMemcpyDeviceToHost));
  postprocess(gold, host_transpose_data, N * N, ms);

  /*
    Device : NVIDIA GeForce RTX 4090
    Matrix size: 1024 1024, Block size: 32 8, Tile size: 32 32
    dimGrid: 32 32 1. dimBlock 32 8 1
                        copy             2220.05
          shared memory copy             2214.05
              naive transpose              266.58
          coalesced tranpose             1376.81
      conflict-free transpose             2214.05
  */

  delete[] host_input_data;
  delete[] host_copy_data;
  delete[] host_transpose_data;
  delete[] gold;
  checkCuda(cudaFree(dev_input_data));
  checkCuda(cudaFree(dev_copy_data));
  checkCuda(cudaFree(dev_transpose_data));

  return 0;
}
