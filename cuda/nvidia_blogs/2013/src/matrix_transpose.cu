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

constexpr auto TILE_DIM = 32U;
constexpr auto BLOCK_ROWS = 8U;
constexpr auto NUM_REPS = 100U;

// Check errors and print GB/s
void postprocess(const float *ref, const float *res, uint n, float ms) {
  auto passed = true;
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
__global__ void copy(float *out, const float *in) {
  const auto x = blockIdx.x * TILE_DIM + threadIdx.x;
  const auto y = blockIdx.y * TILE_DIM + threadIdx.y;
  const auto width = gridDim.x * TILE_DIM;

  for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
    out[(y + j) * width + x] = in[(y + j) * width + x];
}

// copy kernel using shared memory
// Also used as reference case, demonstrating effect of using shared memory.
__global__ void copySharedMem(float *out, const float *in) {
  __shared__ float tile[TILE_DIM * TILE_DIM];

  const auto x = blockIdx.x * TILE_DIM + threadIdx.x;
  const auto y = blockIdx.y * TILE_DIM + threadIdx.y;
  const auto width = gridDim.x * TILE_DIM;

  for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
    tile[(threadIdx.y + j) * TILE_DIM + threadIdx.x] = in[(y + j) * width + x];

  // Included to mimic the transpose behavior.
  __syncthreads();

  for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
    out[(y + j) * width + x] = tile[(threadIdx.y + j) * TILE_DIM + threadIdx.x];
}

// naive transpose
// Simplest transpose; doesn't use shared memory.
// Global memory reads are coalesced but writes are not.
__global__ void transposeNaive(float *out, const float *in) {
  const auto x = blockIdx.x * TILE_DIM + threadIdx.x;
  const auto y = blockIdx.y * TILE_DIM + threadIdx.y;
  const auto width = gridDim.x * TILE_DIM;

  for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
    out[x * width + (y + j)] = in[(y + j) * width + x];
}

// coalesced transpose
// Uses shared memory to achieve coalescing in both reads and writes
// Tile width == #banks causes shared memory bank conflicts.
__global__ void transposeCoalesced(float *out, const float *in) {
  __shared__ float tile[TILE_DIM][TILE_DIM];

  auto x = blockIdx.x * TILE_DIM + threadIdx.x;
  auto y = blockIdx.y * TILE_DIM + threadIdx.y;
  const auto width = gridDim.x * TILE_DIM;

  for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
    tile[threadIdx.y + j][threadIdx.x] = in[(y + j) * width + x];

  __syncthreads();

  x = blockIdx.y * TILE_DIM + threadIdx.x;
  y = blockIdx.x * TILE_DIM + threadIdx.y;

  for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
    out[(y + j) * width + x] = tile[threadIdx.x][threadIdx.y + j];
}

// No bank-conflict transpose
// Same as transposeCoalesced except the first tile dimension is padded
// to avoid shared memory bank conflicts.
__global__ void transposeNoBankConflicts(float *out, const float *in) {
  __shared__ float tile[TILE_DIM][TILE_DIM + 1];

  auto x = blockIdx.x * TILE_DIM + threadIdx.x;
  auto y = blockIdx.y * TILE_DIM + threadIdx.y;
  const auto width = gridDim.x * TILE_DIM;

  for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
    tile[threadIdx.y + j][threadIdx.x] = in[(y + j) * width + x];

  __syncthreads();

  x = blockIdx.y * TILE_DIM + threadIdx.x;
  y = blockIdx.x * TILE_DIM + threadIdx.y;

  for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
    out[(y + j) * width + x] = tile[threadIdx.x][threadIdx.y + j];
}

int main(int argc, char *argv[]) {
  constexpr auto N = 1024U;
  constexpr auto mem_size = N * N * sizeof(float);

  const dim3 dimGrid(N / TILE_DIM, N / TILE_DIM);
  const dim3 dimBlock(TILE_DIM, BLOCK_ROWS);

  const auto devId = argc > 1 ? std::stoi(argv[1]) : 0;

  cudaDeviceProp prop;
  CHECK_CUDA_ERROR(cudaGetDeviceProperties(&prop, devId));
  std::printf("\nDevice : %s\n", prop.name);
  std::printf("Matrix size: %d %d, Block size: %d %d, Tile size: %d %d\n", N, N,
              TILE_DIM, BLOCK_ROWS, TILE_DIM, TILE_DIM);
  std::printf("dimGrid: %d %d %d. dimBlock %d %d %d\n", dimGrid.x, dimGrid.y,
              dimGrid.z, dimBlock.x, dimBlock.y, dimBlock.z);
  CHECK_CUDA_ERROR(cudaSetDevice(devId));

  std::vector<float> host_input_data(N * N);
  std::vector<float> host_copy_data(N * N);
  std::vector<float> host_transpose_data(N * N);
  std::vector<float> host_transpose_ref_data(N * N);

  float *dev_input_data, *dev_copy_data, *dev_transpose_data;
  CHECK_CUDA_ERROR(cudaMalloc(&dev_input_data, mem_size));
  CHECK_CUDA_ERROR(cudaMalloc(&dev_copy_data, mem_size));
  CHECK_CUDA_ERROR(cudaMalloc(&dev_transpose_data, mem_size));

  // check parameters and calculate execution configuration
  static_assert(N % TILE_DIM == 0);
  static_assert(TILE_DIM % BLOCK_ROWS == 0);

  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      const auto tmp = i * N + j;
      host_input_data[i * N + j] = tmp;
      host_transpose_ref_data[j * N + i] = tmp;
    }
  }
  CHECK_CUDA_ERROR(cudaMemcpy(dev_input_data, host_input_data.data(), mem_size,
                              cudaMemcpyDefault));

  float ms;
  cudaEvent_t start_event, stop_event;
  CHECK_CUDA_ERROR(cudaEventCreate(&start_event));
  CHECK_CUDA_ERROR(cudaEventCreate(&stop_event));

  // copy
  std::printf("%25s", "copy");
  CHECK_CUDA_ERROR(cudaMemset(dev_copy_data, 0, mem_size));
  copy<<<dimGrid, dimBlock>>>(dev_copy_data, dev_input_data);
  CHECK_CUDA_ERROR(cudaEventRecord(start_event, 0));
  for (int i = 0; i < NUM_REPS; ++i)
    copy<<<dimGrid, dimBlock>>>(dev_copy_data, dev_input_data);
  CHECK_CUDA_ERROR(cudaEventRecord(stop_event, 0));
  CHECK_CUDA_ERROR(cudaEventSynchronize(stop_event));
  CHECK_CUDA_ERROR(cudaEventElapsedTime(&ms, start_event, stop_event));
  CHECK_CUDA_ERROR(cudaMemcpy(host_copy_data.data(), dev_copy_data, mem_size,
                              cudaMemcpyDefault));
  postprocess(host_input_data.data(), host_copy_data.data(), N * N, ms);

  // copySharedMem
  std::printf("%25s", "shared memory copy");
  CHECK_CUDA_ERROR(cudaMemset(dev_copy_data, 0, mem_size));
  copySharedMem<<<dimGrid, dimBlock>>>(dev_copy_data, dev_input_data);
  CHECK_CUDA_ERROR(cudaEventRecord(start_event, 0));
  for (int i = 0; i < NUM_REPS; ++i)
    copySharedMem<<<dimGrid, dimBlock>>>(dev_copy_data, dev_input_data);
  CHECK_CUDA_ERROR(cudaEventRecord(stop_event, 0));
  CHECK_CUDA_ERROR(cudaEventSynchronize(stop_event));
  CHECK_CUDA_ERROR(cudaEventElapsedTime(&ms, start_event, stop_event));
  CHECK_CUDA_ERROR(cudaMemcpy(host_copy_data.data(), dev_copy_data, mem_size,
                              cudaMemcpyDefault));
  postprocess(host_input_data.data(), host_copy_data.data(), N * N, ms);

  // transposeNaive
  std::printf("%25s", "naive transpose");
  CHECK_CUDA_ERROR(cudaMemset(dev_transpose_data, 0, mem_size));
  transposeNaive<<<dimGrid, dimBlock>>>(dev_transpose_data, dev_input_data);
  CHECK_CUDA_ERROR(cudaEventRecord(start_event, 0));
  for (int i = 0; i < NUM_REPS; ++i)
    transposeNaive<<<dimGrid, dimBlock>>>(dev_transpose_data, dev_input_data);
  CHECK_CUDA_ERROR(cudaEventRecord(stop_event, 0));
  CHECK_CUDA_ERROR(cudaEventSynchronize(stop_event));
  CHECK_CUDA_ERROR(cudaEventElapsedTime(&ms, start_event, stop_event));
  CHECK_CUDA_ERROR(cudaMemcpy(host_transpose_data.data(), dev_transpose_data,
                              mem_size, cudaMemcpyDefault));
  postprocess(host_transpose_ref_data.data(), host_transpose_data.data(), N * N,
              ms);

  // transposeCoalesced
  std::printf("%25s", "coalesced tranpose");
  CHECK_CUDA_ERROR(cudaMemset(dev_transpose_data, 0, mem_size));
  transposeCoalesced<<<dimGrid, dimBlock>>>(dev_transpose_data, dev_input_data);
  CHECK_CUDA_ERROR(cudaEventRecord(start_event, 0));
  for (int i = 0; i < NUM_REPS; ++i)
    transposeCoalesced<<<dimGrid, dimBlock>>>(dev_transpose_data,
                                              dev_input_data);
  CHECK_CUDA_ERROR(cudaEventRecord(stop_event, 0));
  CHECK_CUDA_ERROR(cudaEventSynchronize(stop_event));
  CHECK_CUDA_ERROR(cudaEventElapsedTime(&ms, start_event, stop_event));
  CHECK_CUDA_ERROR(cudaMemcpy(host_transpose_data.data(), dev_transpose_data,
                              mem_size, cudaMemcpyDefault));
  postprocess(host_transpose_ref_data.data(), host_transpose_data.data(), N * N,
              ms);

  // transposeNoBankConflicts
  std::printf("%25s", "conflict-free transpose");
  CHECK_CUDA_ERROR(cudaMemset(dev_transpose_data, 0, mem_size));
  transposeNoBankConflicts<<<dimGrid, dimBlock>>>(dev_transpose_data,
                                                  dev_input_data);
  CHECK_CUDA_ERROR(cudaEventRecord(start_event, 0));
  for (int i = 0; i < NUM_REPS; ++i)
    transposeNoBankConflicts<<<dimGrid, dimBlock>>>(dev_transpose_data,
                                                    dev_input_data);
  CHECK_CUDA_ERROR(cudaEventRecord(stop_event, 0));
  CHECK_CUDA_ERROR(cudaEventSynchronize(stop_event));
  CHECK_CUDA_ERROR(cudaEventElapsedTime(&ms, start_event, stop_event));
  CHECK_CUDA_ERROR(cudaMemcpy(host_transpose_data.data(), dev_transpose_data,
                              mem_size, cudaMemcpyDefault));
  postprocess(host_transpose_ref_data.data(), host_transpose_data.data(), N * N,
              ms);

  CHECK_CUDA_ERROR(cudaFree(dev_input_data));
  CHECK_CUDA_ERROR(cudaFree(dev_copy_data));
  CHECK_CUDA_ERROR(cudaFree(dev_transpose_data));
  CHECK_CUDA_ERROR(cudaEventDestroy(start_event));
  CHECK_CUDA_ERROR(cudaEventDestroy(stop_event));

  return 0;
}
