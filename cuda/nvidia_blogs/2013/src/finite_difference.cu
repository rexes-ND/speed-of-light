#include <math.h>

#include <cstdlib>
#include <iostream>
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

constexpr auto fx = 1.0f;
constexpr auto fy = 1.0f;
constexpr auto fz = 1.0f;
constexpr auto mx = 64;
constexpr auto my = 64;
constexpr auto mz = 64;

// shared memory tiles will be m*-by-*Pencils
// sPencils is used when each thread calculates the derivative at one point
// lPencils is used for coalescing in y and z where each thread has to
//     calculate the derivative at mutiple points
constexpr auto s_pencils = 4;  // small # pencils
constexpr auto l_pencils = 32; // large # pencils

dim3 grid[3][2];
dim3 block[3][2];

// stencil coefficients
__constant__ float c_ax, c_bx, c_cx, c_dx;
__constant__ float c_ay, c_by, c_cy, c_dy;
__constant__ float c_az, c_bz, c_cz, c_dz;

// host routine to set constant data
void set_derivative_parameters() {
  // check to make sure dimensions are integral multiples of sPencils
  if ((mx % s_pencils != 0) || (my % s_pencils != 0) || (mz % s_pencils != 0)) {
    std::cout << "'mx', 'my', and 'mz' must be integral multiples of sPencils"
              << std::endl;
    std::exit(1);
  }

  if ((mx % l_pencils != 0) || (my % l_pencils != 0)) {
    std::cout << "'mx' and 'my' must be multiples of lPencils" << std::endl;
    std::exit(1);
  }

  // stencil weights (for unit length problem)
  float dsinv = mx - 1.0f;

  const auto ax = 4.0f / 5.0f * dsinv;
  const auto bx = -1.0f / 5.0f * dsinv;
  const auto cx = 4.0f / 105.0f * dsinv;
  const auto dx = -1.0f / 280.0f * dsinv;
  CHECK_CUDA_ERROR(
      cudaMemcpyToSymbol(c_ax, &ax, sizeof(float), 0, cudaMemcpyDefault));
  CHECK_CUDA_ERROR(
      cudaMemcpyToSymbol(c_bx, &bx, sizeof(float), 0, cudaMemcpyDefault));
  CHECK_CUDA_ERROR(
      cudaMemcpyToSymbol(c_cx, &cx, sizeof(float), 0, cudaMemcpyDefault));
  CHECK_CUDA_ERROR(
      cudaMemcpyToSymbol(c_dx, &dx, sizeof(float), 0, cudaMemcpyDefault));

  dsinv = my - 1.0f;

  const auto ay = 4.0f / 5.0f * dsinv;
  const auto by = -1.0f / 5.0f * dsinv;
  const auto cy = 4.0f / 105.0f * dsinv;
  const auto dy = -1.0f / 280.0f * dsinv;
  CHECK_CUDA_ERROR(
      cudaMemcpyToSymbol(c_ay, &ay, sizeof(float), 0, cudaMemcpyDefault));
  CHECK_CUDA_ERROR(
      cudaMemcpyToSymbol(c_by, &by, sizeof(float), 0, cudaMemcpyDefault));
  CHECK_CUDA_ERROR(
      cudaMemcpyToSymbol(c_cy, &cy, sizeof(float), 0, cudaMemcpyDefault));
  CHECK_CUDA_ERROR(
      cudaMemcpyToSymbol(c_dy, &dy, sizeof(float), 0, cudaMemcpyDefault));

  dsinv = mz - 1.f;

  const auto az = 4.0f / 5.0f * dsinv;
  const auto bz = -1.0f / 5.0f * dsinv;
  const auto cz = 4.0f / 105.0f * dsinv;
  const auto dz = -1.0f / 280.0f * dsinv;
  CHECK_CUDA_ERROR(
      cudaMemcpyToSymbol(c_az, &az, sizeof(float), 0, cudaMemcpyDefault));
  CHECK_CUDA_ERROR(
      cudaMemcpyToSymbol(c_bz, &bz, sizeof(float), 0, cudaMemcpyDefault));
  CHECK_CUDA_ERROR(
      cudaMemcpyToSymbol(c_cz, &cz, sizeof(float), 0, cudaMemcpyDefault));
  CHECK_CUDA_ERROR(
      cudaMemcpyToSymbol(c_dz, &dz, sizeof(float), 0, cudaMemcpyDefault));

  // Execution configurations for small and large pencil tiles

  grid[0][0] = dim3(my / s_pencils, mz, 1);
  block[0][0] = dim3(mx, s_pencils, 1);

  grid[0][1] = dim3(my / l_pencils, mz, 1);
  block[0][1] = dim3(mx, s_pencils, 1);

  grid[1][0] = dim3(mx / s_pencils, mz, 1);
  block[1][0] = dim3(s_pencils, my, 1);

  grid[1][1] = dim3(mx / l_pencils, mz, 1);
  // we want to use the same number of threads as above,
  // so when we use lPencils instead of sPencils in one
  // dimension, we multiply the other by sPencils/lPencils
  block[1][1] = dim3(l_pencils, my * s_pencils / l_pencils, 1);

  grid[2][0] = dim3(mx / s_pencils, my, 1);
  block[2][0] = dim3(s_pencils, mz, 1);

  grid[2][1] = dim3(mx / l_pencils, my, 1);
  block[2][1] = dim3(l_pencils, mz * s_pencils / l_pencils, 1);
}

void init_input(float *f, int dim) {
  const auto twopi = 8.0f * atanf(1.0);

  for (int k = 0; k < mz; k++) {
    for (int j = 0; j < my; j++) {
      for (int i = 0; i < mx; i++) {
        switch (dim) {
        case 0:
          f[k * mx * my + j * mx + i] =
              cos(fx * twopi * (i - 1.f) / (mx - 1.f));
          break;
        case 1:
          f[k * mx * my + j * mx + i] =
              cos(fy * twopi * (j - 1.f) / (my - 1.f));
          break;
        case 2:
          f[k * mx * my + j * mx + i] =
              cos(fz * twopi * (k - 1.f) / (mz - 1.f));
          break;
        }
      }
    }
  }
}

void init_sol(float *sol, int dim) {
  const auto twopi = 8.0f * atanf(1.0);

  for (int k = 0; k < mz; k++) {
    for (int j = 0; j < my; j++) {
      for (int i = 0; i < mx; i++) {
        switch (dim) {
        case 0:
          sol[k * mx * my + j * mx + i] =
              -fx * twopi * sin(fx * twopi * (i - 1.f) / (mx - 1.f));
          break;
        case 1:
          sol[k * mx * my + j * mx + i] =
              -fy * twopi * sin(fy * twopi * (j - 1.f) / (my - 1.f));
          break;
        case 2:
          sol[k * mx * my + j * mx + i] =
              -fz * twopi * sin(fz * twopi * (k - 1.f) / (mz - 1.f));
          break;
        }
      }
    }
  }
}

void check_results(double &error, double &max_error, float *sol, float *df) {
  // error = sqrt(sum((sol-df)**2)/(mx*my*mz))
  // maxError = maxval(abs(sol-df))
  max_error = 0;
  error = 0;
  for (int k = 0; k < mz; k++) {
    for (int j = 0; j < my; j++) {
      for (int i = 0; i < mx; i++) {
        float s = sol[k * mx * my + j * mx + i];
        float f = df[k * mx * my + j * mx + i];
        // printf("%d %d %d: %f %f\n", i, j, k, s, f);
        error += (s - f) * (s - f);
        if (fabs(s - f) > max_error)
          max_error = fabs(s - f);
      }
    }
  }
  error = std::sqrt(error / (mx * my * mz));
}

// -------------
// x derivatives
// -------------

__global__ void derivative_x(const float *f, float *df) {
  __shared__ float s_f[s_pencils][mx + 8]; // 4-wide halo

  const auto i = threadIdx.x;
  const auto j = blockIdx.x * blockDim.y + threadIdx.y;
  const auto k = blockIdx.y;
  const auto si = i + 4;
  const auto sj = threadIdx.y;

  const auto idx = k * mx * my + j * mx + i;

  s_f[sj][si] = f[idx];

  __syncthreads();

  // fill in periodic images in shared memory array
  if (i < 4) {
    s_f[sj][si - 4] = s_f[sj][si + mx - 5];
    s_f[sj][si + mx] = s_f[sj][si + 1];
  }

  __syncthreads();

  df[idx] = (c_ax * (s_f[sj][si + 1] - s_f[sj][si - 1]) +
             c_bx * (s_f[sj][si + 2] - s_f[sj][si - 2]) +
             c_cx * (s_f[sj][si + 3] - s_f[sj][si - 3]) +
             c_dx * (s_f[sj][si + 4] - s_f[sj][si - 4]));
}

// this version uses a 64x32 shared memory tile,
// still with 64*sPencils threads

__global__ void derivative_x_l_pencils(const float *f, float *df) {
  __shared__ float s_f[l_pencils][mx + 8]; // 4-wide halo

  int i = threadIdx.x;
  int jBase = blockIdx.x * l_pencils;
  int k = blockIdx.y;
  int si = i + 4; // local i for shared memory access + halo offset

  for (int sj = threadIdx.y; sj < l_pencils; sj += blockDim.y) {
    int globalIdx = k * mx * my + (jBase + sj) * mx + i;
    s_f[sj][si] = f[globalIdx];
  }

  __syncthreads();

  // fill in periodic images in shared memory array
  if (i < 4) {
    for (int sj = threadIdx.y; sj < l_pencils; sj += blockDim.y) {
      s_f[sj][si - 4] = s_f[sj][si + mx - 5];
      s_f[sj][si + mx] = s_f[sj][si + 1];
    }
  }

  __syncthreads();

  for (int sj = threadIdx.y; sj < l_pencils; sj += blockDim.y) {
    int globalIdx = k * mx * my + (jBase + sj) * mx + i;
    df[globalIdx] = (c_ax * (s_f[sj][si + 1] - s_f[sj][si - 1]) +
                     c_bx * (s_f[sj][si + 2] - s_f[sj][si - 2]) +
                     c_cx * (s_f[sj][si + 3] - s_f[sj][si - 3]) +
                     c_dx * (s_f[sj][si + 4] - s_f[sj][si - 4]));
  }
}

// -------------
// y derivatives
// -------------

__global__ void derivative_y(const float *f, float *df) {
  __shared__ float s_f[my + 8][s_pencils];

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = threadIdx.y;
  int k = blockIdx.y;
  int si = threadIdx.x;
  int sj = j + 4;

  int globalIdx = k * mx * my + j * mx + i;

  s_f[sj][si] = f[globalIdx];

  __syncthreads();

  if (j < 4) {
    s_f[sj - 4][si] = s_f[sj + my - 5][si];
    s_f[sj + my][si] = s_f[sj + 1][si];
  }

  __syncthreads();

  df[globalIdx] = (c_ay * (s_f[sj + 1][si] - s_f[sj - 1][si]) +
                   c_by * (s_f[sj + 2][si] - s_f[sj - 2][si]) +
                   c_cy * (s_f[sj + 3][si] - s_f[sj - 3][si]) +
                   c_dy * (s_f[sj + 4][si] - s_f[sj - 4][si]));
}

// y derivative using a tile of 32x64,
// launch with thread block of 32x8
__global__ void derivative_y_l_pencils(const float *f, float *df) {
  __shared__ float s_f[my + 8][l_pencils];

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int k = blockIdx.y;
  int si = threadIdx.x;

  for (int j = threadIdx.y; j < my; j += blockDim.y) {
    int globalIdx = k * mx * my + j * mx + i;
    int sj = j + 4;
    s_f[sj][si] = f[globalIdx];
  }

  __syncthreads();

  int sj = threadIdx.y + 4;
  if (sj < 8) {
    s_f[sj - 4][si] = s_f[sj + my - 5][si];
    s_f[sj + my][si] = s_f[sj + 1][si];
  }

  __syncthreads();

  for (int j = threadIdx.y; j < my; j += blockDim.y) {
    int globalIdx = k * mx * my + j * mx + i;
    int sj = j + 4;
    df[globalIdx] = (c_ay * (s_f[sj + 1][si] - s_f[sj - 1][si]) +
                     c_by * (s_f[sj + 2][si] - s_f[sj - 2][si]) +
                     c_cy * (s_f[sj + 3][si] - s_f[sj - 3][si]) +
                     c_dy * (s_f[sj + 4][si] - s_f[sj - 4][si]));
  }
}

// ------------
// z derivative
// ------------

__global__ void derivative_z(const float *f, float *df) {
  __shared__ float s_f[mz + 8][s_pencils];

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y;
  int k = threadIdx.y;
  int si = threadIdx.x;
  int sk = k + 4; // halo offset

  int globalIdx = k * mx * my + j * mx + i;

  s_f[sk][si] = f[globalIdx];

  __syncthreads();

  if (k < 4) {
    s_f[sk - 4][si] = s_f[sk + mz - 5][si];
    s_f[sk + mz][si] = s_f[sk + 1][si];
  }

  __syncthreads();

  df[globalIdx] = (c_az * (s_f[sk + 1][si] - s_f[sk - 1][si]) +
                   c_bz * (s_f[sk + 2][si] - s_f[sk - 2][si]) +
                   c_cz * (s_f[sk + 3][si] - s_f[sk - 3][si]) +
                   c_dz * (s_f[sk + 4][si] - s_f[sk - 4][si]));
}

__global__ void derivative_z_l_pencils(const float *f, float *df) {
  __shared__ float s_f[mz + 8][l_pencils];

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y;
  int si = threadIdx.x;

  for (int k = threadIdx.y; k < mz; k += blockDim.y) {
    int globalIdx = k * mx * my + j * mx + i;
    int sk = k + 4;
    s_f[sk][si] = f[globalIdx];
  }

  __syncthreads();

  int k = threadIdx.y + 4;
  if (k < 8) {
    s_f[k - 4][si] = s_f[k + mz - 5][si];
    s_f[k + mz][si] = s_f[k + 1][si];
  }

  __syncthreads();

  for (int k = threadIdx.y; k < mz; k += blockDim.y) {
    int globalIdx = k * mx * my + j * mx + i;
    int sk = k + 4;
    df[globalIdx] = (c_az * (s_f[sk + 1][si] - s_f[sk - 1][si]) +
                     c_bz * (s_f[sk + 2][si] - s_f[sk - 2][si]) +
                     c_cz * (s_f[sk + 3][si] - s_f[sk - 3][si]) +
                     c_dz * (s_f[sk + 4][si] - s_f[sk - 4][si]));
  }
}

// Run the kernels for a given dimension. One for sPencils, one for lPencils
void run_test(int dimension) {
  void (*fp_deriv[2])(const float *, float *);

  switch (dimension) {
  case 0:
    fp_deriv[0] = derivative_x;
    fp_deriv[1] = derivative_x_l_pencils;
    break;
  case 1:
    fp_deriv[0] = derivative_y;
    fp_deriv[1] = derivative_y_l_pencils;
    break;
  case 2:
    fp_deriv[0] = derivative_z;
    fp_deriv[1] = derivative_z_l_pencils;
    break;
  }

  // dimension x {0: small, 1: large} x {shared memory dimensions}
  int shared_dims[3][2][2] = {s_pencils, mx, l_pencils, mx, s_pencils, my,
                              l_pencils, my, s_pencils, mz, l_pencils, mz};

  std::vector<float> f_host(mx * my * mz);
  std::vector<float> df_host(mx * my * mz);
  std::vector<float> sol_host(mx * my * mz);

  init_input(f_host.data(), dimension);
  init_sol(sol_host.data(), dimension);

  // device arrays
  const auto bytes = mx * my * mz * sizeof(float);
  float *f_dev, *df_dev;
  CHECK_CUDA_ERROR(cudaMalloc(&f_dev, bytes));
  CHECK_CUDA_ERROR(cudaMalloc(&df_dev, bytes));

  const auto num_reps = 20;
  float ms;
  cudaEvent_t start_event, stop_event;
  CHECK_CUDA_ERROR(cudaEventCreate(&start_event));
  CHECK_CUDA_ERROR(cudaEventCreate(&stop_event));

  double error, max_error;

  std::cout << static_cast<char>('X' + dimension) << " derivatives" << std::endl
            << std::endl;

  for (int fp = 0; fp < 2; fp++) {
    CHECK_CUDA_ERROR(
        cudaMemcpy(f_dev, f_host.data(), bytes, cudaMemcpyDefault));
    CHECK_CUDA_ERROR(cudaMemset(df_dev, 0, bytes));

    fp_deriv[fp]<<<grid[dimension][fp], block[dimension][fp]>>>(f_dev, df_dev);
    CHECK_CUDA_ERROR(cudaEventRecord(start_event, 0));
    for (int i = 0; i < num_reps; i++)
      fp_deriv[fp]<<<grid[dimension][fp], block[dimension][fp]>>>(f_dev,
                                                                  df_dev);

    CHECK_CUDA_ERROR(cudaEventRecord(stop_event, 0));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop_event));
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&ms, start_event, stop_event));

    CHECK_CUDA_ERROR(
        cudaMemcpy(df_host.data(), df_dev, bytes, cudaMemcpyDefault));

    check_results(error, max_error, sol_host.data(), df_host.data());

    std::printf("  Using shared memory tile of %d x %d\n",
                shared_dims[dimension][fp][0], shared_dims[dimension][fp][1]);
    std::printf("   RMS error: %e\n", error);
    std::printf("   MAX error: %e\n", max_error);
    std::printf("   Average time (ms): %f\n", ms / num_reps);
    std::printf("   Average Bandwidth (GB/s): %f\n\n",
                2.0f * 1e-6 * mx * my * mz * num_reps * sizeof(float) / ms);
  }

  CHECK_CUDA_ERROR(cudaEventDestroy(start_event));
  CHECK_CUDA_ERROR(cudaEventDestroy(stop_event));

  CHECK_CUDA_ERROR(cudaFree(f_dev));
  CHECK_CUDA_ERROR(cudaFree(df_dev));
}

int main() {
  cudaDeviceProp prop;
  CHECK_CUDA_ERROR(cudaGetDeviceProperties(&prop, 0));
  std::cout << std::endl;
  std::cout << "Device Name: " << prop.name << std::endl;
  std::cout << "Compute Capability: " << prop.major << "." << prop.minor
            << std::endl
            << std::endl;

  set_derivative_parameters(); // initialize

  run_test(0); // x derivative
  run_test(1); // y derivative
  run_test(2); // z derivative

  return 0;
}
