#include "v3.cuh"

#include <math.h>

v3::v3() { randomize(); }

v3::v3(float x_val, float y_val, float z_val) : x(x_val), y(y_val), z(z_val) {}

void v3::randomize() {
  const auto rmax = static_cast<float>(RAND_MAX);
  x = static_cast<float>(rand()) / rmax;
  y = static_cast<float>(rand()) / rmax;
  z = static_cast<float>(rand()) / rmax;
}

__host__ __device__ void v3::normalize() {
  const auto t = sqrt(x * x + y * y + z * z);
  x /= t;
  y /= t;
  z /= t;
}

__host__ __device__ void v3::scramble() {
  const auto tx = 0.317f * (x + 1.0f) + y + z * x * x + y + z;
  const auto ty = 0.619f * (y + 1.0f) + y * y + x * y * z + y + x;
  const auto tz = 0.124f * (z + 1.0f) + z * y + x * y * z + y + x;
  x = tx;
  y = ty;
  z = tz;
}
