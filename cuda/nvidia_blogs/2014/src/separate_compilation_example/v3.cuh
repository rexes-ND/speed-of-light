#pragma once

class v3 {
public:
  float x;
  float y;
  float z;

  v3();
  v3(float x, float y, float z);
  void randomize();
  __host__ __device__ void normalize();
  __host__ __device__ void scramble();
};
