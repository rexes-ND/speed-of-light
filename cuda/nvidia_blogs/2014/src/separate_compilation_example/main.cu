#include "particle.cuh"

#include <cstdio>
#include <cstdlib>
#include <vector>

__global__ void advance_particles(float dt, particle *particle_array,
                                  int num_particles) {
  const auto i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < num_particles)
    particle_array[i].advance(dt);
}

int main(int argc, char **argv) {
  int n = 1000000;

  if (argc > 1)
    // Number of particles
    n = atoi(argv[1]);
  if (argc > 2)
    // Random seed
    srand(atoi(argv[2]));

  std::vector<particle> particle_array(n);
  particle *particle_array_dev = nullptr;
  cudaMalloc(&particle_array_dev, n * sizeof(particle));
  cudaMemcpy(particle_array_dev, particle_array.data(), n * sizeof(particle),
             cudaMemcpyDefault);
  for (int i = 0; i < 100; ++i) { // Random distance each step
    const auto dt = static_cast<float>(rand()) / RAND_MAX;
    advance_particles<<<(n + 256 - 1) / 256, 256>>>(dt, particle_array_dev, n);
  }
  cudaMemcpy(particle_array.data(), particle_array_dev, n * sizeof(particle),
             cudaMemcpyDefault);

  v3 totalDistance(0, 0, 0);
  for (int i = 0; i < n; i++) {
    const auto &tmp = particle_array[i].getTotalDistance();
    totalDistance.x += tmp.x;
    totalDistance.y += tmp.y;
    totalDistance.z += tmp.z;
  }
  const auto avgX = totalDistance.x / n;
  const auto avgY = totalDistance.y / n;
  const auto avgZ = totalDistance.z / n;
  float avgNorm = sqrt(avgX * avgX + avgY * avgY + avgZ * avgZ);
  std::printf(
      "Moved %d particles 100 steps. Average distance traveled is |(%f, %f, "
      "%f)| = %f\n",
      n, avgX, avgY, avgZ, avgNorm);

  return 0;
}
