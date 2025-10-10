#include "particle.cuh"

particle::particle() : position(), velocity(), totalDistance(0, 0, 0) {}

__device__ __host__ void particle::advance(float d) {
  velocity.normalize();
  const auto dx = d * velocity.x;
  const auto dy = d * velocity.y;
  const auto dz = d * velocity.z;
  position.x += dx;
  position.y += dy;
  position.z += dz;
  totalDistance.x += dx;
  totalDistance.y += dy;
  totalDistance.z += dz;
  velocity.scramble();
}

const v3 &particle::getTotalDistance() const { return totalDistance; }
