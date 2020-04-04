#pragma once

#include "math.h"
#include "vector.h"

namespace gm {
class Ray {
 public:
  __device__ Ray() : tMax(INFINITY){};

  __device__ Ray(const Vector3f &origin, const Vector3f &direction,
                 float tMax = INFINITY)
      : origin(origin), direction(direction), tMax(tMax){};

  /// Returns a point along the ray using the given t value
  __device__ Vector3f operator()(float t) const {
    return origin + direction * t;
  }

  __device__ bool hasNans() const {
    return (origin.hasNans() || direction.hasNans() || isnan(tMax));
  }

  Vector3f origin;
  Vector3f direction;
  float tMax;
};
}  // namespace gm