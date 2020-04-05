#pragma once

#include <cuda.h>

#include "math.h"
#include "vector.h"

namespace gm {
class Ray {
 public:
  Ray() : tMax(INFINITY){};

  Ray(const Vector3f &origin, const Vector3f &direction, float tMax = INFINITY)
      : origin(origin), direction(direction), tMax(tMax){};

  /// Returns a point along the ray using the given t value
  Vector3f operator()(float t) const { return origin + direction * t; }

  bool hasNans() const {
    return (origin.hasNans() || direction.hasNans() || isnan(tMax));
  }

  Vector3f origin;
  Vector3f direction;
  float tMax;
};
}  // namespace gm