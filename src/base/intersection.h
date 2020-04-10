#pragma once

#include <math.h>

#include "vector.h"

/// A utility struct that holds intersection information.
namespace gm {
struct Intersection {
  Intersection() : surfacePoint(Vector3f(0.0f)), tHit(INFINITY) {}

  Intersection(const Vector3f &surfacePoint, float tHit)
      : surfacePoint(surfacePoint), tHit(tHit) {}

  Vector3f surfacePoint;
  float tHit;
};
}  // namespace gm