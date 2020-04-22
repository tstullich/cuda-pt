#pragma once

#include <cmath>
#include <iostream>

#include "vector.h"

/// A utility struct that holds intersection information.
namespace gm {
struct Intersection {
  Intersection() : surfacePoint(Vector3f(0.0f)), tHit(INFINITY) {}

  Intersection(const Vector3f &surfacePoint, const Vector3f &normal, float tHit)
      : surfacePoint(surfacePoint), normal(normal), tHit(tHit) {}

  Vector3f surfacePoint;
  Vector3f normal;
  float tHit;
  std::string name;
};
}  // namespace gm