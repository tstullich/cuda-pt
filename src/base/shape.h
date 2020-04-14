#pragma once

#include <memory>

#include "intersection.h"
#include "ray.h"
#include "vector.h"

/// Generic interface which needs to be implemented for all primitives
/// which need to be part of intersection testing.
namespace gm {
class Shape {
 public:
  /// The method should return true if there exists an intersection along
  /// a ray. TODO convert this to a hit record later
  virtual bool intersect(
      const Ray &ray,
      const std::unique_ptr<Intersection> &intersection) const = 0;

  /// Returns the surface area of the shape. This will be useful later when
  /// sampling area lights.
  virtual float area() const = 0;

  /// Returns the probality distribution function. Important for multiple
  /// importance sampling later.
  virtual float pdf() const { return 1.0f / area(); }

  // Return the surface normal of the shape given the point on the surface
  virtual Vector3f normal(const Vector3f &surfacePoint) const = 0;
};
}  // namespace gm