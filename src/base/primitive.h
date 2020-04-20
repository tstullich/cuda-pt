#pragma once

#include "intersection.h"
#include "ray.h"
#include "vector.h"

namespace gm {
class Primitive {
 public:
  /// The method should return true if there exists an intersection along
  /// a ray. The ray needs to be transformed to the objects' local object
  /// space before testing!
  virtual bool intersect(
      const Ray &ray,
      const std::shared_ptr<Intersection> &intersection) const = 0;

  /// Returns the surface area of the shape. This will be useful later when
  /// sampling area lights.
  virtual float area() const = 0;

  /// Returns the probality distribution function. Important for multiple
  /// importance sampling later.
  virtual float pdf() const { return 1.0f / area(); }

  // Return the surface normal of the shape given the point on the surface
  virtual Vector3f normal(const Vector3f &surfacePoint) const = 0;
};
};  // namespace gm