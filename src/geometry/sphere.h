#pragma once

#include "shape.h"
#include "vector.h"

namespace gm {

/// An implicit representation of a sphere. The user merely has to define
/// the center and radius of the sphere to be able to use it. The intersection
/// test finds an analytical solution using a formulation that allows us to
/// apply the quadratic equation to find the roots t0 and t1, which are
/// possible intersection candidates.
class Sphere : public Shape {
 public:
  Sphere(const Vector3f &center, float radius)
      : center(center), radius(radius){};

  /// We use an analytical solution to solve for the intersection of a ray with
  /// a sphere. This involves solving the quadratic equation to find out if
  /// the ray passes through the boundaries of the sphere.
  bool intersect(
      const Ray &ray,
      const std::unique_ptr<Intersection> &intersection) const override {
    // Vector going from the ray origin to the center of the sphere
    Vector3f L = ray.origin - center;

    // Build parameters for the quadratic equation solver
    float a = dot(ray.direction, ray.direction);
    float b = 2.0f * dot(ray.direction, L);
    float c = dot(L, L) - radius * radius;
    float t0 = 0.0f;
    float t1 = 0.0f;
    if (!solveQuadratic(a, b, c, t0, t1)) {
      // No analytical solution exists so no intersection has been made
      return false;
    }

    if (t0 < 0.0f && t1 < 0.0f) {
      // Both solutions to the quadratic equation are negative so no
      // intersections have been made.
      return false;
    }

    // Check if t values need to be swapped before comparing
    if (t0 > t1) {
      // If t0 is negative we swap the values
      std::swap(t0, t1);
    }

    // t0 holds our desired t value, so we update our intersection data
    // accordingly
    intersection->surfacePoint = ray.origin + ray.direction * t0;
    intersection->tHit = t0;
    return true;
  }

  // Returns the surface area of the sphere defined by the radius
  float area() const override { return 4.0f * M_PI * radius * radius; }

 private:
  /// Function to solve the quadratic equation. Solving this equation is fairly
  /// costly since it involves using a square root and divide, but certain
  /// measures have been taken to use these functions only when needed.
  bool solveQuadratic(const float &a, const float &b, const float &c, float &x0,
                      float &x1) const {
    float discr = b * b - 4.0f * a * c;
    if (discr < 0.0f) {
      // The discriminant is negative so no real solution exists
      return false;
    } else if (discr == 0.0f) {
      x0 = x1 = -0.5f * b / a;
    } else {
      float q =
          (b > 0.0f) ? -0.5f * (b + sqrt(discr)) : -0.5f * (b - sqrt(discr));
      x0 = q / a;
      x1 = c / q;
    }
    return true;
  }

  Vector3f center;
  float radius;
};
}  // namespace gm