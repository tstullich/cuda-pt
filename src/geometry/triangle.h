#pragma once

#include <memory>

#include "shape.h"
#include "vector.h"

namespace gm {

/// A class that represents a single triangle. The intersection test is based on
/// the Möller–Trumbore algorithm:
/// https://en.wikipedia.org/wiki/M%C3%B6ller%E2%80%93Trumbore_intersection_algorithm
/// The assumption is made that the coordinates for a given triangle are in
/// world space which might change later on if we want to implement meshes that
/// have a local coordinate system.
class Triangle : public Shape {
 public:
  Triangle(const Vector3f &v0, const Vector3f &v1, const Vector3f &v2)
      : v0(v0), v1(v1), v2(v2){};

  /// The intersection test assumes that the coordinates are in world space,
  /// otherwise the intersection test will run into undefined behavior
  bool intersect(
      const Ray &ray,
      const std::unique_ptr<Intersection> &intersection) const override {
    Vector3f edge1, edge2, h, s, q;
    float a, f, u, v;
    edge1 = v1 - v0;
    edge2 = v2 - v0;
    h = cross(ray.direction, edge2);
    a = dot(edge1, h);
    if (a > -EPSILON && a < EPSILON) {
      return false;  // This ray is parallel to this triangle.
    }

    f = 1.0f / a;
    s = ray.origin - v0;
    u = f * dot(s, h);
    if (u < 0.0f || u > 1.0f) {
      return false;
    }

    q = cross(s, edge1);
    v = f * dot(ray.direction, q);
    if (v < 0.0f || u + v > 1.0f) {
      return false;
    }

    // At this stage we can compute t to find out where the intersection point
    // is on the line.
    float t = f * dot(edge2, q);
    if (t < EPSILON) {
      // This means that there is a line intersection but not a ray
      // intersection.
      return false;
    }

    // Update intersection info and return
    intersection->surfacePoint = ray.origin + ray.direction * t;
    intersection->tHit = t;
    return true;
  }

  /// Returns the surface area of the shape. This will be useful later when
  /// sampling area lights.
  float area() const override {
    return 0.5f * cross(v1 - v0, v2 - v0).length();
  }

  Vector3f normal(const Vector3f &surfacePoint) const override {
    return Vector3f(0.0f);
  }

 private:
  const float EPSILON = 0.0000001f;  // For preventing self-intersections

  Vector3f v0, v1, v2;  // For testing. Should have pointer to mesh here
};
};  // namespace gm