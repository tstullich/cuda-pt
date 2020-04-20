#pragma once

#include <memory>

#include "mesh.h"
#include "primitive.h"
#include "vector.h"

namespace gm {

static const float EPSILON = 0.00001f;  // For preventing self-intersections

/// A class that represents a single triangle. The intersection test is based on
/// the Möller–Trumbore algorithm:
/// https://en.wikipedia.org/wiki/M%C3%B6ller%E2%80%93Trumbore_intersection_algorithm
/// The assumption is made that the coordinates for a given triangle are in
/// world space which might change later on if we want to implement meshes that
/// have a local coordinate system.
class Triangle : public Primitive {
 public:
  Triangle(const std::shared_ptr<Mesh> &mesh,
           const std::shared_ptr<Vector3i> &faceIndices)
      : mesh(mesh), faceIndices(faceIndices){};

  /// The intersection test assumes that the coordinates are in world space,
  /// otherwise the intersection test will run into undefined behavior
  bool intersect(
      const Ray &ray,
      const std::shared_ptr<Intersection> &intersection) const override {
    Vector3f edge1, edge2, h, s, q;
    float a, f, u, v;

    // Grab the vertices out of the mesh corresponding to the triangle face
    // to build an orthonormal basis
    edge1 = mesh->vertices[faceIndices->y] - mesh->vertices[faceIndices->x];
    edge2 = mesh->vertices[faceIndices->z] - mesh->vertices[faceIndices->x];
    h = cross(ray.direction, edge2);
    a = dot(edge1, h);
    if (a > -EPSILON && a < EPSILON) {
      return false;  // This ray is parallel to this triangle.
    }

    f = 1.0f / a;
    s = ray.origin - mesh->vertices[faceIndices->x];
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

    intersection->surfacePoint = ray.origin + ray.direction * t;
    intersection->tHit = t;
    return true;
  }

  /// Returns the surface area of the shape. This will be useful later when
  /// sampling area lights.
  float area() const override {
    Vector3f v0 = mesh->vertices[faceIndices->x];
    Vector3f v1 = mesh->vertices[faceIndices->y];
    Vector3f v2 = mesh->vertices[faceIndices->z];
    return 0.5f * cross(v1 - v0, v2 - v0).length();
  }

  // Return the surface normal of the shape given the point on the surface
  Vector3f normal(const Vector3f &surfacePoint) const override {
    Vector3f edge1 =
        mesh->vertices[faceIndices->y] - mesh->vertices[faceIndices->x];
    Vector3f edge2 =
        mesh->vertices[faceIndices->z] - mesh->vertices[faceIndices->x];
    return normal(cross(edge1, edge2));
  }

 private:
  // Pointers into the mesh and the mesh itself
  std::shared_ptr<Vector3i> faceIndices;
  std::shared_ptr<Mesh> mesh;
};
};  // namespace gm