#pragma once

#include <memory>

#include "mesh.h"
#include "primitive.h"
#include "vector.h"

namespace gm {

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
      : mesh(mesh),
        faceIndices(faceIndices){};

  /// The intersection test assumes that the coordinates are in world space,
  /// otherwise the intersection test will run into undefined behavior
  bool intersect(
      const Ray &ray,
      const std::shared_ptr<Intersection> &intersection) const override {
    Vector3f v0 = mesh->vertices[faceIndices->x];
    Vector3f v1 = mesh->vertices[faceIndices->y];
    Vector3f v2 = mesh->vertices[faceIndices->z];

    v0 = mesh->meshToWorld.multiplyPoint(v0);
    v1 = mesh->meshToWorld.multiplyPoint(v1);
    v2 = mesh->meshToWorld.multiplyPoint(v2);

    Vector3f edge1 = v1 - v0;
    Vector3f edge2 = v2 - v0;

    Vector3f h = cross(ray.direction, edge2);
    float det = dot(edge1, h);
    if (det > -EPSILON && det < EPSILON) {
      // This ray is parallel to this triangle
      return false;
    }

    float invDet = 1.0f / det;
    Vector3f s = ray.origin - v0;
    float u =  dot(s, h) * invDet;
    if (u < 0.0f || u > 1.0f) {
      return false;
    }

    Vector3f q = cross(s, edge1);
    float v = dot(ray.direction, q) * invDet;
    if (v < 0.0f || u + v > 1.0f) {
      return false;
    }

    // At this stage we can compute t to find out where the intersection point is on the line.
    float t = dot(edge2, q) * invDet;
    if (t < EPSILON) {
      // This means that there is a line intersection but not a ray intersection.
      return false;
    }

    intersection->tHit = t;
    intersection->name = mesh->name;
    // Surface point in barycentric coordinates
    intersection->surfacePoint = Vector3f(u, v, 1.0f - u - v);

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
  Vector3f surfaceNormal(const Vector3f &surfacePoint) const override {
    Vector3f v0 = mesh->vertices[faceIndices->x];
    Vector3f v1 = mesh->vertices[faceIndices->y];
    Vector3f v2 = mesh->vertices[faceIndices->z];
    return normalize(cross(v1 - v0, v2 - v0));
  }

 private:
  constexpr static const float EPSILON = 1e-7;// For preventing self-intersections

  // Pointers into the mesh and the mesh itself
  std::shared_ptr<Mesh> mesh;
  std::shared_ptr<Vector3i> faceIndices;
};
};// namespace gm