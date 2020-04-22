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
    // Transform incoming ray into local coordinate space for intersection tests
    Ray localRay = transformToLocal(ray);

    Vector3f edge1, edge2, h, s, q;
    float a, f, u, v;
    // Grab the vertices out of the mesh corresponding to the triangle face
    // to build an orthonormal basis
    edge1 = mesh->vertices[faceIndices->y] - mesh->vertices[faceIndices->x];
    edge2 = mesh->vertices[faceIndices->z] - mesh->vertices[faceIndices->x];
    h = cross(localRay.direction, edge2);
    a = dot(edge1, h);
    if (a > -EPSILON && a < EPSILON) {
      return false;  // This ray is parallel to this triangle.
    }

    f = 1.0f / a;
    s = localRay.origin - mesh->vertices[faceIndices->x];
    u = f * dot(s, h);
    if (u < 0.0f || u > 1.0f) {
      return false;
    }

    q = cross(s, edge1);
    v = f * dot(localRay.direction, q);
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

    // We found an intersection. Now we need to transform the intersection point back into
    // world space and calculate the t there
    Vector3f localSurfacePoint = localRay.origin + localRay.direction * t;

    // Transform local surface point to world space
    Vector3f worldSpacePoint = mesh->meshToWorld.multiplyPoint(localSurfacePoint);

    // T value is the distance between the ray origin and the world-space surface point
    float tWorld = (worldSpacePoint - ray.origin).length();

    intersection->surfacePoint = worldSpacePoint;
    intersection->tHit = tWorld;
    intersection->normal = surfaceNormal(intersection->surfacePoint);
    intersection->name = mesh->name;
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
  Ray transformToLocal(const Ray &incomingRay) const {
    Vector3f localOrigin = mesh->worldToMesh.multiplyPoint(incomingRay.origin);
    Vector3f localDirection = mesh->worldToMesh.multiplyVector(incomingRay.direction);
    return { localOrigin, localDirection };
  }

  // Pointers into the mesh and the mesh itself
  std::shared_ptr<Mesh> mesh;
  std::shared_ptr<Vector3i> faceIndices;
};
};  // namespace gm