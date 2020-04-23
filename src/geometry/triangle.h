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
    // Transform incoming ray into local coordinate space for intersection tests
    Ray localRay = transformToLocal(ray);

    // Grab the vertices out of the mesh corresponding to the triangle face
    // to build an orthonormal basis
    Vector3f v0 = mesh->vertices[faceIndices->x];
    Vector3f v1 = mesh->vertices[faceIndices->y];
    Vector3f v2 = mesh->vertices[faceIndices->z];

    Vector3f edge1 = v1 - v0;
    Vector3f edge2 = v2 - v0;
    Vector3f pvec = cross(localRay.direction, edge2);
    float det = dot(edge1, pvec);
    if (det < EPSILON) {
      // if the determinant is negative the triangle is backfacing
      // if the determinant is close to 0, the ray misses the triangle
      return false;
    }

    if (fabs(det) < EPSILON) {
      // ray and triangle are parallel if det is close to 0
      return false;
    }

    float invDet = 1.0f / det;
    Vector3f tvec = localRay.origin - v0;
    float u = dot(tvec, pvec) * invDet;
    if (u < 0.0f || u > 1.0f) {
      return false;
    }

    Vector3f qvec = cross(tvec, edge1);
    float v = dot(localRay.direction, qvec) * invDet;
    if (v < 0 || u + v > 1) {
      return false;
    }


    float t = dot(edge2, qvec) * invDet;
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
    return {localOrigin, localDirection};
  }

  constexpr static const float EPSILON = 1e-8;// For preventing self-intersections

  // Pointers into the mesh and the mesh itself
  std::shared_ptr<Mesh> mesh;
  std::shared_ptr<Vector3i> faceIndices;
};
};// namespace gm