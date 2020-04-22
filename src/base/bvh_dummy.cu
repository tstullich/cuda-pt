#include "bvh_dummy.h"

gm::BVHDummy::BVHDummy(const std::unique_ptr<Scene> &scene) {
  /// "Build" BVH
  /// Iterate over mesh objects and build triangle primitives
  for (const auto &mesh : scene->meshes) {
    /// Iterate over the faces in the mesh
    for (size_t j = 0; j < mesh->faces.size(); ++j) {
      auto face = std::make_shared<Vector3i>(Vector3i(mesh->faces[j]));
      primitives.emplace_back(Triangle(mesh, face));
    }
  }
}

bool gm::BVHDummy::intersect(
    const Ray &ray, std::shared_ptr<Intersection> &intersection) const {
  /// Iterate over primitives and find nearest intersection
  std::shared_ptr<Intersection> temp = std::make_shared<Intersection>();
  bool hitAnything = false;
  for (const auto &triangle : primitives) {
    bool hit = triangle.intersect(ray, temp);
    if (hit && temp->tHit < intersection->tHit) {
      /// Update intersection record if we have found a closer intersection
      intersection = temp;
      hitAnything = true;
    }
  }
  return hitAnything;
}