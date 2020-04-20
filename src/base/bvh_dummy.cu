#include "bvh_dummy.h"

gm::BVHDummy::BVHDummy(const std::unique_ptr<Scene> &scene) {
  /// "Build" BVH
  /// Iterate over mesh objects and build triangle primitives
  for (size_t i = 0; i < scene->meshes.size(); ++i) {
    std::shared_ptr<Mesh> mesh = scene->meshes[i];
    /// Iterate over the faces in the mesh

    for (size_t j = 0; j < mesh->faces.size(); ++j) {
      auto face = std::shared_ptr<Vector3i>(new Vector3i(mesh->faces[j]));
      Triangle triangle(mesh, face);
      primitives.push_back(triangle);
    }
  }
}

bool gm::BVHDummy::intersect(
    const Ray &ray, std::shared_ptr<Intersection> &intersection) const {
  /// Iterate over primitives and find nearest intersection
  std::shared_ptr<Intersection> temp;
  bool hitAnything = false;
  for (auto triangle : primitives) {
    bool hit = triangle.intersect(ray, temp);
    if (hit && temp->tHit < intersection->tHit) {
      /// Update intersection record if we have found a closer intersection
      intersection = temp;
      hitAnything = true;
    }
  }
  return hitAnything;
}