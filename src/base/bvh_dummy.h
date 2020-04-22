#pragma once

#include <iostream>
#include <memory>

#include "intersection.h"
#include "mesh.h"
#include "ray.h"
#include "scene.h"
#include "triangle.h"

/// Class for testing intersections with a BVH
/// Will be swapped out for actual BVH
namespace gm {
class BVHDummy {
 public:
  BVHDummy(const std::unique_ptr<Scene> &scene);

  bool intersect(const Ray &ray,
                 std::shared_ptr<Intersection> &intersection) const;

  std::vector<Triangle> primitives;
};
}  // namespace gm