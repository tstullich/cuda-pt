#pragma once

#include <string>
#include <vector>

#include "quaternion.h"
#include "vector.h"

namespace gm {
class Mesh {
 public:
  Mesh(const std::vector<Vector3f> &vertices,
       const std::vector<Vector3f> &normals,
       const std::vector<Vector3i> &faces);
  std::vector<Vector3f> vertices;
  std::vector<Vector3f> normals;
  std::vector<Vector3i> faces;
};
}  // namespace gm