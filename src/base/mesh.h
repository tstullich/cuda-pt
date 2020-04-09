#pragma once

#include <string>
#include <vector>

#include "object.h"
#include "quaternion.h"
#include "vector.h"

namespace gm {
class Mesh : public Object {
 public:
  Mesh(const std::vector<Vector3f> &vertices,
       const std::vector<Vector3f> &normals, const std::vector<Vector3i> &faces,
       const std::string &name, const Vector3f &location,
       const Quaternionf &rotation, const Vector3f &scale);
  std::vector<Vector3f> vertices;
  std::vector<Vector3f> normals;
  std::vector<Vector3i> faces;
  virtual bool isMesh() { return true; }
  virtual bool isCamera() { return false; }
  virtual bool isEmpty() { return true; }
};
}  // namespace gm