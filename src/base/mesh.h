#pragma once

#include <string>
#include <vector>

#include "matrix.h"
#include "quaternion.h"
#include "vector.h"

namespace gm {

class Mesh {
 public:
  Mesh(const std::vector<Vector3f> &vertices,
       const std::vector<Vector3f> &normals, const std::vector<Vector3i> &faces,
       const std::string &name, const Matrix4x4f &meshToWorld)
      : vertices(vertices),
        normals(normals),
        faces(faces),
        name(name),
        meshToWorld(meshToWorld),
        worldToMesh(meshToWorld.invert()) {}

  std::vector<Vector3f> vertices;
  std::vector<Vector3f> normals;
  std::vector<Vector3i> faces;

  // Transformation matrices used for various purposes
  Matrix4x4f meshToWorld;
  Matrix4x4f worldToMesh;

  // Name of the mesh from the glTF file
  std::string name;
};
}  // namespace gm