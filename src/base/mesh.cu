#include "mesh.h"

gm::Mesh::Mesh(const std::vector<Vector3f> &vertices,
               const std::vector<Vector3f> &normals,
               const std::vector<Vector3i> &faces, const std::string &name,
               const Vector3f &location, const Quaternionf &rotation,
               const Vector3f &scale)
    : SceneObject(location, rotation, scale, name),
      vertices(vertices),
      normals(normals),
      faces(faces) {}