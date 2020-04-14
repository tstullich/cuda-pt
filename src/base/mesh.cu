#include "mesh.h"

gm::Mesh::Mesh(const std::vector<Vector3f> &vertices,
               const std::vector<Vector3f> &normals,
               const std::vector<Vector3i> &faces)
    : vertices(vertices), normals(normals), faces(faces) {}