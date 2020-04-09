#include "mesh.h"

gm::Mesh::Mesh(std::vector<Vector3f> vertices, std::vector<Vector3f> normals,
               std::vector<Vector3i> faces, std::string name, Vector3f location,
               Quaternionf rotation, Vector3f scale)
    : Object(location, rotation, scale, name) {
  this->vertices = vertices;
  this->normals = normals;
  this->faces = faces;
}