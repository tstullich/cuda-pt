#include "mesh_object.h"

gm::MeshObject::MeshObject(const std::shared_ptr<Mesh> &mesh,
                           const std::string &name, const Vector3f &location,
                           const Quaternionf &rotation, const Vector3f &scale)
    : SceneObject(location, rotation, scale, name), mesh(mesh) {}