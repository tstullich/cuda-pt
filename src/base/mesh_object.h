#pragma once

#include <string>
#include <vector>

#include "mesh.h"
#include "quaternion.h"
#include "scene_object.h"
#include "vector.h"

namespace gm {
class MeshObject : public SceneObject {
 public:
  MeshObject(const std::shared_ptr<Mesh> &mesh, const std::string &name,
             const Vector3f &location, const Quaternionf &rotation,
             const Vector3f &scale)
      : SceneObject(location, rotation, scale, name), mesh(mesh) {}

  std::shared_ptr<Mesh> mesh;
  std::shared_ptr<MeshObject> parent;
  std::vector<std::shared_ptr<MeshObject>> children;
};
}  // namespace gm