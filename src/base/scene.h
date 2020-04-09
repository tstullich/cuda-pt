#pragma once

#include <memory>

#include "camera.h"
#include "mesh.h"
#include "object.h"
#include "tiny_gltf.h"

namespace gm {
class Scene {
 public:
  Scene(){};
  Scene(const std::string &filepath);
  void addObject(const std::shared_ptr<SceneObject> &o);
  std::shared_ptr<PerspectiveCamera> getCamera();
  std::vector<std::shared_ptr<SceneObject>> objects;

 private:
  std::shared_ptr<PerspectiveCamera> load_camera(const tinygltf::Node &node,
                                                 const tinygltf::Model &model);
  std::shared_ptr<Mesh> load_mesh(const tinygltf::Node &node,
                                  const tinygltf::Model &model);
  std::shared_ptr<SceneObject> load_empty(const tinygltf::Node &node,
                                          const tinygltf::Model &model);

  std::vector<std::shared_ptr<SceneObject>> load_objects(
      const std::vector<int> &node_ids, const tinygltf::Model &model,
      const std::shared_ptr<SceneObject> &parent);

  void load_transform(const tinygltf::Node &node, Vector3f &location,
                      Quaternionf &rotation, Vector3f &scale);
};
}  // namespace gm