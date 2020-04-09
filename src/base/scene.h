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
  Scene(std::string filepath);
  void addObject(std::shared_ptr<Object> o);
  std::shared_ptr<PerspectiveCamera> getCamera();
  std::vector<std::shared_ptr<Object>> objects;

 private:
  std::shared_ptr<PerspectiveCamera> load_camera(tinygltf::Node node,
                                                 tinygltf::Model model);
  std::shared_ptr<Mesh> load_mesh(tinygltf::Node node, tinygltf::Model model);
  std::shared_ptr<Object> load_empty(tinygltf::Node node,
                                     tinygltf::Model model);

  std::vector<std::shared_ptr<Object>> load_objects(
      std::vector<int> node_ids, tinygltf::Model model,
      std::shared_ptr<Object> parent = nullptr);

  void load_transform(tinygltf::Node node, Vector3f &location,
                      Quaternionf &rotation, Vector3f &scale);
};
}  // namespace gm