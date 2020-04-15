#pragma once

#include <memory>
#include <unordered_map>

#include "camera.h"
#include "mesh.h"
#include "mesh_object.h"
#include "scene_object.h"
#include "tiny_gltf.h"

namespace gm {
class Scene {
 public:
  Scene(){};

  Scene(const std::string &filepath);

  std::shared_ptr<PerspectiveCamera> getCamera();

  std::vector<std::shared_ptr<MeshObject>> meshObjects;

  std::shared_ptr<PerspectiveCamera> camera;

 private:
  std::shared_ptr<PerspectiveCamera> loadCamera(
      const std::vector<int> &node_ids, const tinygltf::Model &model);

  std::shared_ptr<MeshObject> loadMeshObject(
      const tinygltf::Node &node, const tinygltf::Model &model,
      std::unordered_map<int, std::shared_ptr<Mesh>> &meshes);

  std::shared_ptr<Mesh> loadMesh(const tinygltf::Mesh &mesh,
                                 const tinygltf::Model &model);

  std::vector<std::shared_ptr<MeshObject>> loadMeshObjects(
      const std::vector<int> &node_ids, const tinygltf::Model &model,
      const std::shared_ptr<MeshObject> &parent);

  void loadTransform(const tinygltf::Node &node, Vector3f &location,
                     Quaternionf &rotation, Vector3f &scale);
};
}  // namespace gm