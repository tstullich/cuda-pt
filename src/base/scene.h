#pragma once

#include <memory>
#include <unordered_map>

#include "camera.h"
#include "matrix.h"
#include "mesh.h"
#include "tiny_gltf.h"

namespace gm {

class Scene {
 public:
  Scene(){};

  Scene(const std::string &filePath);

  std::vector<std::shared_ptr<Mesh>> meshes;

  std::shared_ptr<PerspectiveCamera> camera;

 private:
  Matrix4x4f buildTransformationMatrix(const Vector3f &translation, const Quaternionf &rotation);

  Matrix4x4f buildTransformationMatrix(const Vector3f &translation, const Quaternionf &rotation,
                                       const Vector3f &scale);

  /// Checks if the current node is a camera node or not. Will also check the children of
  /// the node to see if the "camera" attribute is contained there. Once found, the camera_id field
  /// will be set indicating the the location of the node with the camera field
  bool isCamera(const tinygltf::Node &node, const tinygltf::Model &model, int &cameraId) const;

  bool isMesh(const tinygltf::Node &node) const;

  void loadCamera(const tinygltf::Node &cameraNode, const tinygltf::Model &model, const int &cameraId);

  std::shared_ptr<Mesh> loadMesh(const tinygltf::Node &meshNode,
                                 const tinygltf::Model &model);

  void loadTransform(const tinygltf::Node &node, Vector3f &translation,
                     Quaternionf &rotation);

  void loadTransform(const tinygltf::Node &node, Vector3f &translation,
                     Quaternionf &rotation, Vector3f &scale);

  tinygltf::Model readGltfFile(const std::string &filePath);

  static const uint8_t HEADER_BYTES = 4;
  static const uint8_t TRIANGLE_VERT_COUNT = 3;
};
}// namespace gm