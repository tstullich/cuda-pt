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
  /// Our initial entry point for traversing and building our data through the
  /// scene graph. We traverse the graph from the top down into all of the child nodes
  /// building a transformation matrix at each step. This should leave us with a local
  /// transformation matrix which can be added to all objects in the scene (meshes or cameras).
  void buildScene(const tinygltf::Scene &scene, const tinygltf::Model &model);

  /// Build a Translation * Rotation * Scale matrix based on the given quantities.
  /// A default value of (1, 1, 1,) for the scaling vector can be used, in cases
  /// where a scaling transformation is undesired (i.e, camera transformation)
  Matrix4x4f buildTransformationMatrix(const tinygltf::Node &node, const tinygltf::Model &model);

  /// Checks if the current node is a camera node or not. Will also check the children of
  /// the node to see if the "camera" attribute is contained there. Once found, the camera_id field
  /// will be set indicating the the location of the node with the camera field
  bool isCamera(const tinygltf::Node &node) const;

  /// A check to determine if the given node is a leaf node or not. This is dependent on whether
  /// or not the "children" array is empty or not
  bool isLeafNode(const tinygltf::Node &node) const;

  bool isMesh(const tinygltf::Node &node) const;

  void loadCamera(const tinygltf::Node &cameraNode, const tinygltf::Model &model,
                  const Matrix4x4f &parentTransformation);

  void loadMesh(const tinygltf::Node &meshNode,
                const tinygltf::Model &model,
                const Matrix4x4f &parentTransformation);

  void loadTransforms(const tinygltf::Node &node, Vector3f &translation,
                      Quaternionf &rotation, Vector3f &scale);

  tinygltf::Model readGltfFile(const std::string &filePath);

  void traverseNode(const tinygltf::Node &rootNode, const tinygltf::Model &model, Matrix4x4f &transformationMatrix);

  static const uint8_t HEADER_BYTES = 4;
  static const uint8_t TRIANGLE_VERT_COUNT = 3;
};
}// namespace gm