#include <stack>

#include "scene.h"

#define TINYGLTF_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#include "tiny_gltf.h"

__host__ gm::Scene::Scene(const std::string &filepath) {
  tinygltf::Model model;
  tinygltf::TinyGLTF loader;
  std::string err;
  std::string warn;

  std::ifstream fileStream;

  fileStream.open(filepath);
  if (fileStream.fail()) {
    std::cout << "Unable to open file: " << filepath << std::endl;
  }

  static const uint8_t HEADER_BYTES = 4;

  char magicNumber[HEADER_BYTES];
  fileStream.read(magicNumber, HEADER_BYTES);
  fileStream.close();

  bool ret = false;
  // Check magic number 0x46546c67 == "glTF"
  if (*reinterpret_cast<int *>(magicNumber) == 0x46546c67) {
    ret = loader.LoadBinaryFromFile(&model, &err, &warn, filepath);
  } else if (filepath.substr(filepath.length() - 5, filepath.length()) ==
             ".gltf") {
    ret = loader.LoadASCIIFromFile(&model, &err, &warn, filepath);
  } else {
    // Error here because we got an unknown file extension.
    std::cout << "Unknown file extension!" << std::endl;
    exit(1);
  }

  if (!warn.empty()) {
    std::cout << "Warning: " << warn.c_str() << std::endl;
  }

  if (!err.empty()) {
    std::cout << "Error: " << err.c_str() << std::endl;
    exit(1);
  }

  if (!ret) {
    std::cout << "Failed to parse glTF file!" << std::endl;
    exit(1);
  }

  tinygltf::Scene gltfScene = model.scenes[model.defaultScene];
  std::cout << "Loading scene: '" << gltfScene.name << "'" << std::endl;

  for (auto node_id : gltfScene.nodes) {
    std::cout << node_id << std::endl;
  }

  loadMeshes(gltfScene.nodes, model);
  camera = loadCamera(gltfScene.nodes, model);
}

void gm::Scene::loadMeshes(const std::vector<int> &node_ids,
                           const tinygltf::Model &model) {
  // meshes = std::vector<std::shared_ptr<MeshObject>>(node_ids.size());
  // std::unordered_map<int, std::shared_ptr<Mesh>> meshes;

  // std::shared_ptr<MeshObject> current;
  // meshObjects.reserve(node_ids.size());
  // if (node_ids.size() > 0) {
  //  std::cout << "Loading " << node_ids.size() << " object(s)" << std::endl;
  //}
  // for (size_t i = 0; i < node_ids.size(); ++i) {
  //  tinygltf::Node node = model.nodes[node_ids[i]];

  //  // NOTE this needs to be extended once more object types are added
  //  if (node.mesh != -1) {
  //    current = loadMeshObject(node, model, meshes);
  //    // current->children = loadMeshObjects(node.children, model, current);
  //    meshObjects.push_back(current);
  //  }
  //}
  //// Not all nodes are mesh objects thus the preallocated vector might be
  /// larger / than necessary
  // meshObjects.shrink_to_fit();
  // return meshObjects;
}

// std::shared_ptr<gm::Mesh> gm::Scene::loadMeshObject(
//    const tinygltf::Node &node, const tinygltf::Model &model,
//    std::unordered_map<int, std::shared_ptr<Mesh>> &meshes) {
//  Vector3f location;
//  Quaternionf rotation;
//  Vector3f scale;
//  loadTransform(node, location, rotation, scale);
//
//  // This object is an empty
//  if (node.mesh == -1) {
//    return std::shared_ptr<MeshObject>(
//        new MeshObject(nullptr, node.name, location, rotation, scale));
//  }
//
//  auto it = meshes.find(node.mesh);
//  std::shared_ptr<Mesh> mesh;
//  if (it == meshes.end()) {
//    mesh = loadMesh(model.meshes[node.mesh], model);
//    meshes.insert({node.mesh, mesh});
//  }
//  return std::shared_ptr<MeshObject>(
//      new MeshObject(mesh, node.name, location, rotation, scale));
//}

// std::shared_ptr<gm::Mesh> gm::Scene::loadMesh(const tinygltf::Mesh &mesh,
//                                              const tinygltf::Model &model) {
//  static const uint8_t TRIANGLE_VERT_COUNT = 3;
//
//  std::vector<tinygltf::Mesh> meshes = model.meshes;
//  std::vector<tinygltf::Buffer> buffers = model.buffers;
//  std::vector<tinygltf::Accessor> accessors = model.accessors;
//  std::vector<tinygltf::BufferView> bufferViews = model.bufferViews;
//
//  std::vector<Vector3f> positions;
//  std::vector<Vector3f> normals;
//  std::vector<Vector3i> faces;
//
//  size_t primitiveOffset = 0;
//
//  // NOTE we are combining all primitives of this object to a single mesh
//  for (size_t i = 0; i < mesh.primitives.size(); ++i) {
//    int positionsIndex =
//    mesh.primitives[i].attributes.find("POSITION")->second; int normalsIndex =
//    mesh.primitives[i].attributes.find("NORMAL")->second; int facesIndex =
//    mesh.primitives[i].indices;
//
//    tinygltf::Accessor positionsAccessor = accessors[positionsIndex];
//    tinygltf::Accessor normalsAccessor = accessors[normalsIndex];
//    tinygltf::Accessor facesAccessor = accessors[facesIndex];
//
//    tinygltf::BufferView positionsBufferView =
//        bufferViews[positionsAccessor.bufferView];
//    tinygltf::BufferView normalsBufferView =
//        bufferViews[normalsAccessor.bufferView];
//    tinygltf::BufferView facesBufferView =
//        bufferViews[facesAccessor.bufferView];
//
//    tinygltf::Buffer positionsBuffer = buffers[positionsBufferView.buffer];
//    tinygltf::Buffer normalsBuffer = buffers[normalsBufferView.buffer];
//    tinygltf::Buffer facesBuffer = buffers[facesBufferView.buffer];
//
//    // Extract vertex positions
//    size_t positionsBufferOffset = positionsBufferView.byteOffset;
//    size_t positionsBufferLength = positionsBufferView.byteLength;
//    // TODO handle byteStride
//    Vector3f *positionsBytes = reinterpret_cast<Vector3f *>(
//        positionsBuffer.data.data() + positionsBufferOffset);
//    std::vector<Vector3f> primitivePositions(
//        positionsBytes,
//        positionsBytes + positionsBufferLength / sizeof(Vector3f));
//
//    // Extract normals
//    size_t normalsBufferOffset = normalsBufferView.byteOffset;
//    size_t normalsBufferLength = normalsBufferView.byteLength;
//    // TODO handle byteStride
//    Vector3f *normalsBytes = reinterpret_cast<Vector3f *>(
//        normalsBuffer.data.data() + normalsBufferOffset);
//    std::vector<Vector3f> primitiveNormals(
//        normalsBytes, normalsBytes + normalsBufferLength / sizeof(Vector3f));
//
//    // Extract face indices
//    size_t facesBufferOffset = facesBufferView.byteOffset;
//
//    static const size_t faceCount =
//        facesAccessor.count /
//        TRIANGLE_VERT_COUNT; // gltf only supports triangles. No quads or
//                             // ngons
//
//    std::vector<Vector3i> primitiveFaces(faceCount);
//
//    // TODO handle byteStride
//
//    if (facesAccessor.componentType == 5123) { // Component type unsigned
//    short
//      unsigned short *facesBytes = reinterpret_cast<unsigned short *>(
//          facesBuffer.data.data() + facesBufferOffset);
//      for (size_t f = 0; f < faceCount; ++f) {
//        primitiveFaces[f] =
//            Vector3i(facesBytes[f * TRIANGLE_VERT_COUNT] + primitiveOffset,
//                     facesBytes[f * TRIANGLE_VERT_COUNT + 1] +
//                     primitiveOffset, facesBytes[f * TRIANGLE_VERT_COUNT + 2]
//                     + primitiveOffset);
//      }
//
//    } else if (facesAccessor.componentType ==
//               5125) { // Component type unsigned int
//      unsigned int *facesBytes = reinterpret_cast<unsigned int *>(
//          facesBuffer.data.data() + facesBufferOffset);
//      for (size_t f = 0; f < faceCount; ++f) {
//        primitiveFaces[f] =
//            Vector3i(facesBytes[f * TRIANGLE_VERT_COUNT] + primitiveOffset,
//                     facesBytes[f * TRIANGLE_VERT_COUNT + 1] +
//                     primitiveOffset, facesBytes[f * TRIANGLE_VERT_COUNT + 2]
//                     + primitiveOffset);
//      }
//    }
//
//    positions.insert(positions.end(), primitivePositions.begin(),
//                     primitivePositions.end());
//    normals.insert(normals.end(), primitiveNormals.begin(),
//                   primitiveNormals.end());
//    faces.insert(faces.end(), primitiveFaces.begin(), primitiveFaces.end());
//
//    primitiveOffset += primitivePositions.size();
//  }
//
//  return std::shared_ptr<Mesh>(new Mesh(positions, normals, faces));
//}

std::shared_ptr<gm::PerspectiveCamera>
gm::Scene::loadCamera(const std::vector<int> &node_ids,
                      const tinygltf::Model &model) {
  // NOTE The following section uses raw pointers to improve performance and
  // prevent unnecessary copies. Smart pointers can not be used here as we are
  // not managing the memory to which we are pointing.
  const tinygltf::Node *camera = nullptr;
  std::unordered_map<const tinygltf::Node *, const tinygltf::Node *> parents;

  // find camera node
  for (size_t i = 0; i < node_ids.size() && !camera; ++i) {
    std::stack<const tinygltf::Node *> toVisit;

    toVisit.push(&model.nodes[node_ids[i]]);

    while (!toVisit.empty() && !camera) {
      const tinygltf::Node *current = toVisit.top();
      toVisit.pop();

      if (current->camera != -1) {
        camera = current;
        break;
      }

      // add all child nodes to the stack
      for (size_t c = 0; c < current->children.size(); ++c) {
        const tinygltf::Node *child = &model.nodes[current->children[c]];
        toVisit.push(child);
        parents.insert({child, current});
      }
    }
  }

  if (!camera) {
    std::cout << "NO CAMERA FOUND!" << std::endl;
    // Error no camera in scene!
    // TODO handle error
  }

  // construct parent chain for camera node
  std::vector<const tinygltf::Node *> parentTree;
  const tinygltf::Node *current = camera;
  auto result = parents.find(current);
  while (result != parents.end()) {
    parentTree.push_back(result->second);
    current = result->second;
    result = parents.find(current);
  }

  // accumulate transforms
  Vector3f locationEff;
  Quaternionf rotationEff;
  Vector3f scaleEff(1);

  for (size_t p = parentTree.size(); p--;) {
    Vector3f location;
    Quaternionf rotation;
    Vector3f scale;
    loadTransform(*parentTree[p], location, rotation, scale);

    rotationEff = rotationEff * rotation;
    locationEff += rotationEff * (location * scaleEff);
    scaleEff += rotationEff * scale;
  }

  tinygltf::Camera cameraData = model.cameras[camera->camera];

  if (cameraData.type != "perspective") {
    // Error only perspective cameras are supported
    // TODO handle error
  }
  float fov = cameraData.perspective.yfov;

  return std::shared_ptr<PerspectiveCamera>(
      new PerspectiveCamera(locationEff, rotationEff, fov));
}

void gm::Scene::loadTransform(const tinygltf::Node &node, Vector3f &location,
                              Quaternionf &rotation, Vector3f &scale) {
  if (node.translation.size() == 3) {
    location =
        Vector3f(node.translation[0], node.translation[1], node.translation[2]);
  } else {
    location = Vector3f(0);
  }

  if (node.scale.size() == 3) {
    scale = Vector3f(node.scale[0], node.scale[1], node.scale[2]);
  } else {
    scale = Vector3f(1);
  }

  if (node.rotation.size() == 4) {
    rotation = Quaternionf(node.rotation[0], node.rotation[1], node.rotation[2],
                           node.rotation[3]);
  } else {
    rotation = Quaternionf();
  }
}
