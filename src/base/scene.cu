#include <stack>

#include "scene.h"

#define TINYGLTF_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#include "tiny_gltf.h"

gm::Scene::Scene(const std::string &filepath) {
  tinygltf::Model model = readGltfFile(filepath);

  // We deserialize the first scene in the GLTF file
  tinygltf::Scene gltfScene = model.scenes[model.defaultScene];
  std::cout << "Loading scene: '" << gltfScene.name << "'" << std::endl;

  // We take a performance hit not preallocating the mesh vector, but
  // instead our parsing loop is going to be a little cleaner. Since we
  // are not going to work with large and complex scenes this should be fine
  meshes = std::vector<std::shared_ptr<Mesh>>();

  // If there are multiple cameras defined for this scene we only want to
  // load the first one we encounter until we can support camera selection
  bool loadedCamera = false;
  for (auto node_id : gltfScene.nodes) {
    tinygltf::Node node = model.nodes[node_id];

    /// Expand logic to apply transformations from parent node
    if (isMesh(node)) {
      // Encountered mesh node. Load it
      loadMesh(node, model);
    }

    int cameraNodeId;
    if (isCamera(node, model, cameraNodeId) && !loadedCamera) {
      // Encountered camera node. Load it
      loadCamera(node, model, cameraNodeId);
      loadedCamera = true;
    }
  }

  if (!loadedCamera) {
    // Some glTF scenes do not provide a camera node. In this case we should find
    // a good placement for the camera based on the bounding boxes of the scene.
    // For now we place the camera at position (0, 0, 2), looking down the negative
    // z axis
    camera = std::make_shared<PerspectiveCamera>(0.7f);
  }
}

gm::Matrix4x4f gm::Scene::buildTransformationMatrix(const Vector3f &translation, const Quaternionf &rotation,
                                                    const Vector3f &scale) {
  Matrix4x4f I;// Identity matrix
  Matrix4x4f T = translationMatrix(translation);
  Matrix4x4f R = rotation.toMat4();
  Matrix4x4f S = scaleMatrix(scale);
  return I * T * R * S;
}

bool gm::Scene::isCamera(const tinygltf::Node &node, const tinygltf::Model &model, int &cameraId) const {
  if (node.camera != -1) {
    // 'camera' attribute has been found in the parent node
    cameraId = node.camera;
    return true;
  }
  // If node has children search them to see if they have a 'camera' attribute
  for (int childId : node.children) {
    tinygltf::Node childNode = model.nodes[childId];
    if (childNode.camera != -1) {
      // Found camera attribute in the child node
      cameraId = childId;
      return true;
    }
  }
  return false;
}

bool gm::Scene::isMesh(const tinygltf::Node &node) const {
  return node.mesh != -1;
}

void gm::Scene::loadCamera(const tinygltf::Node &cameraNode,
                           const tinygltf::Model &model,
                           const int &cameraId) {
  tinygltf::Camera cameraData;
  if (cameraNode.camera == cameraId) {
    // The camera is contained in the node itself
    cameraData = model.cameras[cameraId];
  } else {
    // Get the camera data from the child node
    cameraData = model.cameras[model.nodes[cameraId].camera];
  }

  if (cameraData.type != "perspective") {
    // Only perspective projection cameras are supported
    std::cout << "Only perspective projection cameras are allowed at the moment!" << std::endl;
    exit(1);
  }

  Vector3f translation;
  Quaternionf rotation;
  Matrix4x4f transformationMatrix;
  if (cameraNode.camera == cameraId) {
    // The parent node holds the camera attribute. We assume that all information is contained
    // here and that we do not need to recurse into the child nodes to get more transformation
    // information.
    loadCameraTransform(cameraNode, translation, rotation);
    transformationMatrix = buildTransformationMatrix(translation, rotation);
  } else {
    // If no camera attribute is found in the parent node we need to build two
    // transformation matrices. The first for the parent and the second from one of the child nodes.
    // The case where more transformations are contained farther down in the scene graph
    // is not supported, but could be expanded on in the future.
    loadCameraTransform(cameraNode, translation, rotation);
    Matrix4x4f transformationMatrixParent = buildTransformationMatrix(translation, rotation);

    // Here we also load the node that was found through the first pass of the child nodes.
    Vector3f translationChild;
    Quaternionf rotationChild;
    tinygltf::Node childNode = model.nodes[cameraId];
    loadCameraTransform(childNode, translationChild, rotationChild);
    Matrix4x4f transformationMatrixChild = buildTransformationMatrix(translationChild, rotationChild);

    // Apply the transformation matrices
    transformationMatrix = transformationMatrixParent * transformationMatrixChild;
  }

  auto fov = static_cast<float>(cameraData.perspective.yfov);
  camera = std::make_shared<PerspectiveCamera>(PerspectiveCamera(transformationMatrix, fov));
}

void gm::Scene::loadMesh(const tinygltf::Node &meshNode,
                         const tinygltf::Model &model) {
  tinygltf::Mesh mesh = model.meshes[meshNode.mesh];
  std::vector<tinygltf::Buffer> buffers = model.buffers;
  std::vector<tinygltf::Accessor> accessors = model.accessors;
  std::vector<tinygltf::BufferView> bufferViews = model.bufferViews;

  std::vector<Vector3f> positions;
  std::vector<Vector3f> normals;
  std::vector<Vector3i> faces;

  size_t primitiveOffset = 0;

  // NOTE we are combining all primitives of this object to a single mesh
  for (size_t i = 0; i < mesh.primitives.size(); ++i) {
    int positionsIndex =
        mesh.primitives[i].attributes.find("POSITION")->second;
    int normalsIndex =
        mesh.primitives[i].attributes.find("NORMAL")->second;
    int facesIndex =
        mesh.primitives[i].indices;

    tinygltf::Accessor positionsAccessor = accessors[positionsIndex];
    tinygltf::Accessor normalsAccessor = accessors[normalsIndex];
    tinygltf::Accessor facesAccessor = accessors[facesIndex];

    tinygltf::BufferView positionsBufferView =
        bufferViews[positionsAccessor.bufferView];
    tinygltf::BufferView normalsBufferView =
        bufferViews[normalsAccessor.bufferView];
    tinygltf::BufferView facesBufferView =
        bufferViews[facesAccessor.bufferView];

    tinygltf::Buffer positionsBuffer = buffers[positionsBufferView.buffer];
    tinygltf::Buffer normalsBuffer = buffers[normalsBufferView.buffer];
    tinygltf::Buffer facesBuffer = buffers[facesBufferView.buffer];

    // Extract vertex positions
    size_t positionsBufferOffset = positionsBufferView.byteOffset;
    size_t positionsBufferLength = positionsBufferView.byteLength;
    // TODO handle byteStride
    Vector3f *positionsBytes = reinterpret_cast<Vector3f *>(
        positionsBuffer.data.data() + positionsBufferOffset);
    std::vector<Vector3f> primitivePositions(
        positionsBytes,
        positionsBytes + positionsBufferLength / sizeof(Vector3f));

    // Extract normals
    size_t normalsBufferOffset = normalsBufferView.byteOffset;
    size_t normalsBufferLength = normalsBufferView.byteLength;
    // TODO handle byteStride
    Vector3f *normalsBytes = reinterpret_cast<Vector3f *>(
        normalsBuffer.data.data() + normalsBufferOffset);
    std::vector<Vector3f> primitiveNormals(
        normalsBytes, normalsBytes + normalsBufferLength / sizeof(Vector3f));

    // Extract face indices
    size_t facesBufferOffset = facesBufferView.byteOffset;

    static const size_t faceCount =
        facesAccessor.count / gm::Scene::TRIANGLE_VERT_COUNT;// gltf only supports triangles. No quads or
                                                             // ngons

    std::vector<Vector3i> primitiveFaces(faceCount);

    // TODO handle byteStride

    if (facesAccessor.componentType == 5123) {// Component type unsigned
      auto *facesBytes = reinterpret_cast<unsigned short *>(
          facesBuffer.data.data() + facesBufferOffset);
      for (size_t f = 0; f < faceCount; ++f) {
        primitiveFaces[f] =
            Vector3i(facesBytes[f * TRIANGLE_VERT_COUNT] + primitiveOffset,
                     facesBytes[f * TRIANGLE_VERT_COUNT + 1] + primitiveOffset, facesBytes[f * TRIANGLE_VERT_COUNT + 2] + primitiveOffset);
      }

    } else if (facesAccessor.componentType == 5125) {// Component type unsigned int
      auto *facesBytes = reinterpret_cast<unsigned int *>(
          facesBuffer.data.data() + facesBufferOffset);
      for (size_t f = 0; f < faceCount; ++f) {
        primitiveFaces[f] =
            Vector3i(facesBytes[f * TRIANGLE_VERT_COUNT] + primitiveOffset,
                     facesBytes[f * TRIANGLE_VERT_COUNT + 1] + primitiveOffset, facesBytes[f * TRIANGLE_VERT_COUNT + 2] + primitiveOffset);
      }
    }

    positions.insert(positions.end(), primitivePositions.begin(),
                     primitivePositions.end());
    normals.insert(normals.end(), primitiveNormals.begin(),
                   primitiveNormals.end());
    faces.insert(faces.end(), primitiveFaces.begin(), primitiveFaces.end());

    primitiveOffset += primitivePositions.size();
  }

  // TODO Add support for loading a complete matrix
  Vector3f translation;
  Quaternionf rotation;
  Vector3f scale;
  loadMeshTransform(meshNode, translation, rotation, scale);

  Matrix4x4f localTransformation = buildTransformationMatrix(translation, rotation, scale);
  meshes.push_back(std::make_shared<Mesh>(Mesh(positions, normals, faces, mesh.name, localTransformation)));
}

void gm::Scene::loadCameraTransform(const tinygltf::Node &node, Vector3f &translation,
                                    Quaternionf &rotation) {
  if (node.translation.size() == 3) {
    translation =
        Vector3f(node.translation[0], node.translation[1], node.translation[2]);
  } else {
    translation = Vector3f(0);
  }

  if (node.rotation.size() == 4) {
    rotation = Quaternionf(node.rotation[0], node.rotation[1], node.rotation[2],
                           node.rotation[3]);
  } else {
    rotation = Quaternionf();
  }
}

void gm::Scene::loadMeshTransform(const tinygltf::Node &node, Vector3f &translation,
                                  Quaternionf &rotation, Vector3f &scale) {
  if (node.translation.size() == 3) {
    translation =
        Vector3f(node.translation[0], node.translation[1], node.translation[2]);
  } else {
    translation = Vector3f(0);
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

tinygltf::Model gm::Scene::readGltfFile(const std::string &filePath) {
  tinygltf::TinyGLTF loader;
  tinygltf::Model model;
  std::string err;
  std::string warn;
  std::ifstream fileStream;

  fileStream.open(filePath);
  if (fileStream.fail()) {
    std::cout << "Unable to open file: " << filePath << std::endl;
  }

  char magicNumber[HEADER_BYTES];
  fileStream.read(magicNumber, HEADER_BYTES);
  fileStream.close();

  bool loaded = false;
  // Check magic number 0x46546c67 == "glTF"
  if (*reinterpret_cast<int *>(magicNumber) == 0x46546c67) {
    loaded = loader.LoadBinaryFromFile(&model, &err, &warn, filePath);
  } else if (filePath.substr(filePath.length() - 5, filePath.length()) == ".gltf") {
    loaded = loader.LoadASCIIFromFile(&model, &err, &warn, filePath);
  } else {
    // Error here because we got an unknown file extension.
    std::cout << "Unknown file extension! Make sure your file uses the .gltf or .glb extension" << std::endl;
    exit(1);
  }

  if (!loaded) {
    std::cout << "Failed to parse glTF file!" << std::endl;
    exit(1);
  }

  if (!err.empty()) {
    std::cout << "Error: " << err.c_str() << std::endl;
    exit(1);
  }

  if (!warn.empty()) {
    std::cout << "Warning: " << warn.c_str() << std::endl;
  }
  return model;
}
