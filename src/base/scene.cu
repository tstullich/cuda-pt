#include <stack>

#include "scene.h"

#define TINYGLTF_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#include "tiny_gltf.h"

gm::Scene::Scene(const std::string &filepath, const RenderOptions &options) {
  tinygltf::Model model = readGltfFile(filepath);

  meshes = std::vector<std::shared_ptr<Mesh>>();

  // Check if a default scene is available, otherwise use the first scene available
  int defaultId = (model.defaultScene != -1) ? model.defaultScene : 0;
  tinygltf::Scene defaultScene = model.scenes[defaultId];
  buildScene(defaultScene, model, options);

  // Shrink the vector in case we had to resize and did not
  // fill the entire vector
  meshes.shrink_to_fit();
}

void gm::Scene::buildScene(const tinygltf::Scene &scene, const tinygltf::Model &model, const RenderOptions &options) {
  std::cout << "Loading scene: '" << scene.name << "'" << std::endl;

  // If there are multiple cameras defined for this scene we only want to
  // load the first one we encounter until we can support camera selection
  for (auto rootNodeId : scene.nodes) {
    Matrix4x4f rootTransformation;// Initialized to Identity matrix
    traverseNode(model.nodes[rootNodeId], model, rootTransformation);
  }

  if (!loadedCamera) {
    // Some glTF scenes do not provide a camera node. In this case we place the camera at the world space origin
    // facing down the -Z axis. Later we can find a better placement based on the meshes in the scene
    camera = std::make_shared<PerspectiveCamera>(PerspectiveCamera(0.7f, 0.1f, 100.0f));
  }

  /// Use the camera parameters to initialize the camera matrices
  camera->init(options.imageWidth, options.imageHeight);
}

gm::Matrix4x4f gm::Scene::buildTransformationMatrix(const tinygltf::Node &node,
                                                    const tinygltf::Model &model) {
  if (!node.matrix.empty()) {
    // Node has a transformation matrix already available. Initialize it
    return Matrix4x4f(node.matrix);
  } else {
    Vector3f translation;
    Quaternionf rotation;
    Vector3f scale;
    loadTransforms(node, translation, rotation, scale);

    Matrix4x4f T = translationMatrix(translation);
    Matrix4x4f R = rotation.toMat4();
    Matrix4x4f S = scaleMatrix(scale);
    return T * R * S;
  }
}

bool gm::Scene::isCamera(const tinygltf::Node &node) const {
  return node.camera != -1;
}

bool gm::Scene::isLeafNode(const tinygltf::Node &node) const {
  return node.children.empty();
}

bool gm::Scene::isMesh(const tinygltf::Node &node) const {
  return node.mesh != -1;
}

void gm::Scene::loadCamera(const tinygltf::Node &cameraNode, const tinygltf::Model &model,
                           const Matrix4x4f &parentTransformation) {
  if (cameraNode.camera == -1) {
    // Something went wrong. We shouldn't be here
    std::cout << "Encountered camera node when we should not have!" << std::endl;
    exit(1);
  }

  tinygltf::Camera cameraData = model.cameras[cameraNode.camera];
  if (cameraData.type != "perspective") {
    // Only perspective projection cameras are supported
    std::cout << "Only perspective projection cameras are allowed at the moment!" << std::endl;
    exit(1);
  }

  Matrix4x4f localTransformation = parentTransformation * buildTransformationMatrix(cameraNode, model);
  auto fov = static_cast<float>(cameraData.perspective.yfov);
  auto near = static_cast<float>(cameraData.perspective.znear);
  // zfar might not always be present. Default to set the far plane at 100 units if it's not present
  auto far = cameraData.perspective.zfar > 0.0f ? static_cast<float>(cameraData.perspective.zfar) : 100.0f;
  camera = std::make_shared<PerspectiveCamera>(PerspectiveCamera(localTransformation, fov, near, far));
  loadedCamera = true; // Flip this flag
}

void gm::Scene::loadMesh(const tinygltf::Node &meshNode,
                         const tinygltf::Model &model,
                         const Matrix4x4f &parentTransformation) {
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

    static const size_t faceCount =
        facesAccessor.count / gm::Scene::TRIANGLE_VERT_COUNT;// gltf only supports triangles. No quads or
                                                             // ngons
    std::vector<Vector3i> primitiveFaces(faceCount);

    // TODO handle byteStride

    // Extract face indices
    size_t facesBufferOffset = facesBufferView.byteOffset;
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

  Matrix4x4f localTransformation = parentTransformation * buildTransformationMatrix(meshNode, model);
  meshes.push_back(std::make_shared<Mesh>(Mesh(positions, normals, faces, mesh.name, localTransformation)));
}

void gm::Scene::loadTransforms(const tinygltf::Node &node, Vector3f &translation,
                               Quaternionf &rotation, Vector3f &scale) {
  if (node.translation.empty()) {
    translation = Vector3f(0.0f);
  } else {
    translation = Vector3f(node.translation[0], node.translation[1], node.translation[2]);
  }

  if (node.scale.empty()) {
    scale = Vector3f(1);
  } else {
    scale = Vector3f(node.scale[0], node.scale[1], node.scale[2]);
  }

  if (node.rotation.empty()) {
    rotation = Quaternionf();
  } else {
    rotation = Quaternionf(node.rotation[0], node.rotation[1], node.rotation[2],
                           node.rotation[3]);
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

void gm::Scene::traverseNode(const tinygltf::Node &node, const tinygltf::Model &model,
                             Matrix4x4f &transformationMatrix) {
  if (!isLeafNode(node)) {
    // Encountered a node with more children. Build transformation matrix and pass it down
    transformationMatrix = transformationMatrix * buildTransformationMatrix(node, model);
    for (const auto &childNodeId : node.children) {
      traverseNode(model.nodes[childNodeId], model, transformationMatrix);
    }
  }

  if (isMesh(node)) {
    // Reached a mesh leaf node. Build it
    loadMesh(node, model, transformationMatrix);
  }

  if (isCamera(node)) {
    // Encountered camera node. Load it
    loadCamera(node, model, transformationMatrix);
  }
}
