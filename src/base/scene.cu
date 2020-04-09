#include "object.h"
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

  char magicNumber[4];
  fileStream.read(magicNumber, 4);
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
    // TODO error handling
  }

  if (!warn.empty()) {
    printf("Warn: %s\n", warn.c_str());
  }

  if (!err.empty()) {
    printf("Err: %s\n", err.c_str());
  }

  if (!ret) {
    printf("Failed to parse glTF\n");
  }

  tinygltf::Scene gltfScene = model.scenes[model.defaultScene];
  std::cout << "Loading scene " << gltfScene.name << std::endl;
  this->objects = load_objects(gltfScene.nodes, model);

  for (size_t i = 0; i < this->objects.size(); ++i) {
    if (this->objects[i] != nullptr) {
      std::cout << this->objects[i]->name
                << " location:" << this->objects[i]->location[0] << ", "
                << this->objects[i]->location[1] << ", "
                << this->objects[i]->location[2] << std::endl;
    }
  }
}

std::vector<std::shared_ptr<gm::SceneObject>> gm::Scene::load_objects(
    std::vector<int> node_ids, tinygltf::Model model,
    std::shared_ptr<SceneObject> parent) {
  std::vector<std::shared_ptr<SceneObject>> objects;
  std::shared_ptr<SceneObject> current;
  objects.reserve(node_ids.size());
  std::cout << "Loading " << node_ids.size() << " objects" << std::endl;
  for (size_t i = 0; i < node_ids.size(); ++i) {
    tinygltf::Node node = model.nodes[node_ids[i]];
    if (node.mesh != -1) {
      current = load_mesh(node, model);
    } else if (node.camera != -1) {
      current = load_camera(node, model);
    } else {
      current = load_empty(node, model);
    }
    current->parent = parent;
    current->children = load_objects(node.children, model, current);
    objects.push_back(current);
  }
  return objects;
}

std::shared_ptr<gm::Mesh> gm::Scene::load_mesh(tinygltf::Node node,
                                               tinygltf::Model model) {
  Vector3f location;
  Quaternionf rotation;
  Vector3f scale;
  load_transform(node, location, rotation, scale);

  std::vector<tinygltf::Mesh> meshes = model.meshes;
  std::vector<tinygltf::Buffer> buffers = model.buffers;
  std::vector<tinygltf::Accessor> accessors = model.accessors;
  std::vector<tinygltf::BufferView> bufferViews = model.bufferViews;

  tinygltf::Mesh mesh = meshes[node.mesh];

  std::vector<Vector3f> positions;
  std::vector<Vector3f> normals;
  std::vector<Vector3i> faces;

  size_t primitiveOffset = 0;

  // NOTE we are combining all primitives of this object to a single mesh
  for (size_t i = 0; i < mesh.primitives.size(); ++i) {
    int positionsIndex = mesh.primitives[i].attributes["POSITION"];
    int normalsIndex = mesh.primitives[i].attributes["NORMAL"];
    int facesIndex = mesh.primitives[i].indices;

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
    size_t facesBufferLength = facesBufferView.byteLength;
    // TODO handle byteStride
    // TODO check data type
    short *facesBytes =
        reinterpret_cast<short *>(facesBuffer.data.data() + facesBufferOffset);
    static const size_t faceCount =
        facesAccessor.count /
        3;  // gltf only supports triangles. No quads or ngons

    std::vector<Vector3i> primitiveFaces(faceCount);
    for (size_t f = 0; f < faceCount; ++f) {
      primitiveFaces[f] = Vector3i(facesBytes[f * 3] + primitiveOffset,
                                   facesBytes[f * 3 + 1] + primitiveOffset,
                                   facesBytes[f * 3 + 2] + primitiveOffset);
    }

    positions.insert(positions.end(), primitivePositions.begin(),
                     primitivePositions.end());
    normals.insert(normals.end(), primitiveNormals.begin(),
                   primitiveNormals.end());
    faces.insert(faces.end(), primitiveFaces.begin(), primitiveFaces.end());

    primitiveOffset += primitivePositions.size();
  }

  return std::shared_ptr<Mesh>(new Mesh(positions, normals, faces, node.name,
                                        location, rotation, scale));
}

std::shared_ptr<gm::SceneObject> gm::Scene::load_empty(tinygltf::Node node,
                                                       tinygltf::Model model) {
  Vector3f location;
  Quaternionf rotation;
  Vector3f scale;
  load_transform(node, location, rotation, scale);

  return std::shared_ptr<SceneObject>(
      new SceneObject(location, rotation, scale, node.name));
}

std::shared_ptr<gm::PerspectiveCamera> gm::Scene::load_camera(
    tinygltf::Node node, tinygltf::Model model) {
  Vector3f location;
  Quaternionf rotation;
  Vector3f scale;
  load_transform(node, location, rotation, scale);

  tinygltf::Camera cameraData = model.cameras[node.camera];

  if (cameraData.type != "perspective") {
    // Error only perspective cameras are supported
    // TODO handle error
  }
  float fov = cameraData.perspective.yfov;

  // TODO fix camera paramters
  return std::shared_ptr<PerspectiveCamera>(new PerspectiveCamera(
      location, Vector3f(1, 0, 0), Vector3f(0, 0, 1), 200, 200, fov));
}

void gm::Scene::load_transform(tinygltf::Node node, Vector3f &location,
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
    std::cout << "loading rotation" << std::endl;
  } else {
    rotation = Quaternionf();
  }
}
