#include "scene.h"

#define TINYGLTF_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION

#include "tiny_gltf.h"

using namespace tinygltf;

__host__ gm::Scene::Scene(std::string filepath) {
  Model model;
  TinyGLTF loader;
  std::string err;
  std::string warn;

  bool ret = loader.LoadASCIIFromFile(&model, &err, &warn, filepath);
  // bool ret = loader.LoadBinaryFromFile(&model, &err, &warn, argv[1]); // for
  // binary glTF(.glb)

  if (!warn.empty()) {
    printf("Warn: %s\n", warn.c_str());
  }

  if (!err.empty()) {
    printf("Err: %s\n", err.c_str());
  }

  if (!ret) {
    printf("Failed to parse glTF\n");
  }
}
