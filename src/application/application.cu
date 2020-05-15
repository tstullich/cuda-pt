#include "application.h"

gm::Application::Application(const std::string &filePath) {
  // Set our rendering options
  options = std::unique_ptr<RenderOptions>(new RenderOptions(400, 300, 4));

  // Load our scene
  scene = std::unique_ptr<Scene>(new Scene(filePath, *options));

  // Build BVH
  bvh = std::unique_ptr<BVHDummy>(new BVHDummy(scene));

  // Initialize the output buffer
  image = std::unique_ptr<RGBImage>(new RGBImage(options->imageWidth, options->imageHeight));

  // Create our integrator
  integrator = std::unique_ptr<Integrator>(new Integrator());
}

void gm::Application::run() {
  integrator->pathtrace(scene, bvh, image, *options);
  image->writePNG("test.png");
}
