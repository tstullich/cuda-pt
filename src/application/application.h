#pragma once

#include <memory>
#include <string>

#include "bvh_dummy.h"
#include "image.h"
#include "integrator.h"
#include "scene.h"

namespace gm {

class Integrator;

struct RenderOptions {
  RenderOptions(const uint32_t &imageWidth,
                const uint32_t &imageHeight,
                const uint32_t &samplesPerPixel) : imageWidth(imageWidth),
                                                   imageHeight(imageHeight),
                                                   samplesPerPixel(samplesPerPixel) {}

  uint32_t imageWidth;
  uint32_t imageHeight;
  uint32_t samplesPerPixel;
};


class Application {
 public:
  Application(const std::string &filePath);

  void run();

 private:
  std::unique_ptr<RenderOptions> options;
  std::unique_ptr<BVHDummy> bvh;
  std::unique_ptr<Integrator> integrator;
  std::unique_ptr<RGBImage> image;
  std::unique_ptr<Scene> scene;
};
} // namespace gm