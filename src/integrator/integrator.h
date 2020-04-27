#pragma once

#include <cuda.h>

#include <iostream>
#include <memory>
#include <string>

#include "bvh_dummy.h"
#include "camera.h"
#include "image.h"
#include "intersection.h"
#include "pcg_sampler.h"
#include "scene.h"
#include "triangle.h"

namespace gm {
// The Integrator takes care of the task of evaluating the Light Transport
// Equation and producing a final image. This class serves as the launch point
// for the CUDA kernels used for path tracing.

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

class Integrator {
 public:
  Integrator(const std::string &filePath, const RenderOptions &options);

  void pathtrace();

 private:
  const static uint8_t BLOCK_SIZE = 8;
  std::unique_ptr<RGBImage> image;
  std::unique_ptr<Scene> scene;
  std::unique_ptr<BVHDummy> bvh;
  RenderOptions options;
};
}// namespace gm