#pragma once

#include <cuda.h>

#include <iostream>
#include <memory>

#include "camera.h"
#include "image.h"

namespace gm {
// The Integrator takes care of the task of evaluating the Light Transport
// Equation and producing a final image. This class serves as the launch point
// for the CUDA kernels used for path tracing.
class Integrator {
 public:
  Integrator();

  void pathtrace();

  void integrate();

 private:
  const static uint8_t BLOCK_SIZE = 8;
  const static uint32_t IMAGE_WIDTH = 400;
  const static uint32_t IMAGE_HEIGHT = 300;
  std::unique_ptr<RGBImage> image;
  std::shared_ptr<PerspectiveCamera> camera;
};
}  // namespace gm