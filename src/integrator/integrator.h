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

  void integrate();

 private:
  const static int BLOCK_SIZE = 8;
  const static int IMAGE_WIDTH = 400;
  const static int IMAGE_HEIGHT = 300;
  std::unique_ptr<RGBImage> image;
  PerspectiveCamera camera;
};
}  // namespace gm