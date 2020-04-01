#pragma once

#include <memory>

#include "image.h"

namespace gm {
// The Integrator takes care of the task of evaluating the Light Transport
// Equation and producing a final image. This class serves as the launch point
// for the CUDA kernels used for path tracing.
class Integrator {
 public:
  Integrator();

  void trace();

 private:
  std::unique_ptr<RGBImage> image;
};
}  // namespace gm