#pragma once

#include <cuda.h>

#include <iostream>
#include <memory>

#include "application.h"
#include "bvh_dummy.h"
#include "camera.h"
#include "image.h"
#include "intersection.h"
#include "pcg_sampler.h"
#include "ray.h"
#include "scene.h"

namespace gm {

class RenderOptions;
class BVHDummy;
class Scene;

// The Integrator takes care of the task of evaluating the Light Transport
// Equation and producing a final image. This class serves as the launch point
// for the CUDA kernels used for path tracing.
class Integrator {
 public:
  Integrator() {};

  void pathtrace(const std::unique_ptr<Scene> &scene,
                 const std::unique_ptr<BVHDummy> &bvh,
                 std::unique_ptr<RGBImage> &image,
                 const RenderOptions &options);
};
}// namespace gm