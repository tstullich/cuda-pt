#pragma once

#include <cuda.h>

#include <iostream>
#include <memory>
#include <string>

#include "camera.h"
#include "image.h"
#include "intersection.h"
#include "pcg_sampler.h"
#include "scene.h"
#include "sphere.h"
#include "triangle.h"

namespace gm {
// The Integrator takes care of the task of evaluating the Light Transport
// Equation and producing a final image. This class serves as the launch point
// for the CUDA kernels used for path tracing.
class Integrator {
 public:
  Integrator(const std::string &filePath);

  void pathtrace();

 private:
  bool intersectScene(const Ray &ray,
                      std::unique_ptr<Intersection> &intersection) const;

  const static uint8_t BLOCK_SIZE = 8;
  const static uint32_t IMAGE_WIDTH = 400;
  const static uint32_t IMAGE_HEIGHT = 300;
  std::unique_ptr<RGBImage> image;
  std::unique_ptr<Scene> scene;
};
}  // namespace gm