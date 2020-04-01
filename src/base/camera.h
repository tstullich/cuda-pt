#pragma once

#include <cuda.h>

#include "ray.h"

namespace gm {
class PerspectiveCamera {
 public:
  PerspectiveCamera();

  __device__ Ray generate_ray();
};
}  // namespace gm
