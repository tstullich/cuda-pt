#pragma once

#include "ray.h"

namespace gm {
class Camera {
  Camera();

  Ray generate_ray();
};
}  // namespace gm
