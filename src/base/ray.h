#pragma once

#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

namespace gm {
struct Ray {
  Ray(glm::vec3 origin, glm::vec3 direction)
      : origin(origin), direction(direction) {}

  glm::vec3 origin;
  glm::vec3 direction;
};
}  // namespace gm