#pragma once

#include <memory>
#include <string>
#include <vector>

#include "quaternion.h"
#include "vector.h"

namespace gm {
class SceneObject {
 public:
  SceneObject(const Vector3f &location = Vector3f(),
              const Quaternionf &rotation = Quaternionf(),
              const Vector3f &scale = Vector3f(1.0f, 1.0f, 1.0f),
              const std::string &name = "");

  Vector3f get_location();
  std::string name;
  Vector3f location;
  Quaternionf rotation;
  Vector3f scale;
};
}  // namespace gm