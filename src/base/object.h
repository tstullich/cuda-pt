#include <memory>
#include <string>
#pragma once

#include <vector>

#include "vector.h"

namespace gm {
class Object {
 public:
  Object(const Vector3f location = Vector3f(),
         const Vector3f scale = Vector3f(1.f, 1.f, 1.f),
         const std::string name = "");

  Vector3f get_location();
  std::string name;
  Vector3f location;
  Vector3f scale;
  std::shared_ptr<Object> parent;
  std::vector<std::shared_ptr<Object>> children;
  virtual bool isMesh() { return false; }
  virtual bool isCamera() { return false; }
  virtual bool isEmpty() { return true; }
};
}  // namespace gm