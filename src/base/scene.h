#pragma once

#include <memory>

#include "camera.h"
#include "object.h"

namespace gm {
class Scene {
  Scene(){};
  Scene(std::string filepath);
  void addObject(std::shared_ptr<Object> o);
  std::shared_ptr<PerspectiveCamera> getCamera();
};
}  // namespace gm