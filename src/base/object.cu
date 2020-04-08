#include "object.h"
#include "vector.h"

gm::Object::Object(const Vector3f location, const Vector3f scale,
                   const std::string name) {
  this->location = location;
  this->scale = scale;
  this->name = name;
}