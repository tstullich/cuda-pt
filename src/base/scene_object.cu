#include "scene_object.h"
#include "vector.h"

gm::SceneObject::SceneObject(const Vector3f &location,
                             const Quaternionf &rotation, const Vector3f &scale,
                             const std::string &name)
    : location(location), rotation(rotation), scale(scale), name(name) {}