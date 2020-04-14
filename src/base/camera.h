#pragma once

#include <cuda.h>

#include <cmath>
#include <iostream>

#include "matrix.h"
#include "ray.h"
#include "scene_object.h"

namespace gm {

/// A camera class that implements a camera with perspective projection.
/// When generating primary camera rays the convention is to form rays
/// at the image plane and then transform them using the cameraToWorld
/// matrix provided in this class.
class PerspectiveCamera : public SceneObject {
 public:
  PerspectiveCamera(){};

  PerspectiveCamera(const Vector3f &position, const Vector3f &lookAt,
                    const Vector3f &up, size_t imageWidth, size_t imageHeight,
                    float fov);

  // This constructor is based on the information that is included in a glTF
  // file.
  PerspectiveCamera(const Vector3f &location, const Quaternionf &rotation,
                    const float &fov, const std::string &name);

  // Compute a new camera ray for the given raster space coordinate
  Ray generate_ray(uint32_t xPos, uint32_t yPos);

  // Allow setting the image dimensions
  void setImageSize(const size_t &width, const size_t &height) {
    imageWidth = width;
    imageHeight = height;
    aspectRatio = static_cast<float>(imageWidth) / imageHeight;
  }

  virtual bool isMesh() { return false; }
  virtual bool isCamera() { return true; }
  virtual bool isEmpty() { return false; }

 private:
  void setCameraToWorld(const Vector3f &position, const Vector3f &lookAt,
                        const Vector3f &up);

  float aspectRatio;
  float scale;

  size_t imageWidth;
  size_t imageHeight;

  Matrix4x4f cameraToWorld;
};
}  // namespace gm
