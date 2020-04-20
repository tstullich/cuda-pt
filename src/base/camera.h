#pragma once

#include <cuda.h>

#include <cmath>

#include "matrix.h"
#include "quaternion.h"
#include "ray.h"
#include "vector.h"

namespace gm {

/// A camera class that implements a camera with perspective projection.
/// When generating primary camera rays the convention is to form rays
/// at the image plane and then transform them using the cameraToWorld
/// matrix provided in this class.
class PerspectiveCamera {
 public:
  PerspectiveCamera(){};

  // This constructor is based on the information that is included in a glTF
  // file.
  PerspectiveCamera(const Vector3f &location, const Quaternionf &rotation,
                    const float &fov);

  /// Compute a new camera ray for the given raster space coordinate. Also
  /// requires a sample to generate sampled coordinates
  Ray generate_ray(uint32_t xPos, uint32_t yPos, const Vector2f &sample);

  // Allow setting the image dimensions
  void setImagePlane(const size_t &width, const size_t &height);

 private:
  float aspectRatio;
  float scale;

  size_t imageWidth;
  size_t imageHeight;

  Matrix4x4f cameraToWorld;
};
}  // namespace gm
