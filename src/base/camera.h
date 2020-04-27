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
  /// Default constructor in case the glTF scene does not contain a camera
  PerspectiveCamera(const float &fov, const float &near, const float &far);

  PerspectiveCamera(const Matrix4x4f &cameraToWorld, const float &fov, const float &near, const float &far);

  /// Compute a new camera ray for the given raster space coordinate. Also
  /// requires a sample to generate sampled coordinates
  Ray generateRay(uint32_t xPos, uint32_t yPos, const Vector2f &sample);

  // Allow setting the image dimensions
  void initializeMatrices(const size_t &width, const size_t &height);

 private:
  // Projection plane's near and far positions
  float near;
  float far;

  float fov;

  Matrix4x4f cameraToWorld;
  Matrix4x4f projection;
};
}  // namespace gm
