#pragma once

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

  PerspectiveCamera(const Matrix4x4f &modelMatrix, const float &fov, const float &near, const float &far);

  /// Compute a new camera ray for the given raster space coordinate. Also
  /// requires a sample to generate sampled coordinates
  Ray generateRay(uint32_t pixelX, uint32_t pixelY, const Vector2f &sample);

  void init(const uint32_t &width, const uint32_t &height);

 private:
  Matrix4x4f buildView();

  Matrix4x4f buildProjection();

  // Projection plane's near and far positions
  float near;
  float far;
  float fov;

  // Variables for the projection matrix
  float aspectRatio;
  float scaleFactor;

  uint32_t width;
  uint32_t height;

  Matrix4x4f model;
};
}  // namespace gm
