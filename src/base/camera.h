#pragma once

#include <cuda.h>

#include "ray.h"
#include "transform.h"
#include "vector.h"

namespace gm {
class PerspectiveCamera {
 public:
  __device__ PerspectiveCamera(const Vector3f &position, const Vector3f &target,
                               const Vector3f &up, size_t width, size_t height,
                               float fov, float lensRadius,
                               float focalDistance);

  __device__ Ray generate_ray();

 private:
  float lensRadius;
  float focalDistance;

  Vector3f pMin;
  Vector3f pMax;

  Transform cameraToScreen, rasterToCamera;
  Transform screenToRaster, rasterToScreen;
};
}  // namespace gm