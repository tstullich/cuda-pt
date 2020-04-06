#pragma once

#include <cuda.h>

#include <cmath>
#include <iostream>

#include "matrix.h"
#include "ray.h"

namespace gm {

class PerspectiveCamera {
 public:
  PerspectiveCamera(){};

  PerspectiveCamera(const Vector3f &position, const Vector3f &lookAt,
                    const Vector3f &up, size_t imageWidth, size_t imageHeight,
                    float fov);

  // Compute a new camera ray for the given raster space coordinate
  Ray generate_ray(uint32_t xPos, uint32_t yPos);

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