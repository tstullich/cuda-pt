#pragma once

#include <cuda.h>

#include <cmath>
#include <iostream>

#include "ray.h"
#include "transform.h"

namespace gm {
/// Struct that stores innformation regarding a camera's
/// aperture. The aperture information passed in should be
/// in inches, not millimeters.
struct FilmInfo {
  FilmInfo(float width, float height) {
    apertureWidth = width * INCH_TO_MM;
    apertureHeight = height * INCH_TO_MM;
  }

  float apertureWidth;
  float apertureHeight;

  const float INCH_TO_MM = 25.4f;
};

class PerspectiveCamera {
 public:
  PerspectiveCamera();

  PerspectiveCamera(const FilmInfo &filmInfo, size_t imageWidth,
                    size_t imageHeight, float focalLength);

  Ray generate_ray();

 private:
  // Settings for the near and far clipping planes
  float nearClip = 0.1f;   // Set slightly in front of the camera position
  float farClip = 1000.f;  // TODO Figure out sensible setting for this
  float focalLength;       // In millimeters
};
}  // namespace gm