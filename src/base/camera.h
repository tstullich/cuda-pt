#pragma once

#include <cuda.h>

#include <cmath>
#include <iostream>

#include "matrix.h"
#include "ray.h"

namespace gm {

enum ScanMode { Fill, Overscan };

/// Struct that stores innformation regarding a camera's
/// aperture. The aperture information passed in should be
/// in inches, not millimeters.
struct FilmInfo {
  FilmInfo(float width, float height, ScanMode scanMode) {
    apertureWidth = width * INCH_TO_MM;
    apertureHeight = height * INCH_TO_MM;
    mode = scanMode;
  }

  float apertureWidth;
  float apertureHeight;
  ScanMode mode;

  // Conversion from inches to mm
  const float INCH_TO_MM = 25.4f;
};

class PerspectiveCamera {
 public:
  PerspectiveCamera(){};

  PerspectiveCamera(const FilmInfo &filmInfo, size_t imageWidth,
                    size_t imageHeight, float focalLength);

  // Compute a new camera ray
  Ray generate_ray(float x, float y);

 private:
  // Settings for the near and far clipping planes
  float nearClip = 0.1f;  // Set slightly in front of the camera position
  float imageAspectRatio;
  float scale;

  size_t imageWidth;
  size_t imageHeight;

  Vector3f top;
  Vector3f bottom;

  Matrix4x4f cameraToWorld;
};
}  // namespace gm