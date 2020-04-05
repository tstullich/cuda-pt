#include "camera.h"

gm::PerspectiveCamera::PerspectiveCamera() {
  // The camera should not be initialized with this constructor
}

gm::PerspectiveCamera::PerspectiveCamera(const FilmInfo &filmInfo,
                                         size_t imageWidth, size_t imageHeight,
                                         float focalLength)
    : focalLength(focalLength) {
  // Compute the aspect ratio of the final image. This needs to
  // be preserved or else the conversion from screen space to raster space
  // leads to stretching of the image.
  float filmAspectRatio = filmInfo.apertureWidth / filmInfo.apertureHeight;
  float deviceAspectRatio = static_cast<float>(imageWidth) / imageHeight;

  float top = ((filmInfo.apertureHeight / 2.0f) / focalLength) * nearClip;
  float right = ((filmInfo.apertureWidth / 2.0f) / focalLength) * nearClip;

  float bottom = -top;
  float left = -right;

  std::cout << "Screen window coordinates: (" << bottom << ", " << left << ", "
            << top << ", " << right << ")" << std::endl;
  std::cout << "Film aspect ratio: " << filmAspectRatio << std::endl;
  std::cout << "Device aspect ratio: " << deviceAspectRatio << std::endl;
  std::cout << "Angle of view: "
            << 2.0f * atan(((filmInfo.apertureWidth / 2.0f) / focalLength) *
                           180.0f / M_PI)
            << std::endl;
}

gm::Ray gm::PerspectiveCamera::generate_ray() { return gm::Ray(); }