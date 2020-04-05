#include "camera.h"

gm::PerspectiveCamera::PerspectiveCamera(const FilmInfo &filmInfo,
                                         size_t imageWidth, size_t imageHeight,
                                         float fov, float focalLength) {
  // Compute the aspect ratio of the final image. This needs to
  // be preserved or else the conversion from screen space to raster space
  // leads to stretching of the image.
  float filmAspectRatio = filmInfo.apertureWidth / filmInfo.apertureHeight;
  imageAspectRatio = static_cast<float>(imageWidth) / imageHeight;

  float t = ((filmInfo.apertureHeight / 2.0f) / focalLength) * nearClip;
  float r = ((filmInfo.apertureWidth / 2.0f) / focalLength) * nearClip;

  // Assign fill mode
  float xscale = 1.0f;
  float yscale = 1.0f;
  if (filmInfo.mode == ScanMode::Fill) {
    if (filmAspectRatio > imageAspectRatio) {
      xscale = imageAspectRatio / filmAspectRatio;
    } else {
      yscale = filmAspectRatio / imageAspectRatio;
    }
  } else {
    if (filmAspectRatio > imageAspectRatio) {
      yscale = filmAspectRatio / imageAspectRatio;
    } else {
      xscale = imageAspectRatio / filmAspectRatio;
    }
  }

  // Scale the top and right coordinate according to fill mode
  t *= yscale;
  r *= xscale;

  scale = tan((fov * 0.5f) * M_PI / 180.0f);

  top = Vector3f(r, t, nearClip);      // (r, t, n)
  bottom = Vector3f(-r, -t, nearClip); // (l, b, n)

  std::cout << "Top Vector: (" << top.x << ", " << top.y << ", " << top.z << ")"
            << std::endl;
  std::cout << "Bottom Vector: (" << bottom.x << ", " << bottom.y << ", "
            << bottom.z << ")" << std::endl;
  std::cout << "Film aspect ratio: " << filmAspectRatio << std::endl;
  std::cout << "Device aspect ratio: " << imageAspectRatio << std::endl;
  std::cout << "Angle of view: "
            << 2.0f * atan(((filmInfo.apertureWidth / 2.0f) / focalLength) *
                           180.0f / M_PI)
            << std::endl;
}

gm::Ray gm::PerspectiveCamera::generate_ray(int xCoord, int yCoord) {
  Vector3f origin;
  Vector3f direction;

  float x = (2.0f * (xCoord + 0.5f) / static_cast<float>(imageWidth) - 1.0f) *
            imageAspectRatio * scale;
  float y =
      (1.0f - 2.0f * (yCoord + 0.5f) / static_cast<float>(imageHeight)) * scale;

  return Ray(origin, normalize(direction));
}