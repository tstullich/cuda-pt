#include "camera.h"

gm::PerspectiveCamera::PerspectiveCamera(const float &fov) {
  scale = tanf((fov * 0.5f) * M_PI / 180.0f);

  //cameraToWorld[0][0]
}

gm::PerspectiveCamera::PerspectiveCamera(const Matrix4x4f &cameraToWorld,
                                         const float &fov) : cameraToWorld(cameraToWorld) {
  // Set the scaling factor based on the fov
  scale = tanf((fov * 0.5f) * M_PI / 180.0f);
}

gm::Ray gm::PerspectiveCamera::generate_ray(uint32_t xCoord, uint32_t yCoord,
                                            const Vector2f &sample) {
  // Transform origin point using the camera-to-world matrix
  Vector3f origin = cameraToWorld.multiplyPoint(Vector3f(0.0f));

  // Create a projection point on the image plane using normalized device
  // coordinates. Move the initial point from the center using two samples
  float x =
      (2.0f * (xCoord + sample.x + 0.5f) / static_cast<float>(imageWidth) - 1.0f) * aspectRatio * scale;
  float y = (1.0f - 2.0f * (yCoord + sample.y + 0.5f) / static_cast<float>(imageHeight)) * scale;

  // Position vector at the image plane looking in the negative z direction
  Vector3f direction(x, y, -1.0f);

  // Transform direction vector using the camera-to-world matrix and
  // normalize
  direction = normalize(cameraToWorld.multiplyVector(direction));

  return {origin, direction};
}

void gm::PerspectiveCamera::setImagePlane(const size_t &width,
                                          const size_t &height) {
  imageWidth = width;
  imageHeight = height;
  aspectRatio = static_cast<float>(imageWidth) / imageHeight;
}