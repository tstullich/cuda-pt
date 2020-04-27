#include "camera.h"

gm::PerspectiveCamera::PerspectiveCamera(const float &fov, const float &near, const float &far) : fov(fov),
                                                                                                  near(near),
                                                                                                  far(far) {}

gm::PerspectiveCamera::PerspectiveCamera(const Matrix4x4f &cameraToWorld, const float &fov, const float &near, const float &far) : cameraToWorld(cameraToWorld),
                                                                                                                                   fov(fov),
                                                                                                                                   near(near),
                                                                                                                                   far(far) {}

gm::Ray gm::PerspectiveCamera::generateRay(uint32_t xCoord, uint32_t yCoord,
                                           const Vector2f &sample) {
  // Create slightly randomized position
  float worldX = xCoord + sample.x + 0.5f;
  float worldY = yCoord + sample.y + 0.5f;

  // Do a perspective projection on the x and y coordinate
  Vector3f pos = Vector3f(worldX, worldY, 0.0f);
  pos = projection.multiplyPoint(pos);

  // Transform origin point using the camera-to-world matrix
  Vector3f origin = cameraToWorld.multiplyPoint(Vector3f(0.0f));

  // Create a projection point on the image plane using normalized device
  // coordinates. Move the initial point from the center using two samples
  //float x =
  //    (2.0f * (xCoord + sample.x + 0.5f) / static_cast<float>(imageWidth) - 1.0f) * aspectRatio * scale;
  //float y = (1.0f - 2.0f * (yCoord + sample.y + 0.5f) / static_cast<float>(imageHeight)) * scale;

  // Position vector at the image plane looking in the negative z direction
  //Vector3f direction(x, y, -1.0f);

  // Transform direction vector using the camera-to-world matrix and
  // normalize
  //direction = normalize(cameraToWorld.multiplyVector(direction));

  return {origin, Vector3f(0.0f)};
}

void gm::PerspectiveCamera::initializeMatrices(const size_t &width, const size_t &height) {
}