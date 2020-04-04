#include "camera.h"

__device__ gm::PerspectiveCamera::PerspectiveCamera(const Vector3f &position,
                                                    const Vector3f &target,
                                                    const Vector3f &up,
                                                    size_t width, size_t height,
                                                    float fov, float lensRadius,
                                                    float focalDistance)
    : lensRadius(lensRadius), focalDistance(focalDistance) {

  // Figure out where cameraToScreen comes from
  cameraToScreen = perspective(fov, 1e-2f, 1000.0f);

  // Create projective camera transformations
  screenToRaster = scale(width, height, 1.0f);
  rasterToScreen = screenToRaster.inverse();
  rasterToCamera = cameraToScreen.inverse() * rasterToScreen;
}

__device__ gm::Ray gm::PerspectiveCamera::generate_ray() { return gm::Ray(); }