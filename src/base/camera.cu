#include "camera.h"

/// TODO Add focalLength for defocus blur
gm::PerspectiveCamera::PerspectiveCamera(const Vector3f &position,
                                         const Vector3f &lookAt,
                                         const Vector3f &up, size_t imageWidth,
                                         size_t imageHeight, float fov)
    : imageWidth(imageWidth), imageHeight(imageHeight) {
  aspectRatio = static_cast<float>(imageWidth) / imageHeight;
  scale = tan((fov * 0.5f) * M_PI / 180.0f);

  // Create the camera to world matrix to transform rays
  setCameraToWorld(position, lookAt, up);
}

gm::PerspectiveCamera::PerspectiveCamera(const Vector3f &location,
                                         const Quaternionf &rotation,
                                         const float &fov,
                                         const std::string &name)
    : SceneObject(location, rotation, Vector3f(1), name) {
  scale = tan((fov * 0.5f) * M_PI / 180.0f);
  auto rotationMatrix = rotation.toMat4();

  cameraToWorld[0][0] = rotationMatrix[0][0];
  cameraToWorld[0][1] = rotationMatrix[0][1];
  cameraToWorld[0][2] = rotationMatrix[0][2];
  cameraToWorld[1][0] = rotationMatrix[1][0];
  cameraToWorld[1][1] = rotationMatrix[1][1];
  cameraToWorld[1][2] = rotationMatrix[1][2];
  cameraToWorld[2][0] = rotationMatrix[2][0];
  cameraToWorld[2][1] = rotationMatrix[2][1];
  cameraToWorld[2][2] = rotationMatrix[2][2];
  cameraToWorld[3][0] = location.x;
  cameraToWorld[3][1] = location.y;
  cameraToWorld[3][2] = location.z;
  cameraToWorld[3][3] = 1.0f;
}


gm::Ray gm::PerspectiveCamera::generate_ray(uint32_t xCoord, uint32_t yCoord,
                                            const Vector2f &sample) {
  Vector3f origin;
  // Transform origin point using the camera-to-world matrix
  origin = cameraToWorld.multiplyPoint(origin);

  // Create a projection point on the image plane using normalized device
  // coordinates. Move the initial point from the center using two samples
  float x =
      (2.0f * (xCoord + sample.x + 0.5f) / static_cast<float>(imageWidth) -
       1.0f) *
      aspectRatio * scale;
  float y = (1.0f - 2.0f * (yCoord + sample.y + 0.5f) /
                        static_cast<float>(imageHeight)) *
            scale;

  // Position vector at the image plane looking in the negative z direction
  Vector3f direction(x, y, -1.0f);

  // Transform direction vector using the camera-to-world matrix and
  // normalize
  direction = normalize(cameraToWorld.multiplyVector(direction));

  return Ray(origin, direction);
}

void gm::PerspectiveCamera::setCameraToWorld(const Vector3f &position,
                                             const Vector3f &lookAt,
                                             const Vector3f &up) {
  // Create the three basis vectors for the camera orientation
  Vector3f view = normalize(position - lookAt);
  Vector3f right = normalize(cross(normalize(up), view));
  Vector3f newUp = cross(view, right);

  if (right.length() == 0.0f) {
    // Error here because the up vector and viewing direction are the same!.
    // Need to handle error here
  }

  // Set the components of the camera to world transformation matrix
  cameraToWorld[0][0] = right.x;
  cameraToWorld[0][1] = right.y;
  cameraToWorld[0][2] = right.z;
  cameraToWorld[1][0] = newUp.x;
  cameraToWorld[1][1] = newUp.y;
  cameraToWorld[1][2] = newUp.z;
  cameraToWorld[2][0] = view.x;
  cameraToWorld[2][1] = view.y;
  cameraToWorld[2][2] = view.z;
  cameraToWorld[3][0] = position.x;
  cameraToWorld[3][1] = position.y;
  cameraToWorld[3][2] = position.z;
  cameraToWorld[3][3] = 1.0f;
}

void gm::PerspectiveCamera::setImagePlane(const size_t &width,
                                          const size_t &height) {
  imageWidth = width;
  imageHeight = height;
  aspectRatio = static_cast<float>(imageWidth) / imageHeight;
}