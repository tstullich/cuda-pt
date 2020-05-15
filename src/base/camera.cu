#include "camera.h"

gm::PerspectiveCamera::PerspectiveCamera(const float &fov, const float &near, const float &far) : fov(fov),
                                                                                                  near(near),
                                                                                                  far(far) {
  scaleFactor = 1.0f / tanf(fov * 0.5f);
}

gm::PerspectiveCamera::PerspectiveCamera(const Matrix4x4f &modelMatrix, const float &fov, const float &near, const float &far) : model(modelMatrix),
                                                                                                                                 fov(fov),
                                                                                                                                 near(near),
                                                                                                                                 far(far) {
  scaleFactor = 1.0f / tanf(fov * 0.5f);
}

gm::Matrix4x4f gm::PerspectiveCamera::buildView() {
  Vector3f position(0.0f, 0.0f, 1.0f);
  Vector3f lookAt(0.0f, 0.0f, 0.0f);
  Vector3f up(0.0f, 1.0f, 0.0f);

  Vector3f forward = normalize(position - lookAt);
  Vector3f right = cross(normalize(up), forward);
  Vector3f newUp = cross(forward, right);

  Matrix4x4f orientation;
  orientation[0][0] = right.x;
  orientation[0][1] = right.y;
  orientation[0][2] = right.z;

  orientation[1][0] = newUp.x;
  orientation[1][1] = newUp.y;
  orientation[1][2] = newUp.z;

  orientation[2][0] = forward.x;
  orientation[2][1] = forward.y;
  orientation[2][2] = forward.z;

  Matrix4x4f translation;
  translation[0][3] = -position.x;
  translation[1][3] = -position.y;
  translation[2][3] = -position.z;

  return orientation * translation;
}

gm::Matrix4x4f gm::PerspectiveCamera::buildProjection() {
  Matrix4x4f frustum;
  frustum[0][0] = scaleFactor / aspectRatio;
  frustum[1][1] = scaleFactor;
  frustum[2][2] = -(far + near) / (far - near);
  frustum[2][3] = -(2.0f * far * near) / (far - near);
  frustum[3][2] = -1.0f;
  frustum[3][3] = 0;
  return frustum;
}

gm::Ray gm::PerspectiveCamera::generateRay(uint32_t pixelX, uint32_t pixelY,
                                           const Vector2f &sample) {
  float x = (2.0f * (pixelX + sample.x + 0.5f) / static_cast<float>(width) - 1.0f) * aspectRatio * scaleFactor;
  float y = (1.0f - 2.0f * (pixelX + sample.y + 0.5f) / static_cast<float>(height)) * scaleFactor;

  Vector3f origin(x, y, 0.0f);
  origin = model.multiplyPoint(origin);

  Vector3f direction(x, y, 1.0f);
  direction = model.multiplyVector(direction);

  return {origin, direction};
}

void gm::PerspectiveCamera::init(const uint32_t &imageWidth, const uint32_t &imageHeight) {
  width = imageWidth;
  height = imageHeight;
  aspectRatio = static_cast<float>(width) / height;

  Matrix4x4f view = buildView();
  Matrix4x4f projection = buildProjection();
  auto result = projection * view * model;
  model = invert(result);
}