#pragma once

#include <cuda.h>

#include <cstring>

#include "vector.h"

namespace gm {

/// An implementation of a 4x4 matrix. This class can be used to work with
/// the affine transformations below. The matrix is represented in row-major
/// form and the transformations operate on a right-handed Cartesian coordinate
/// system.
template <typename T>
class Matrix4x4 {
 public:
  Matrix4x4() {}

  const T *operator[](uint8_t i) const { return m[i]; }

  T *operator[](uint8_t i) { return m[i]; }

  /// Multiply a point using the inner 3x3 matrix. We implicitly assume that our
  /// points have homogeneous coordinates so we need to normalize the w
  /// component
  Vector3<T> multiplyPoint(const Vector3<T> &rhs) const {
    T x = m[0][0] * rhs.x + m[1][0] * rhs.y + m[2][0] * rhs.z + m[3][0];
    T y = m[0][1] * rhs.x + m[1][1] * rhs.y + m[2][1] * rhs.z + m[3][1];
    T z = m[0][2] * rhs.x + m[1][2] * rhs.y + m[2][2] * rhs.z + m[3][2];
    T w = m[0][3] * rhs.x + m[1][3] * rhs.y + m[2][3] * rhs.z + m[3][3];

    // Normalize points by w if needed
    return (w == 1.0f || w == 0.0f) ? Vector3<T>(x, y, z)
                                    : Vector3<T>(x, y, z) / w;
  }

  /// Multiply a vector using the inner 3x3 matrix
  Vector3<T> multiplyVector(const Vector3<T> &rhs) const {
    return Vector3<T>(m[0][0] * rhs.x + m[1][0] * rhs.y + m[2][0] * rhs.z,
                      m[0][1] * rhs.x + m[1][1] * rhs.y + m[2][1] * rhs.z,
                      m[0][2] * rhs.x + m[1][2] * rhs.y + m[2][2] * rhs.z);
  }

  // Initialize matrix as the identity matrix
  T m[4][4] = {{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 1}};
};  // namespace gm

typedef Matrix4x4<float> Matrix4x4f;
}  // namespace gm