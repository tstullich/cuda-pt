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

  Matrix4x4(float mat[4][4]) {
    // TODO Check if this needs to get changed to cudaMemcpy()
    memcpy(m, mat, 16 * sizeof(float));
  }

  Matrix4x4(float t00, float t01, float t02, float t03, float t10, float t11,
            float t12, float t13, float t20, float t21, float t22, float t23,
            float t30, float t31, float t32, float t33) {
    m[0][0] = t00;
    m[0][1] = t01;
    m[0][2] = t02;
    m[0][3] = t03;
    m[1][0] = t10;
    m[1][1] = t11;
    m[1][2] = t12;
    m[1][3] = t13;
    m[2][0] = t20;
    m[2][1] = t21;
    m[2][2] = t22;
    m[2][3] = t23;
    m[3][0] = t30;
    m[3][1] = t31;
    m[3][2] = t32;
    m[3][3] = t33;
  }

  const T *operator[](uint8_t i) const { return m[i]; }

  T *operator[](uint8_t i) { return m[i]; }

  bool operator==(const Matrix4x4<T> &m2) const {
    for (uint8_t i = 0; i < 4; ++i) {
      for (uint8_t j = 0; j < 4; ++j) {
        if (m[i][j] != m2.m[i][j]) {
          return false;
        }
      }
    }
    return true;
  }

  bool operator!=(const Matrix4x4<T> &m2) const {
    for (uint8_t i = 0; i < 4; ++i) {
      for (uint8_t j = 0; j < 4; ++j) {
        if (m[i][j] != m2.m[i][j]) {
          return true;
        }
      }
    }
    return false;
  }

  Matrix4x4<T> operator*(const Matrix4x4<T> &rhs) const {
    Matrix4x4<T> r;
    for (uint8_t i = 0; i < 4; ++i) {
      for (uint8_t j = 0; j < 4; ++j) {
        r.m[i][j] = m[i][0] * rhs.m[0][j] + m[i][1] * rhs.m[1][j] +
                    m[i][2] * rhs.m[2][j] + m[i][3] * rhs.m[3][j];
      }
    }
    return r;
  }

  /// Multiply a point using the inner 3x3 matrix. We implicitly assume that our
  /// points have homogeneous coordinates so we need to normalize the w
  /// component
  Vector3<T> multiplyPoint(const Vector3<T> &rhs) const {
    T x = m[0][0] * rhs.x + m[1][0] * rhs.y + m[2][0] * rhs.z + m[3][0];
    T y = m[0][1] * rhs.x + m[1][1] * rhs.y + m[2][1] * rhs.z + m[3][1];
    T z = m[0][2] * rhs.x + m[1][2] * rhs.y + m[2][2] * rhs.z + m[3][2];
    T w = m[0][3] * rhs.x + m[1][3] * rhs.y + m[2][3] * rhs.z + m[3][3];

    if (w == 1.0f) {
      return Vector3<T>(x, y, z);
    } else {
      // Normalize components by w
      return Vector3<T>(x, y, z) / w;
    }
  }

  /// Multiply a vector using the inner 3x3 matrix
  Vector3<T> multiplyVector(const Vector3<T> &rhs) const {
    return Vector3<T>(m[0][0] * rhs.x + m[1][0] * rhs.y + m[2][0] * rhs.z,
                      m[0][1] * rhs.x + m[1][1] * rhs.y + m[2][1] * rhs.z,
                      m[0][2] * rhs.x + m[1][2] * rhs.y + m[2][2] * rhs.z);
  }

  Matrix4x4<T> transpose() const {
    return Matrix4x4<T>(m[0][0], m[1][0], m[2][0], m[3][0], m[0][1], m[1][1],
                        m[2][1], m[3][1], m[0][2], m[1][2], m[2][2], m[3][2],
                        m[0][3], m[1][3], m[2][3], m[3][3]);
  }

  // Initialize matrix as the identity matrix
  T m[4][4] = {{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 1}};
};

typedef Matrix4x4<float> Matrix4x4f;
}  // namespace gm