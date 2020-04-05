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

  bool operator==(const Matrix4x4 &m2) const {
    for (uint8_t i = 0; i < 4; ++i) {
      for (uint8_t j = 0; j < 4; ++j) {
        if (m[i][j] != m2.m[i][j]) {
          return false;
        }
      }
    }
    return true;
  }

  bool operator!=(const Matrix4x4 &m2) const {
    for (uint8_t i = 0; i < 4; ++i) {
      for (uint8_t j = 0; j < 4; ++j) {
        if (m[i][j] != m2.m[i][j]) {
          return true;
        }
      }
    }
    return false;
  }

  Matrix4x4 operator*(const Matrix4x4 &rhs) const {
    Matrix4x4 r;
    for (uint8_t i = 0; i < 4; ++i) {
      for (uint8_t j = 0; j < 4; ++j) {
        r.m[i][j] = m[i][0] * rhs.m[0][j] + m[i][1] * rhs.m[1][j] +
                    m[i][2] * rhs.m[2][j] + m[i][3] * rhs.m[3][j];
      }
    }
    return r;
  }

  void multVecMatrix(const Vector3<T> &src, Vector3<T> &dst) const {
    dst.x = src.x * m[0][0] + src.y * m[1][0] + src.z * m[2][0] + m[3][0];
    dst.y = src.x * m[0][1] + src.y * m[1][1] + src.z * m[2][1] + m[3][1];
    dst.z = src.x * m[0][2] + src.y * m[1][2] + src.z * m[2][2] + m[3][2];
    T w = src.x * m[0][3] + src.y * m[1][3] + src.z * m[2][3] + m[3][3];
    if (w != 1 && w != 0) {
      dst.x = x / w;
      dst.y = y / w;
      dst.z = z / w;
    }
  }

  void multDirMatrix(const Vector3<T> &src, Vector3<T> &dst) const {
    dst.x = src.x * m[0][0] + src.y * m[1][0] + src.z * m[2][0];
    dst.y = src.x * m[0][1] + src.y * m[1][1] + src.z * m[2][1];
    dst.z = src.x * m[0][2] + src.y * m[1][2] + src.z * m[2][2];
  }

  T m[4][4] = {{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 1}};
};

typedef Matrix4x4<float> Matrix4x4f;
}  // namespace gm