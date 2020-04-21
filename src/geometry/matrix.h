#pragma once

#include <cstring>
#include <iostream>

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
};

typedef Matrix4x4<float> Matrix4x4f;

/// Functions to build linear transformation matrices
template<typename T>
Matrix4x4<T> scaleMatrix(const Vector3<T> &scale) {
  Matrix4x4<T> mat;
  mat[0][0] = scale.x;
  mat[1][1] = scale.y;
  mat[2][2] = scale.z;
  return mat;
}

template<typename T>
Matrix4x4<T> translationMatrix(const Vector3<T> &translation) {
  Matrix4x4<T> mat;
  mat[3][0] = translation.x;
  mat[3][1] = translation.y;
  mat[3][2] = translation.z;
  return mat;
}

template<typename T>
Matrix4x4<T> invert(Matrix4x4<T> m) {
  Matrix4x4<T> mat;
  for (size_t column = 0; column < 4; ++column) {
    // Swap row in case our pivot point is not working
    if (m[column][column] == 0) {
      size_t big = column;
      for (size_t row = 0; row < 4; ++row) {
        if (fabs(m[row][column]) > fabs(m[big][column])) {
          big = row;
        }
      }
      if (big == column) {
        std::cout << "Singular matrix! Returning identity" << std::endl;
        return Matrix4x4<T>();
      } else {
        // Swap rows
        for (size_t j = 0; j < 4; ++j) {
          std::swap(m[column][j], m[big][j]);
          std::swap(mat.m[column][j], mat.m[big][j]);
        }
      }
    }
    // Set each row in the column to 0
    for (size_t row = 0; row < 4; ++row) {
      if (row != column) {
        T coeff = m[row][column] / m[column][column];
        if (coeff != 0) {
          for (size_t j = 0; j < 4; ++j) {
            m[row][j] -= coeff * m[column][j];
            mat.m[row][j] -= coeff * mat.m[column][j];
          }
          // Set the element to 0 for safety
          m[row][column] = 0;
        }
      }
    }
  }
  // Set each element of the diagonal to 1
  for (size_t row = 0; row < 4; ++row) {
    for (size_t column = 0; column < 4; ++column) {
      mat.m[row][column] /= m[row][row];
    }
  }

  return mat;
}
}// namespace gm