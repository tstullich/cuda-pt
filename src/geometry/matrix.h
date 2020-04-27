#pragma once

#include <cstring>
#include <iostream>
#include <ostream>
#include <vector>

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

  /// Initialize a Matrix using a vector with 16 entries. This is useful
  /// for loading matrices from glTF scene descriptions. Vector is assumed
  /// to store its entries in row-major form. Because glTF stores its matrices
  /// in column-major form we also need to transpose the incoming matrix
  Matrix4x4(const std::vector<double> &matrix) {
    for (size_t row = 0; row < 4; ++row) {
      for (size_t col = 0; col < 4; ++col) {
        size_t idx = (row * 4) + col;
        // Store matrix entry in transposed form
        m[col][row] = static_cast<float>(matrix[idx]);
      }
    }
  }

  const T *operator[](uint8_t i) const { return m[i]; }

  T *operator[](uint8_t i) { return m[i]; }

  Matrix4x4<T> operator*(const Matrix4x4<T> &mat) const {
    Matrix4x4<T> result;
    for (size_t row = 0; row < 4; ++row) {
      for (size_t col = 0; col < 4; ++col) {
        result[row][col] = m[row][0] * mat.m[0][col] + m[row][1] * mat.m[1][col] +
                           m[row][2] * mat.m[2][col] + m[row][3] * mat.m[3][col];
      }
    }
    return result;
  }

  /// Multiply a point using the inner 3x3 matrix. We implicitly assume that our
  /// points have homogeneous coordinates so we need to normalize with the w
  /// component if needed. By convention the 1x3 vector is on the left-hand side,
  /// so we need to multiply through the columns of the matrix
  Vector3<T> multiplyPoint(const Vector3<T> &lhs) const {
    T x = m[0][0] * lhs.x + m[1][0] * lhs.y + m[2][0] * lhs.z + m[3][0];
    T y = m[0][1] * lhs.x + m[1][1] * lhs.y + m[2][1] * lhs.z + m[3][1];
    T z = m[0][2] * lhs.x + m[1][2] * lhs.y + m[2][2] * lhs.z + m[3][2];
    T w = m[0][3] * lhs.x + m[1][3] * lhs.y + m[2][3] * lhs.z + m[3][3];

    // Normalize points by w if needed
    return (w == 1.0f || w == 0.0f) ? Vector3<T>(x, y, z)
                                    : Vector3<T>(x, y, z) / w;
  }

  /// Multiply a vector using the inner 3x3 matrix. By convention the vector is
  /// on the left-hand side, so we need to use the columns of the matrix for multiplication
  Vector3<T> multiplyVector(const Vector3<T> &lhs) const {
    return Vector3<T>(m[0][0] * lhs.x + m[1][0] * lhs.y + m[2][0] * lhs.z,
                      m[0][1] * lhs.x + m[1][1] * lhs.y + m[2][1] * lhs.z,
                      m[0][2] * lhs.x + m[1][2] * lhs.y + m[2][2] * lhs.z);
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

template <typename T>
inline std::ostream &operator<<(std::ostream &stream, const Matrix4x4<T> &mat) {
  stream << "Matrix4x4(" << std::endl << 
    mat.m[0][0] << "," << mat.m[0][1]<< "," << mat.m[0][2] << ","<< mat.m[0][3] << std::endl <<
    mat.m[1][0] << "," << mat.m[1][1]<< "," << mat.m[1][2] << ","<< mat.m[1][3] << std::endl <<
    mat.m[2][0] << "," << mat.m[2][1]<< "," << mat.m[2][2] << ","<< mat.m[2][3] << std::endl <<
    mat.m[3][0] << "," << mat.m[3][1]<< "," << mat.m[3][2] << ","<< mat.m[3][3] << ")"<< std::endl;

  return stream;
}

}// namespace gm
