#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include "math.h"  // For built-in CUDA functions
#include "matrix.h"

namespace gm {

/// An implementation of quaternion. This class can be used to represent
/// rotations or orientations.
template <typename T>
class Quaternion {
 public:
  Quaternion() : x(1), y(0), z(0), w(0) {}
  Quaternion(T x, T y, T z, T w) : x(x), y(y), z(z), w(w) {}

  Matrix4x4<T> toMat4() const {
    Matrix4x4<T> result;
    T qxx(x * x);
    T qyy(y * y);
    T qzz(z * z);
    T qxz(x * z);
    T qxy(x * y);
    T qyz(y * z);
    T qwx(w * x);
    T qwy(w * y);
    T qwz(w * z);

    result[0][0] = T(1) - T(2) * (qyy + qzz);
    result[1][0] = T(2) * (qxy + qwz);
    result[2][0] = T(2) * (qxz - qwy);

    result[0][1] = T(2) * (qxy - qwz);
    result[1][1] = T(1) - T(2) * (qxx + qzz);
    result[2][1] = T(2) * (qyz + qwx);

    result[0][2] = T(2) * (qxz + qwy);
    result[1][2] = T(2) * (qyz - qwx);
    result[2][2] = T(1) - T(2) * (qxx + qyy);
    return result;
  }

  T x;
  T y;
  T z;
  T w;
};

// Convenience typedefs.
typedef Quaternion<float> Quaternionf;
}  // namespace gm