#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include <ostream>

#include "math.h"  // For built-in CUDA functions
#include "matrix.h"

namespace gm {

/// An implementation of quaternion. This class can be used to represent
/// rotations or orientations.
template <typename T>
class Quaternion {
 public:
  Quaternion() : x(0), y(0), z(0), w(1) {}
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

  Quaternion<T> operator*(const Quaternion<T> &q) {
    T t0 = w * q.x + x * q.w + y * q.z - z * q.y;
    T t1 = w * q.y + y * q.w + z * q.x - x * q.z;
    T t2 = w * q.z + z * q.w + x * q.y - y * q.x;
    T t3 = w * q.w - x * q.x - y * q.y - z * q.z;

    return Quaternion<T>(t0, t1, t2, t3);
  }

  Quaternion<T> conjugate() { return Quaternion<T>(-x, -y, -z, w); }

  Vector3<T> operator*(const Vector3<T> &v) {
    Quaternion<T> vq = Quaternion<T>(v.x, v.y, v.z, 0);
    Quaternion<T> r = *this * vq * this->conjugate();

    return Vector3<T>(r.x, r.y, r.z);
  }

  T x;
  T y;
  T z;
  T w;  // this is the scalar
};

template <typename T>
inline std::ostream &operator<<(std::ostream &stream, const Quaternion<T> &q) {
  stream << "Quaternion(" << q.x << ", " << q.y << ", " << q.z << ", " << q.w
         << ")";

  return stream;
}

// Convenience typedefs.
typedef Quaternion<float> Quaternionf;
}  // namespace gm