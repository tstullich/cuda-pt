#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include <ostream>

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
    T x2 = x + x;
    T y2 = y + y;
    T z2 = z + z;

    T xx2 = x2 * x;
    T xy2 = x2 * y;
    T xz2 = x2 * z;

    T yy2 = y2 * y;
    T yz2 = y2 * z;
    T zz2 = z2 * z;

    T sy2 = y2 * w;
    T sz2 = z2 * w;
    T sx2 = x2 * w;

    result[0][0] = 1.0f - yy2 - zz2;
    result[0][1] = xy2 + sz2;
    result[0][2] = xz2 - sy2;
    result[0][3] = 0.0f;

    result[1][0] = xy2 - sz2;
    result[1][1] = 1.0f - xx2 - zz2;
    result[1][2] = yz2 + sx2;
    result[1][3] = 0.0f;

    result[2][0] = xz2 + sy2;
    result[2][1] = yz2 - sx2;
    result[2][2] = 1.0f - xx2 - yy2;
    result[2][3] = 0.0f;

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