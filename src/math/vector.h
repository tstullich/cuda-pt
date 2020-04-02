#pragma once

/// Including math.h for CUDA
#include "math.h"

/// There are two common vector classes contained within this file.
/// Vector2 and Vector3 have been templated to allow for flexible
/// usage with various data types. Common Vector operators have been
/// implemented as well, but feel free to expand these operations as
/// needed. Vec2f, Vec2i, Vec3f, and Vec3i are types that have been
/// created for convenience purposes and readability. Use them whenever
/// possible.

namespace gm {
template <typename T>
class Vector2 {
 public:
  __device__ Vector2() : x(0), y(0){};

  __device__ Vector2(T xx, T yy) : x(xx), y(yy) {
    if (hasNans()) {
      // TODO Find a way to handle errors inside device code
    }
  };

  __device__ Vector2<T> operator+(const Vector2<T> &v) const {
    if (v.hasNans()) {
      // TODO Find a way to handle errors inside device code
    }
    return Vector2(x + v.x, y + v.y);
  }

  __device__ Vector2<T> &operator+=(const Vector2<T> &v) const {
    if (v.hasNans()) {
      // TODO Find a way to handle errors inside device code
    }
    x += v.x;
    y += v.y;
    return *this;
  }

  __device__ Vector2<T> operator-(const Vector2<T> &v) const {
    if (v.hasNans()) {
      // TODO Find a way to handle errors inside device code
    }
    return Vector2(x - v.x, y - v.y);
  }

  __device__ Vector2<T> &operator-=(const Vector2<T> &v) const {
    if (v.hasNans()) {
      // TODO Find a way to handle errors inside device code
    }
    x -= v.x;
    y -= v.y;
    return *this;
  }

  __device__ bool operator==(const Vector2<T> &v) const {
    return x == v.x && y == v.y;
  }

  __device__ bool operator!=(const Vector2<T> &v) const {
    return x != v.x || y != v.y;
  }

  template <typename U>
  __device__ Vector2<T> operator*(U s) const {
    if (isnan(s)) {
      // TODO Find a way to handle errors inside device code
    }
    return Vector2<T>(s * x, s * y);
  }

  template <typename U>
  __device__ Vector2<T> &operator*=(U s) {
    if (isnan(s)) {
      // TODO Find a way to handle errors inside device code
    }
    x *= s;
    y *= s;
    return *this;
  }

  template <typename U>
  __device__ Vector2<T> operator/(U s) const {
    // We should not be able to divide by 0
    if (s == 0) {
      // TODO Find a way to handle errors inside device code
    }
    float inv = static_cast<float>(1 / s);
    return Vector2<T>(x * inv, y * inv);
  }

  template <typename U>
  __device__ Vector2<T> &operator/=(U s) {
    // We should not be able to divide by 0
    if (s == 0) {
      // TODO Find a way to handle errors inside device code
    }
    float inv = static_cast<float>(1 / s);
    x *= inv;
    y *= inv;
    return *this;
  }

  __device__ Vector2<T> operator-() const { return Vector2<T>(-x, -y); }

  __device__ T operator[](int i) const {
    if (i < 0 || i > 1) {
      // Out of bounds access is not allowed
      // TODO Find a way to handle errors inside device code
    }
    return i == 0 ? x : y;
  }

  __device__ float lengthSquared() const { return x * x + y * y; }

  __device__ float length() const { return sqrtf(lengthSquared()); }

  __device__ bool hasNans() const { return isnan(x) || isnan(y); }

  // X and Y components are freely accessible
  T x;
  T y;
};

template <typename T>
class Vector3 {
 public:
  __device__ Vector3() : x(0), y(0), z(0){};

  __device__ Vector3(T xx, T yy, T zz) : x(xx), y(yy), z(zz) {
    if (hasNans()) {
      // TODO Find a way to handle errors inside device code
    }
  };

  __device__ Vector3<T> operator+(const Vector3<T> &v) const {
    if (v.hasNans()) {
      // TODO Find a way to handle errors inside of device code
    }
    return Vector3(x + v.x, y + v.y, z + v.z);
  }

  __device__ Vector3<T> &operator+=(const Vector3<T> &v) {
    if (v.hasNans()) {
      // TODO Find a way to handle errors inside of device code
    }
    x += v.x;
    y += v.y;
    z += v.z;
    return *this;
  }

  __device__ Vector3<T> operator-(const Vector3<T> &v) const {
    if (v.hasNans()) {
      // TODO Find a way to handle errors inside of device code
    }
    return Vector3(x - v.x, y - v.y, z - v.z);
  }

  __device__ Vector3<T> &operator-=(const Vector3<T> &v) {
    if (v.hasNans()) {
      // TODO Find a way to handle errors inside of device code
    }
    x -= v.x;
    y -= v.y;
    z -= v.z;
    return *this;
  }

  __device__ bool operator==(const Vector3<T> &v) const {
    return x == v.x && y == v.y && z == v.z;
  }

  __device__ bool operator!=(const Vector3<T> &v) const {
    return x != v.x || y != v.y || z != v.z;
  }

  template <typename U>
  __device__ Vector3<T> operator*(U s) const {
    if (isnan(s)) {
      // Cannot multiply by Nan
      // TODO Find a way to handle errors inside device code
    }
    return Vector3<T>(s * x, s * y, s * z);
  }

  template <typename U>
  __device__ Vector3<T> &operator*=(U s) {
    if (hasNans(s)) {
      // Cannot multiply by Nan
      // TODO Find a way to handle errors inside device code
    }
    x *= s;
    y *= s;
    z *= s;
    return *this;
  }

  template <typename U>
  __device__ Vector3<T> operator/(U s) const {
    if (s == 0) {
      // Cannot divide by 0
      // TODO Find a way to handle errors inside device code
    }
    float inv = static_cast<float>(1 / s);
    return Vector3<T>(x * inv, y * inv, z * inv);
  }

  template <typename U>
  __device__ Vector3<T> &operator/=(U s) {
    if (s == 0) {
      // Cannot divide by 0
      // TODO Find a way to handle errors inside device code
    }
    x /= s;
    y /= s;
    z /= s;
    return *this;
  }

  __device__ Vector3<T> operator-() const { return Vector3<T>(-x, -y, -z); }

  __device__ T operator[](int i) const {
    if (i < 0 || i > 2) {
      // Out of bounds access is not allowed
      // TODO Find a way to handle errors inside device code
    }
    if (i == 0) {
      return x;
    }
    if (i == 1) {
      return y;
    }
    return z;
  }

  __device__ float lengthSquared() const { return x * x + y * y + z * z; }

  __device__ float length() const { return sqrtf(lengthSquared()); }

  __device__ bool hasNans() const { return isnan(x) || isnan(y) || isnan(z); }

  // X, Y, and Z components are freely accessible
  T x;
  T y;
  T z;
};

// Convenience typedefs. These should be used whenever possible
typedef Vector2<float> Vec2f;
typedef Vector2<int> Vec2i;
typedef Vector3<float> Vec3f;
typedef Vector3<int> Vec3i;

}  // namespace gm