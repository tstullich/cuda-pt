#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include <cmath>

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
  Vector2() : x(0), y(0){};

  Vector2(T x, T y) : x(x), y(y) {
    if (hasNans()) {
      // TODO Find a way to handle errors inside device code
    }
  };

  Vector2<T> operator+(const Vector2<T> &v) const {
    if (v.hasNans()) {
      // TODO Find a way to handle errors inside device code
    }
    return Vector2(x + v.x, y + v.y);
  }

  Vector2<T> &operator+=(const Vector2<T> &v) const {
    if (v.hasNans()) {
      // TODO Find a way to handle errors inside device code
    }
    x += v.x;
    y += v.y;
    return *this;
  }

  Vector2<T> operator-(const Vector2<T> &v) const {
    if (v.hasNans()) {
      // TODO Find a way to handle errors inside device code
    }
    return Vector2(x - v.x, y - v.y);
  }

  Vector2<T> &operator-=(const Vector2<T> &v) const {
    if (v.hasNans()) {
      // TODO Find a way to handle errors inside device code
    }
    x -= v.x;
    y -= v.y;
    return *this;
  }

  bool operator==(const Vector2<T> &v) const { return x == v.x && y == v.y; }

  bool operator!=(const Vector2<T> &v) const { return x != v.x || y != v.y; }

  template <typename U>
  Vector2<T> operator*(U s) const {
    if (isnan(s)) {
      // TODO Find a way to handle errors inside device code
    }
    return Vector2<T>(s * x, s * y);
  }

  template <typename U>
  Vector2<T> &operator*=(U s) {
    if (isnan(s)) {
      // TODO Find a way to handle errors inside device code
    }
    x *= s;
    y *= s;
    return *this;
  }

  template <typename U>
  Vector2<T> operator/(U s) const {
    // We should not be able to divide by 0
    if (s == 0) {
      // TODO Find a way to handle errors inside device code
    }
    float inv = static_cast<float>(1 / s);
    return Vector2<T>(x * inv, y * inv);
  }

  template <typename U>
  Vector2<T> &operator/=(U s) {
    // We should not be able to divide by 0
    if (s == 0) {
      // TODO Find a way to handle errors inside device code
    }
    float inv = static_cast<float>(1 / s);
    x *= inv;
    y *= inv;
    return *this;
  }

  Vector2<T> operator-() const { return Vector2<T>(-x, -y); }

  T operator[](int i) const {
    if (i < 0 || i > 1) {
      // Out of bounds access is not allowed
      // TODO Find a way to handle errors inside device code
    }
    return i == 0 ? x : y;
  }

  float lengthSquared() const { return x * x + y * y; }

  float length() const { return sqrtf(lengthSquared()); }

  bool hasNans() const { return isnan(x) || isnan(y); }

  // X and Y components are freely accessible
  T x;
  T y;
};

template <typename T>
class Vector3 {
 public:
  Vector3() : x(0), y(0), z(0){};

  Vector3(T x) : x(x), y(x), z(x){};

  Vector3(T x, T y, T z) : x(x), y(y), z(z) {
    if (hasNans()) {
      // TODO Find a way to handle errors inside device code
    }
  };

  Vector3<T> operator+(const Vector3<T> &v) const {
    if (v.hasNans()) {
      // TODO Find a way to handle errors inside of device code
    }
    return Vector3(x + v.x, y + v.y, z + v.z);
  }

  Vector3<T> &operator+=(const Vector3<T> &v) {
    if (v.hasNans()) {
      // TODO Find a way to handle errors inside of device code
    }
    x += v.x;
    y += v.y;
    z += v.z;
    return *this;
  }

  Vector3<T> operator-(const Vector3<T> &v) const {
    if (v.hasNans()) {
      // TODO Find a way to handle errors inside of device code
    }
    return Vector3(x - v.x, y - v.y, z - v.z);
  }

  Vector3<T> &operator-=(const Vector3<T> &v) {
    if (v.hasNans()) {
      // TODO Find a way to handle errors inside of device code
    }
    x -= v.x;
    y -= v.y;
    z -= v.z;
    return *this;
  }

  bool operator==(const Vector3<T> &v) const {
    return x == v.x && y == v.y && z == v.z;
  }

  bool operator!=(const Vector3<T> &v) const {
    return x != v.x || y != v.y || z != v.z;
  }

  Vector3<T> operator*(Vector3<T> v) const {
    if (v.hasNans()) {
      // Cannot multiply by Nan
      // TODO Find a way to handle errors inside device code
    }
    return Vector3<T>(v.x * x, v.y * y, v.z * z);
  }

  template <typename U>
  Vector3<T> operator*(U s) const {
    if (isnan(s)) {
      // Cannot multiply by Nan
      // TODO Find a way to handle errors inside device code
    }
    return Vector3<T>(s * x, s * y, s * z);
  }

  template <typename U>
  Vector3<T> &operator*=(U s) {
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
  Vector3<T> operator/(U s) const {
    if (s == 0) {
      // Cannot divide by 0
      // TODO Find a way to handle errors inside device code
    }
    float inv = static_cast<float>(1 / s);
    return Vector3<T>(x * inv, y * inv, z * inv);
  }

  template <typename U>
  Vector3<T> &operator/=(U s) {
    if (s == 0) {
      // Cannot divide by 0
      // TODO Find a way to handle errors inside device code
    }
    x /= s;
    y /= s;
    z /= s;
    return *this;
  }

  Vector3<T> operator-() const { return Vector3<T>(-x, -y, -z); }

  T operator[](int i) const {
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
    if (i == 2) {
      return z;
    }
    return -1;
  }

  bool hasNans() const { return isnan(x) || isnan(y) || isnan(z); }

  float lengthSquared() const { return x * x + y * y + z * z; }

  float length() const { return sqrtf(lengthSquared()); }

  // X, Y, and Z components are freely accessible
  T x;
  T y;
  T z;
};

/// These are vector transformation functions, that are useful for various
/// purposes. One assumption is made that the vectors passed into these
/// functions operate using 32-bit floating point values, so that it is
/// possible to use the built-in CUDA functions contained in <math.h>
template <typename T>
float dot(const Vector3<T> &v1, const Vector3<T> &v2) {
  return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
}

template <typename T>
Vector3<T> cross(const Vector3<T> &v1, const Vector3<T> &v2) {
  // Convert vector entries to double to prevent catastrophic cancellation
  double v1x = v1.x, v1y = v1.y, v1z = v1.z;
  double v2x = v2.x, v2y = v2.y, v2z = v2.z;
  return Vector3<T>((v1y * v2z) - (v1z * v2y), (v1z * v2x) - (v1x * v2z),
                    (v1x * v2y) - (v1y * v2x));
}

template <typename T>
Vector3<T> max(const Vector3<T> &v1, const Vector3<T> &v2) {
  return Vector3<T>(fmaxf(v1.x, v2.x), fmaxf(v1.y, v2.y), fmaxf(v1.z, v2.z));
}

template <typename T>
Vector3<T> min(const Vector3<T> &v1, const Vector3<T> &v2) {
  return Vector3<T>(fminf(v1.x, v2.x), fminf(v1.y, v2.y), fminf(v1.z, v2.z));
}

template <typename T>
T maxComponent(const Vector3<T> &v) {
  return fmaxf(v.x, fmaxf(v.y, v.z));
}

template <typename T>
T minComponent(const Vector3<T> &v) {
  return fminf(v.x, fminf(v.y, v.z));
}

template <typename T>
Vector3<T> normalize(const Vector3<T> &v) {
  return v / v.length();
}

template <typename T>
inline std::ostream &operator<<(std::ostream &stream, const Vector3<T> &v) {
  stream << "Vector3(" << v.x << ", " << v.y << ", " << v.z << ")";

  return stream;
}

// Convenience typedefs. These should be used whenever possible
typedef Vector2<float> Vector2f;
typedef Vector2<int> Vector2i;
typedef Vector3<float> Vector3f;
typedef Vector3<int> Vector3i;
}  // namespace gm