#pragma once

#include "intersection.h"
#include "vector.h"

namespace gm {
/// State flags to indicate what kind of reflection the BSDF has to take
/// care of. This will enable us to add in more reflection types later.
enum BxDFType {
  BSDF_DIFFUSE = 1 << 0,
  BSDF_SPECULAR = 1 << 1,
  BSDF_REFLECTION = 1 << 2,
  BSDF_ALL = BSDF_DIFFUSE | BSDF_SPECULAR | BSDF_REFLECTION;
};

/// An abstract class that will represent BxDF functions which are going to be
/// used for shading and sampling new light directions from materials that
/// reflect light.
class BxDF {
 public:
  BxDF(const BxDFType &type) : type(type){};
  virtual Vector3f f(const Vector3f &wi, const Vector3f &wi) const = 0;

  virtual Vector3f sample_f(const Vector3f &wo,
                            const Intersection &surfacePoint,
                            const Vector2f &samples) const;

  virtual Vector3f sample_pdf();

  virtual float pdf(const Vector3f &wi, const Vector3f &wo) const;

  BxDFType type;
};
}  // namespace gm