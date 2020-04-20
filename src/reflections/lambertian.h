#pragma once

#include "bxdf.h"
#include "vector.h"

namespace gm {

/// This class implements a BRDF for lambertian diffuse reflection.
/// There is also a use case for diffuse transmission which is going to be
/// implemented in another class. This diffusion model assumes that the surface
/// is perfectly smooth, so we do not need to implement sample_f(), as the
/// pdf is constnat
class LambertianReflection : public BxDF {
 public:
  LambertianReflection(const Vector3f &r)
      : BxDF(BSDF_REFLECTION | BSDF_DIFFUSE)) {}

  Vector3f f(const Vector3f &wi, const Vector3f &wi) const override;
};
}  // namespace gm