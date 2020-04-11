#pragma once

#include <memory>

#include "vector.h"

namespace gm {

/// Interface class for creating different types of samplers.
/// These samplers should output random samples based on a
/// uniform distribution in the range of [0, 1)^n where n is
/// the number of dimensions. The samplers should create unique
/// sequences based on the pixel that is being sampled.

class Sampler {
 public:
  /// The Sampler is initialized with the number of samples per pixels.
  /// Because we anticipate having to generate a large amount of samples
  /// per pixel we store the spp inside a 64-bit integer
  /// TODO This should be user-defined in the future.
  Sampler(int64_t samplesPerPixel)
      : samplesPerPixel(samplesPerPixel), currentPixelSampleIndex(0){};

  /// We plan on using a single sampler per thread, so we need to be able
  /// to clone samplers for each worker, otherwise we can run into noise
  /// artifacts due to the fact that the same sequences are being reused.
  virtual std::unique_ptr<Sampler> clone(int seed) = 0;

  /// Generate a 1-dimensional sample given the current sample. The index
  /// should correspond to the sample-per-pixel
  virtual float get1D() = 0;

  /// Generate a 1-dimensional sample given the current sample. The index
  /// should correspond to the sample-per-pixel
  virtual Vector2f get2D() = 0;

  /// Convenience method to test camera sampling logic. In the beginning we only
  /// need a single 2D sample for the image plane coordinates, but this method
  /// can be expanded later on for various purposes (i.e. lens sampling for DoF)
  Vector2f getCameraSamples(const Vector2i &pixel) { return get2D(); }

  /// This increments the pixel index and also indicates the to the integrator
  /// that it should stop using random samples after a given iteration
  virtual bool startNextSample() {
    return ++currentPixelSampleIndex < samplesPerPixel;
  }

  int64_t samplesPerPixel;

 protected:
  int64_t currentPixelSampleIndex;
};

}  // namespace gm