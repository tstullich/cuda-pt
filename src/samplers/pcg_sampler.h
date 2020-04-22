#pragma once

#include <memory>
#include <random>
#include <vector>

#include "sampler.h"
#include "vector.h"

namespace gm {

struct pcg32_state {
  uint64_t state;  // RNG state for the sampler
  uint64_t inc;    // Controls the selected RNG sequence (stream). Must be odd!
};

/// This class implements a sampler based on Permutable Congruential Generators.
/// For for more information visit: https://www.pcg-random.org/
/// This sampler should only be used initially for testing other features and
/// will eventually be replaced by better low-discrepancy sequence generators.
/// The generator is seeded using the pixel coordinate being sampled.
class PCGSampler : public Sampler {
 public:
  PCGSampler(const Vector2i &pixel, uint32_t samplesPerPixel);

  float get1D() override;

  Vector2f get2D() override;

 private:
  uint32_t next_pcg32(pcg32_state *rng);

  std::vector<pcg32_state> rng_states;  // One state per sample
};
}  // namespace gm