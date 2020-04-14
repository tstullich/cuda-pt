#include "pcg_sampler.h"

gm::PCGSampler::PCGSampler(const Vector2i &pixel, uint32_t samplesPerPixel)
    : Sampler(samplesPerPixel) {
  rng_states = std::vector<pcg32_state>(samplesPerPixel);
  uint64_t seed = pixel.x + pixel.y;

  // Initialize each sample-per-pixel state
  for (uint32_t idx = 0; idx < samplesPerPixel; ++idx) {
    rng_states[idx].state = 0U;
    rng_states[idx].inc = (((uint64_t)idx + 1) << 1u) | 1u;
    next_pcg32(&rng_states[idx]);
    rng_states[idx].state += (0x853c49e6748fea9bULL + seed);
    next_pcg32(&rng_states[idx]);
  }
}

std::unique_ptr<gm::Sampler> gm::PCGSampler::clone(int seed) {
  // TODO Implement
  return nullptr;
}

// Generate a single precision floating point value on the interval [0, 1)
// Trick from MTGP: generate an uniformly distributed single precision number
// in [1,2) and subtract 1.
float gm::PCGSampler::get1D() {
  union {
    uint32_t u;
    float f;
  } x;
  x.u = (next_pcg32(&rng_states[currentPixelSampleIndex]) >> 9) | 0x3f800000u;
  return x.f - 1.0f;
}

gm::Vector2f gm::PCGSampler::get2D() { return Vector2f(get1D(), get1D()); }

// http://www.pcg-random.org/download.html
uint32_t gm::PCGSampler::next_pcg32(pcg32_state *rng) {
  uint64_t oldstate = rng->state;
  // Advance internal state
  rng->state = oldstate * 6364136223846793005ULL + (rng->inc | 1);
  // Calculate output function (XSH RR), uses old state for max ILP
  uint32_t xorshifted = uint32_t(((oldstate >> 18u) ^ oldstate) >> 27u);
  uint32_t rot = uint32_t(oldstate >> 59u);
  return uint32_t((xorshifted >> rot) | (xorshifted << ((-rot) & 31)));
}