#include "integrator.h"

gm::Integrator::Integrator() { image = std::make_unique<RGBImage>(200, 100); }

void gm::Integrator::trace() {
  image->set_colors();
  image->write_png("test.png");
}