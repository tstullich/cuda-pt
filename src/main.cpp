#include <iostream>
#include <stdexcept>

#include "base/image.h"
#include "base/scene.h"
#include "integrator/integrator.h"

int main(int argc, char** argv) {
  try {
    gm::Integrator integrator;
    // integrator.integrate();
    integrator.pathtrace();
    if (argc > 1) {
      gm::Scene scene(argv[1]);
    }
  } catch (std::exception& e) {
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}