#include <iostream>
#include <stdexcept>

#include "base/image.h"
#include "integrator/integrator.h"

int main(int argc, char** argv) {
  try {
    gm::Integrator integrator;
    integrator.integrate();
  } catch (std::exception& e) {
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}