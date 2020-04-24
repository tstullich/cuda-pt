#include <iostream>
#include <stdexcept>

#include "integrator/integrator.h"

int main(int argc, char** argv) {
  try {
    if (argc > 1) {
      // Load our scene using the integrator. We should move this out
      // in the future to decouple Integrator & Scene classes
      gm::Integrator integrator(argv[1]);
      integrator.pathtrace();
    } else {
      std::cout
          << "No scene specified! Please supply a valid path to a glTF scene."
          << std::endl;
    }
  } catch (std::exception& e) {
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
