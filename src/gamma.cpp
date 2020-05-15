#include <iostream>
#include <stdexcept>

#include "application/application.h"

int main(int argc, char** argv) {
  try {
    if (argc > 1) {
      gm::Application application(argv[1]);
      application.run();
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
