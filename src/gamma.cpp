#include <iostream>
#include <memory>
#include <stdexcept>

#include "base/image.h"
#include "integrator/integrator.h"
#include "geometry/matrix.h"

int main(int argc, char** argv) {
  try {
    if (argc > 1) {
      // Load our scene using the integrator. We should move this out
      // in the future to decouple Integrator & Scene classes
      gm::Matrix4x4f m = gm::Matrix4x4f({12, 5, 6, -6 ,5 ,4 , 8, 4 ,- 6, 9 ,4 ,54 , 0, 0 , 0, 1});
      gm::Vector3f p = gm::Vector3f(68, -4,6);
      gm::Vector3f v = gm::Vector3f(-21, -5,2);
      std::cout << m << std::endl;
      
       std::cout << m.multiplyPoint(p) << std::endl;
       
       std::cout << m.multiplyVector(v) << std::endl;
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
