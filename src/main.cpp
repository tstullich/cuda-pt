#include <iostream>

#include "base/image.h"

int main(int argc, char** argv) {
  gm::RGBImage img(200, 100);

  img.set_colors();

  img.write_png("test.png");

  return 0;
}