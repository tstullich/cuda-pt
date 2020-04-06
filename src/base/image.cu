#include "image.h"

//#define STB_IMAGE_WRITE_IMPLEMENTATION
//#include "stb_image_write.h"



gm::RGBImage::RGBImage(size_t width, size_t height)
    : width(width), height(height) {
  bufferSize = width * height * CHANNELS * sizeof(uint8_t);

  // Allocate CPU memory
  image = std::unique_ptr<uint8_t>(new uint8_t[bufferSize]());
}

