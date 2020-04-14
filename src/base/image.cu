#include "image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

gm::RGBImage::RGBImage(size_t width, size_t height)
    : width(width), height(height) {
  bufferSize = width * height * CHANNELS * sizeof(uint8_t);

  // Allocate CPU memory
  image = std::unique_ptr<uint8_t>(new uint8_t[bufferSize]());
}

void gm::RGBImage::writePNG(const std::string &fileName) {
  stbi_write_png(fileName.data(), width, height, CHANNELS, image.get(), 0);
}