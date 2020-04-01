#pragma once

#include <cuda.h>

#include <iostream>
#include <memory>
#include <string>

namespace gm {
class RGBImage {
 public:
  RGBImage(size_t width, size_t height);

  inline size_t getChannels() const { return CHANNELS; }

  inline size_t getHeight() const { return height; }

  inline size_t getSize() const { return bufferSize; }

  inline size_t getWidth() const { return width; }

  inline uint8_t* getBuffer() const { return image.get(); }

  void writePNG(const std::string& fileName);

 private:
  const size_t CHANNELS = 3;
  size_t width;
  size_t height;
  size_t bufferSize;
  std::unique_ptr<uint8_t> image;
};
}  // namespace gm