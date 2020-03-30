#pragma once

#include <cuda.h>

#include <memory>
#include <string>

namespace gm {
class RGBImage {
 public:
  RGBImage(size_t width, size_t height);

  ~RGBImage();

  void set_colors();

  void write_png(const std::string& file_name);

 private:
  const size_t CHANNELS = 3;
  size_t width;
  size_t height;
  size_t buffer_size;
  uint8_t* image;
};
}  // namespace gm