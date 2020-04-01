#include "image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

gm::RGBImage::RGBImage(size_t width, size_t height)
    : width(width), height(height) {
  buffer_size = width * height * CHANNELS * sizeof(uint8_t);

  // Allocate CPU memory
  image = std::shared_ptr<uint8_t>(new uint8_t[buffer_size],
                                   std::default_delete<uint8_t[]>());
}

__global__ void color(uint8_t *image, size_t width, size_t channels) {
  size_t row = blockIdx.y * blockDim.y + threadIdx.y;
  size_t col = blockIdx.x * blockDim.x + threadIdx.x;
  size_t pixel_idx = (row * width + col) * channels;

  uint8_t r = 255;
  uint8_t g = 0;
  uint8_t b = 0;
  image[pixel_idx] = r;
  image[pixel_idx + 1] = g;
  image[pixel_idx + 2] = b;
}

void gm::RGBImage::set_colors() {
  // Allocate GPU memory
  auto gpu_deleter = [&](uint8_t *ptr) { cudaFree(ptr); };
  std::shared_ptr<uint8_t> gpu_image(new uint8_t[buffer_size], gpu_deleter);
  cudaMalloc((void **)&gpu_image, buffer_size);

  dim3 block_dimensions(8, 8);
  dim3 grid_dimensions((width / block_dimensions.x) + 1,
                       (height / block_dimensions.y) + 1);

  color<<<grid_dimensions, block_dimensions>>>(gpu_image.get(), width,
                                               CHANNELS);

  cudaDeviceSynchronize();

  cudaMemcpy(image.get(), gpu_image.get(), buffer_size, cudaMemcpyDeviceToHost);
}

void gm::RGBImage::write_png(const std::string &file_name) {
  stbi_write_png(file_name.data(), width, height, CHANNELS, image.get(), 0);
}