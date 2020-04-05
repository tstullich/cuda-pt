#include "integrator.h"

gm::Integrator::Integrator() {
  image = std::unique_ptr<RGBImage>(new RGBImage(IMAGE_WIDTH, IMAGE_HEIGHT));
  FilmInfo info(0.825, 0.446, ScanMode::Fill); // Full 35mm aspect ratio
  camera = PerspectiveCamera(info, IMAGE_WIDTH, IMAGE_HEIGHT, 35.0f);
}

// The entry point for the path tracing kernel. This should be called from
// the integrate function only
__global__ void pathtrace(uint8_t *image, size_t width, size_t channels) {
  size_t row = blockIdx.y * blockDim.y + threadIdx.y;
  size_t col = blockIdx.x * blockDim.x + threadIdx.x;
  size_t pixelIdx = (row * width + col) * channels;

  image[pixelIdx] = 255;
  image[pixelIdx + 1] = 0;
  image[pixelIdx + 2] = 0;
}

void gm::Integrator::integrate() {
  // Allocate GPU memory for the image
  const size_t bufferSize = image->getSize();

  // Create a custom deleter since GPU memory needs to be freed after use
  auto gpuDeleter = [&](uint8_t *ptr) { cudaFree(ptr); };
  std::shared_ptr<uint8_t> gpuImage(new uint8_t[bufferSize], gpuDeleter);
  cudaMalloc((void **)&gpuImage, bufferSize);

  // Determine the grid and block dimensions. We need to allocate a grid of
  // blocks containing a thread per pixel. Initially the blocks will be 8x8 = 64
  // threads large
  dim3 blockDimensions(BLOCK_SIZE, BLOCK_SIZE);
  dim3 gridDimensions((image->getWidth() / blockDimensions.x) + 1,
                      (image->getHeight() / blockDimensions.y) + 1);

  // Launch the path tracing kernel. This is the main entry point for
  // gamma's logic
  pathtrace<<<gridDimensions, blockDimensions>>>(
      gpuImage.get(), image->getWidth(), image->getChannels());

  // Sync all of the threads before continuing
  cudaDeviceSynchronize();

  // Copy result into CPU/host memory to write to a file
  cudaMemcpy(image->getBuffer(), gpuImage.get(), bufferSize,
             cudaMemcpyDeviceToHost);

  // Write image to disk
  image->writePNG("test.png");
}