#include "integrator.h"

gm::Integrator::Integrator() {
  image = std::unique_ptr<RGBImage>(new RGBImage(IMAGE_WIDTH, IMAGE_HEIGHT));
  camera = std::shared_ptr<PerspectiveCamera>(new PerspectiveCamera(
      Vector3f(0.0f, 0.0f, 1.0f), Vector3f(0.0f, 0.0f, 0.0f),
      Vector3f(0.0f, 1.0f, 0.0f), IMAGE_WIDTH, IMAGE_HEIGHT, 70.0f));
}

// The entry point for the path tracing kernel. This should be called from
// the integrate function only
__global__ void pathtraceGPU(uint8_t *image, size_t width, size_t channels) {
  size_t row = blockIdx.y * blockDim.y + threadIdx.y;
  size_t col = blockIdx.x * blockDim.x + threadIdx.x;
  size_t pixelIdx = (row * width + col) * channels;

  image[pixelIdx] = 255;
  image[pixelIdx + 1] = 0;
  image[pixelIdx + 2] = 0;
}

/// Only a test function. Should be replaced later on
void gm::Integrator::pathtrace() {
  uint8_t *imageBuffer = image->getBuffer();
  for (uint32_t row = 0; row < image->getHeight(); ++row) {
    for (uint32_t col = 0; col < image->getWidth(); ++col) {
      Ray r = camera->generate_ray(row, col);
      Vector3f hitColor =
          ((r.direction + Vector3f(1.0f, 1.0f, 1.0f)) * 0.5) * 255.99;

      size_t pixelIdx = (row * image->getWidth() + col) * image->getChannels();
      imageBuffer[pixelIdx] = static_cast<uint8_t>(hitColor.x);
      imageBuffer[pixelIdx + 1] = static_cast<uint8_t>(hitColor.y);
      imageBuffer[pixelIdx + 2] = static_cast<uint8_t>(hitColor.z);
    }
  }
  image->writePNG("test.png");
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
  pathtraceGPU<<<gridDimensions, blockDimensions>>>(
      gpuImage.get(), image->getWidth(), image->getChannels());

  // Sync all of the threads before continuing
  cudaDeviceSynchronize();

  // Copy result into CPU/host memory to write to a file
  cudaMemcpy(image->getBuffer(), gpuImage.get(), bufferSize,
             cudaMemcpyDeviceToHost);

  // Write image to disk
  image->writePNG("test.png");
}