#include "integrator.h"

gm::Integrator::Integrator() {
  image = std::unique_ptr<RGBImage>(new RGBImage(IMAGE_WIDTH, IMAGE_HEIGHT));
  // Place the camera at (0, 0, 1) looking in the -z direction
  camera = std::shared_ptr<PerspectiveCamera>(new PerspectiveCamera(
      Vector3f(0.0f, 0.0f, 1.0f), Vector3f(0.0f, 0.0f, 0.0f),
      Vector3f(0.0f, 1.0f, 0.0f), IMAGE_WIDTH, IMAGE_HEIGHT, 70.0f));
}

/// Only a test function. Should be replaced later on
void gm::Integrator::pathtrace() {
  uint8_t *imageBuffer = image->getBuffer();
  size_t imageWidth = image->getWidth();
  size_t imageHeight = image->getHeight();
  uint32_t samplesPerPixel = 4; // Can be configured later

  // Add a triangle going centered around (0, 0, -1)
  // Triangle triangle(Vector3f(-0.5f, -0.5f, 0.0f), Vector3f(0.5f, -0.5f,
  // 0.0f),
  //                  Vector3f(0.0f, 0.5f, 0.0f));

  Sphere sphere(Vector3f(0.0f, 0.0f, -1.0f), 1.0f);

  for (uint32_t yCoord = 0; yCoord < imageHeight; ++yCoord) {
    for (uint32_t xCoord = 0; xCoord < imageWidth; ++xCoord) {

      // Create a sampler here for now. The rendering loop will need to be
      // rewritten anyways
      PCGSampler sampler(Vector2i(xCoord, yCoord), samplesPerPixel);
      // Initialize the pixel color
      Vector3f pixelColor(0.0f);
      for (uint32_t sample = 0; sample < samplesPerPixel; ++sample) {
        // Get a 2D sample for the camera rays
        Vector2f cameraSample = sampler.get2D();

        if (xCoord == imageWidth / 2 && yCoord == imageHeight / 2) {
          std::cout << "Sample: " << cameraSample.x << ", " << cameraSample.y
                    << std::endl;
        }

        // Sample the primary rays
        Ray ray = camera->generate_ray(xCoord, yCoord, cameraSample);

        // Calculate an intersection with the scene
        std::unique_ptr<Intersection> intersection =
            std::make_unique<Intersection>();
        if (sphere.intersect(ray, intersection)) {
          // If we intersect the triangle set the hit color to red
          auto N = sphere.normal(intersection->surfacePoint);
          pixelColor +=
              (N + 1.0f) * 0.5f; // Adjust the normal vector before shading
        }

        // Advance the sampler state for the next sample
        sampler.startNextSample();
      }

      /// Apply basic anti-aliasing by averaging the samples per-pixel
      pixelColor /= samplesPerPixel;

      /// Store the output color
      size_t pixelIdx = (yCoord * imageWidth + xCoord) * image->getChannels();
      imageBuffer[pixelIdx] = static_cast<uint8_t>(pixelColor.x * 255.99);
      imageBuffer[pixelIdx + 1] = static_cast<uint8_t>(pixelColor.y * 255.99);
      imageBuffer[pixelIdx + 2] = static_cast<uint8_t>(pixelColor.z * 255.99);
    }
  }
  image->writePNG("test.png");
}

// The code below will be used at a later time

// void gm::Integrator::integrate() {
//  // Allocate GPU memory for the image
//  const size_t bufferSize = image->getSize();
//
//  // Create a custom deleter since GPU memory needs to be freed after use
//  auto gpuDeleter = [&](uint8_t *ptr) { cudaFree(ptr); };
//  std::shared_ptr<uint8_t> gpuImage(new uint8_t[bufferSize], gpuDeleter);
//  cudaMalloc((void **)&gpuImage, bufferSize);
//
//  // Determine the grid and block dimensions. We need to allocate a grid of
//  // blocks containing a thread per pixel. Initially the blocks will be 8x8 =
//  64
//  // threads large
//  dim3 blockDimensions(BLOCK_SIZE, BLOCK_SIZE);
//  dim3 gridDimensions((image->getWidth() / blockDimensions.x) + 1,
//                      (image->getHeight() / blockDimensions.y) + 1);
//
//  // Launch the path tracing kernel. This is the main entry point for
//  // gamma's logic
//  pathtraceGPU<<<gridDimensions, blockDimensions>>>(
//      gpuImage.get(), image->getWidth(), image->getChannels());
//
//  // Sync all of the threads before continuing
//  cudaDeviceSynchronize();
//
//  // Copy result into CPU/host memory to write to a file
//  cudaMemcpy(image->getBuffer(), gpuImage.get(), bufferSize,
//             cudaMemcpyDeviceToHost);
//
//  // Write image to disk
//  image->writePNG("test.png");
//}

// The entry point for the path tracing kernel. This should be called from
// the integrate function only
//__global__ void pathtraceGPU(uint8_t *image, size_t width, size_t channels) {
//  size_t row = blockIdx.y * blockDim.y + threadIdx.y;
//  size_t col = blockIdx.x * blockDim.x + threadIdx.x;
//  size_t pixelIdx = (row * width + col) * channels;
//
//  image[pixelIdx] = 255;
//  image[pixelIdx + 1] = 0;
//  image[pixelIdx + 2] = 0;
//}