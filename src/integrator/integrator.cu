#include "integrator.h"

gm::Integrator::Integrator(const std::string &filePath, const RenderOptions &options) : options(options) {
  // Load our scene
  scene = std::unique_ptr<Scene>(new Scene(filePath, options));

  // Set some camera settings based on the output image
  scene->camera->setImagePlane(options.imageWidth, options.imageHeight);

  // Build BVH
  bvh = std::unique_ptr<BVHDummy>(new BVHDummy(scene));

  // Initialize the final image
  image = std::unique_ptr<RGBImage>(new RGBImage(options.imageWidth, options.imageHeight));
}

void gm::Integrator::pathtrace() {
  uint8_t *imageBuffer = image->getBuffer();
  size_t imageWidth = image->getWidth();
  size_t imageHeight = image->getHeight();
  uint32_t samplesPerPixel = options.samplesPerPixel;

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

        // Generate primary rays
        Ray ray = scene->camera->generateRay(xCoord, yCoord, cameraSample);

        // Find an intersection point between the rays and the scene
        std::shared_ptr<Intersection> intersection =
            std::make_shared<Intersection>();
        if (!bvh->intersect(ray, intersection)) {
          // For now if we do not make any intersections with the scene
          // simply skip the light contributions for this sample. Later
          // the rendering loop can exit early here
          sampler.startNextSample();
          continue;
        }

        std::cout << "(" << xCoord << ", " << yCoord
                  << ") hit: " << intersection->name
                  << " t: " << intersection->tHit
                  << " p: " << intersection->surfacePoint.x << ", " << intersection->surfacePoint.y << ", " << intersection->surfacePoint.z << std::endl;
        pixelColor += Vector3f(0.5f);// Set a grey color

        // Compute scattering ray based on material BxDFs
        // Sample light sources to find path contribution. Skip for specular
        // materials

        // Sample BSDF for new path direction

        // Apply Russian roulette for early termination

        // Advance the sampler state for the next sample
        sampler.startNextSample();
      }

      /// Apply basic anti-aliasing by averaging the samples per-pixel
      pixelColor /= samplesPerPixel;

      /// Store the output color
      size_t pixelIdx = (yCoord * imageHeight + xCoord) * image->getChannels();
      imageBuffer[pixelIdx] = static_cast<uint8_t>(pixelColor.x * 255.99);
      imageBuffer[pixelIdx + 1] = static_cast<uint8_t>(pixelColor.y * 255.99);
      imageBuffer[pixelIdx + 2] = static_cast<uint8_t>(pixelColor.z * 255.99);
    }
  }
  image->writePNG("test.png");
}
