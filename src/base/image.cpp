#include <image.h>


#ifndef __CUDA_ARCH__
  #include <OpenImageIO/imageio.h>
  using namespace OIIO;
#endif

void gm::RGBImage::writePNG(const std::string &fileName) {

  auto out = ImageOutput::create(fileName);
  if(!out){
      std::cerr << "could not write image"<< std::endl;
  }
  ImageSpec spec (width, height, CHANNELS, TypeDesc::UINT8);


  out->open(fileName, spec);
  out->write_image(TypeDesc::UINT8, this->image.get());
  out->close();

}