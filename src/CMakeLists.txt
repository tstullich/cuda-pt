# It's only necessary to list the source files here
set(SOURCE_FILES
    base/camera.cu
    base/image.cu
    base/image.cpp
    base/ray.cu
    integrator/integrator.cu
)

# Find dependencies
find_package(OpenImageIO REQUIRED)

link_directories(${OPENIMAGEIO_LIBRARIES})
include_directories(${OPENIMAGEIO_INCLUDE_DIRS})


# Static library configurations
add_library(gamma_lib STATIC ${SOURCE_FILES})
target_link_libraries(gamma_lib OpenImageIO::OpenImageIO)

# Target properties can be added here
set_target_properties(gamma_lib PROPERTIES CXX_STANDARD 14)
set_target_properties(gamma_lib PROPERTIES CXX_STANDARD_REQUIRED ON)
set_target_properties(gamma_lib PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Include directories. Add one for each new folder in this directory
target_include_directories(gamma_lib PUBLIC ${CMAKE_CURRENT_LIST_DIR}/base)
target_include_directories(gamma_lib PUBLIC ${CMAKE_CURRENT_LIST_DIR}/integrator)
target_include_directories(gamma_lib PUBLIC ${GLM_INCLUDE_DIR}/glm)

# Executable configurations
add_executable(gamma main.cpp)
#target_link_libraries(gamma gamma_lib OpenImageIO::OpenImageIO)
target_link_libraries(gamma gamma_lib)

install(TARGETS gamma DESTINATION ${GAMMA_INSTALL_BIN_DIR})
install(TARGETS gamma_lib DESTINATION ${GAMMA_INSTALL_BIN_DIR})