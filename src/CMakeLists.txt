set(HEADER_FILES
    base/image.h
)

set(SOURCE_FILES
    base/image.cu
    main.cpp
)

# Static library configurations
add_library(gamma_lib STATIC ${HEADER_FILES} ${SOURCE_FILES})

set_target_properties(gamma_lib PROPERTIES CXX_STANDARD 17)
set_target_properties(gamma_lib PROPERTIES CXX_STANDARD_REQUIRED ON)
set_target_properties(gamma_lib PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Include directories will be added here
target_include_directories(gamma_lib PRIVATE ${CMAKE_CURRENT_SRC_DIR}/base)

# Executable configurations
add_executable(gamma ${SOURCE_FILES})

set_target_properties(gamma PROPERTIES CXX_STANDARD 17)
set_target_properties(gamma PROPERTIES CXX_STANDARD_REQUIRED ON)
set_target_properties(gamma PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

install(TARGETS gamma DESTINATION ${GAMMA_INSTALL_BIN_DIR})