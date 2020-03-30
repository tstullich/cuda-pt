set(SOURCE_FILES 
    base/image.cu
    main.cpp
)

add_executable(gamma ${SOURCE_FILES})

set_target_properties(gamma PROPERTIES CXX_STANDARD 17)
set_target_properties(gamma PROPERTIES CXX_STANDARD_REQUIRED ON)
set_target_properties(gamma PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

install(TARGETS gamma DESTINATION ${GAMMA_INSTALL_BIN_DIR})