# Introduction
Welcome to the source code page of gamma, the hybrid path tracer. For build instructions see the section below
and for more general information visit the wiki.

_Gamma is still a work in progress and the code is currently not ready for a release. Please be aware of this fact when checking out the code._

# Build Instructions
Make sure to have the following dependencies installed somewhere on your system:

1. CUDA toolkit (10.2)
2. CMake (3.13.4)
3. A C/C++ compiler toolchain (gcc-7/clang-7)

then clone the repository using this command:

`git clone --recursive git@github.com:tstullich/gamma.git`

Once inside the top-level of the repository create a build directory:
```
mkdir build
cd build
cmake .. && make install
```

This should install the `gamma` binary inside the bin directory in the project root. For more details
and future improvements come back later :)
