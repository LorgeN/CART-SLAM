# CART-SLAM: CUDA-Accelerated Real-Time SLAM

This is an implementation of SLAM using CUDA to accelerate processing. It is loosely based on ORB-SLAM3, but is not a direct port. It is written as part of my master's thesis at the Norwegian University of Science and Technology (NTNU).

## Building

Use CMake to build the project. The following libraries are required:
 - OpenCV
 - CUDA

Command:
```bash
cmake -S . -B build
```

Optionally you can use `-DCMAKE_CXX_COMPILER=<your compiler>` and `-DCMAKE_CUDA_COMPILER=<your CUDA compiler>` to specify which compilers to use.