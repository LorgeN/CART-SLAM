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

Optionally you can use `-DCMAKE_CXX_COMPILER=<your compiler>` and `-DCMAKE_CUDA_COMPILER=<your CUDA compiler>` to specify which compilers to use. A useful flag for debugging is `-DCMAKE_BUILD_TYPE=Debug`.

### ZedSDK on Manjaro

The ZedSDK is intended for Ubuntu. To get it working on Manjaro, you need to modify the installed CMake file (found at `/usr/local/zed/zed-config.cmake`).

Change the line:
```cmake
set(LIB_PATH_64 "/usr/lib/${CMAKE_SYSTEM_PROCESSOR}-linux-gnu/")
```
to
```cmake
set(LIB_PATH_64 "/usr/lib/")
```

Optionally also add the following line to suppress the warning spam about policy CMP0153:
```cmake
cmake_policy(SET CMP0153 OLD)
```