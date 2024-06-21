# CART-SLAM: CUDA-Accelerated Real-Time SLAM

This is an implementation of SLAM using CUDA to accelerate processing. The current implementation focuses on identifying planes in the environment.

## Running

The program is run from the command line, and has the following syntax:

```bash
./cart-slam <path-to-config>
```
or
```bash
./cart-slam <path-to-source-config> <path-to-modules-config>
```

Some sample configurations are provided in the `config`-folder.

## Building

Use CMake to build the project. The following libraries are required:
 - OpenCV
 - CUDA

Command:
```bash
cmake -S . -B build
```

Optionally you can use `-DCMAKE_CXX_COMPILER=<your compiler>` and `-DCMAKE_CUDA_COMPILER=<your CUDA compiler>` to specify which compilers to use. A useful flag for debugging is `-DCMAKE_BUILD_TYPE=Debug`.

### CMAKE Options

Available options are

 - `ENBALE_TIMING` (default: `OFF`): Enable generation of timing `csv` files to `timings`-folder.
 - `ENABLE_SAVE_SAMPLES` (default: `OFF`): Enable saving of samples to disk in `samples`-folder.
 - `ENABLE_RECORD_SAMPLES` (default: `OFF`): Enable recording of samples to disk in `samples`-folder.

These can be set using `-D<option>=<value>` when running CMake.

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