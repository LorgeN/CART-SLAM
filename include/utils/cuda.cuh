#pragma once

#include <log4cxx/logger.h>

#include "modules/disparity.hpp"
#include "utils/colors.hpp"

// OpenCV uses row-major order
#define INDEX(x, y, rowStep) ((y) * (rowStep) + (x))
#define INDEX_CH(x, y, chCount, ch, rowStep) ((y) * (rowStep) + (x) * (chCount) + (ch))
#define INDEX_BGR(x, y, ch, rowStep) INDEX_CH(x, y, 3, ch, rowStep)
#define CLAMP(x, a, b) (max((a), min((b), (x))))
#define SHARED_INDEX(x, y, xPadding, yPadding, rowStep) (((y) + (yPadding)) * ((rowStep) + 2 * (xPadding)) + ((x) + (xPadding)))

#define CUDA_SAFE_CALL(logger, ans) \
    { cart::gpuAssert((logger), (ans), __FILE__, __LINE__); }

// From https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#atomic-functions
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 600
__device__ double atomicAdd(double *address, double val);
#endif

__device__ inline double atomicSub(double *address, double val) {
    return atomicAdd(address, -val);
}

namespace cart {

/**
 * @brief Struct for passing arrays of cv::cuda::GpuMats to kernels
 *
 * @tparam T Type of the data in the cv::cuda::GpuMat
 */
template <typename T>
struct __align__(16) cv_mat_ptr_t {
    T *data;
    size_t step;
};

void copyColorWheelToDevice(cudaStream_t &stream);

__device__ void assignColor(float fx, float fy, uint8_t *pix);

template <typename T, int XBatch, int YBatch, bool Interpolate = true>
__device__ void copyToShared(T *shared, cv::cuda::PtrStepSz<T> values, int yPadding, int xPadding) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    const int sharedPixelX = threadIdx.x * XBatch;
    const int sharedPixelY = threadIdx.y * YBatch;

    const int pixelX = x * XBatch;
    const int pixelY = y * YBatch;

    const int width = values.cols;
    const int height = values.rows;

    const size_t inputRowStep = values.step / sizeof(T);
    const size_t sharedRowStep = XBatch * blockDim.x;

    for (int i = 0; i < YBatch; i++) {
        for (int j = 0; j < XBatch; j++) {
            if (pixelX + j >= width || pixelY + i >= height) {
                continue;
            }

            shared[SHARED_INDEX(sharedPixelX + j, sharedPixelY + i, xPadding, yPadding, sharedRowStep)] = values[INDEX(pixelX + j, pixelY + i, inputRowStep)];
        }
    }

    // TODO: Potentially make this expansion process its own kernel? It's a bit messy here
    if (yPadding > 0) {
        if (threadIdx.y == 0) {
            // Copy extra rows on top
            for (int i = 1; i <= yPadding; i++) {
                for (int j = 0; j < XBatch; j++) {
                    if (pixelX + j >= width) {
                        break;
                    }

                    T value;
                    if (pixelY - i >= 0) {
                        value = values[INDEX(pixelX + j, pixelY - i, inputRowStep)];
                    } else if (!Interpolate) {  // Should automatically be removed by the compiler, depending on the value
                        value = values[INDEX(pixelX + j, 0, inputRowStep)];
                    } else {
                        // Perform basic 1D interpolation. This does mean some branching, but most likely all threads will
                        // be doing the same thing so the divergence should be minimal
                        T borderValue = values[INDEX(pixelX + j, 0, inputRowStep)];
                        T nextValue = values[INDEX(pixelX + j, 1, inputRowStep)];
                        value = borderValue + (nextValue - borderValue) * i;
                    }

                    shared[SHARED_INDEX(sharedPixelX + j, sharedPixelY - i, xPadding, yPadding, sharedRowStep)] = value;
                }
            }
        }

        if (threadIdx.y == blockDim.y - 1) {
            int maxSharedY = YBatch * blockDim.y;

            // Copy extra rows on bottom
            for (int i = 0; i < yPadding; i++) {
                for (int j = 0; j < XBatch; j++) {
                    if (pixelX + j >= width) {
                        break;
                    }

                    T value;
                    if (pixelY + YBatch + i < height) {
                        value = values[INDEX(pixelX + j, pixelY + YBatch + i, inputRowStep)];
                    } else if (!Interpolate) {
                        value = values[INDEX(pixelX + j, height - 1, inputRowStep)];
                    } else {
                        // Perform basic 1D interpolation. This does mean some branching, but most likely all threads will
                        // be doing the same thing so the divergence should be minimal
                        T borderValue = values[INDEX(pixelX + j, height - 1, inputRowStep)];
                        T prevValue = values[INDEX(pixelX + j, height - 2, inputRowStep)];
                        value = borderValue + (borderValue - prevValue) * (i + 1);
                    }

                    shared[SHARED_INDEX(sharedPixelX + j, maxSharedY + i, xPadding, yPadding, sharedRowStep)] = value;
                }
            }
        }
    }

    if (xPadding > 0) {
        if (threadIdx.x == 0) {
            // Copy extra columns on left
            for (int i = 0; i < YBatch; i++) {
                if (pixelY + i >= height) {
                    break;
                }

                for (int j = 1; j <= xPadding; j++) {
                    T value;

                    if (pixelX - j >= 0) {
                        value = values[INDEX(pixelX - j, pixelY + i, inputRowStep)];
                    } else if (!Interpolate) {
                        value = values[INDEX(0, pixelY + i, inputRowStep)];
                    } else {
                        // Perform basic 1D interpolation. This does mean some branching, but most likely all threads will
                        // be doing the same thing so the divergence should be minimal
                        T borderValue = values[INDEX(0, pixelY + i, inputRowStep)];
                        T nextValue = values[INDEX(1, pixelY + i, inputRowStep)];
                        value = borderValue + (nextValue - borderValue) * j;
                    }

                    shared[SHARED_INDEX(sharedPixelX - j, sharedPixelY + i, xPadding, yPadding, sharedRowStep)] = value;
                }
            }
        }

        if (threadIdx.x == blockDim.x - 1) {
            int maxSharedX = XBatch * blockDim.x;

            // Copy extra columns on right
            for (int i = 0; i < YBatch; i++) {
                if (pixelY + i >= height) {
                    break;
                }

                for (int j = 0; j < xPadding; j++) {
                    T value;

                    if (pixelX + XBatch + j < width) {
                        value = values[INDEX(pixelX + XBatch + j, pixelY + i, inputRowStep)];
                    } else if (!Interpolate) {
                        value = values[INDEX(width - 1, pixelY + i, inputRowStep)];
                    } else {
                        // Perform basic 1D interpolation. This does mean some branching, but most likely all threads will
                        // be doing the same thing so the divergence should be minimal
                        T borderValue = values[INDEX(width - 1, pixelY + i, inputRowStep)];
                        T prevValue = values[INDEX(width - 2, pixelY + i, inputRowStep)];
                        value = borderValue + (borderValue - prevValue) * (j + 1);
                    }

                    shared[SHARED_INDEX(maxSharedX + j, sharedPixelY + i, xPadding, yPadding, sharedRowStep)] = value;
                }
            }
        }
    }
}

inline void gpuAssert(log4cxx::LoggerPtr logger, cudaError_t code, const char *file, int line, bool abort = true) {
    if (code == cudaSuccess) {
        return;
    }

    LOG4CXX_ERROR(logger, "An error occurred while performing CUDA operation: " << cudaGetErrorString(code) << " " << file << " " << line);
    if (abort) {
        exit(code);
    }
}

void reportMemoryUsage(log4cxx::LoggerPtr logger);
}  // namespace cart
