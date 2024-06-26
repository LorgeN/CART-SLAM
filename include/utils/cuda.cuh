#pragma once

#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
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

// Based on https://forums.developer.nvidia.com/t/how-to-use-atomiccas-to-implement-atomicadd-short-trouble-adapting-programming-guide-example/22712/9
// Not safe for overflow, but should be faster than atomicAdd
__device__ inline uint16_t unsafeAtomicAdd(uint16_t *address, uint16_t val) {
    unsigned int *base_address = (unsigned int *)((size_t)address & ~2);  // Align to 4 bytes
    unsigned int intVal = ((size_t)address & 2) ? ((unsigned int)val << 16) : val;
    unsigned int intOld = atomicAdd(base_address, intVal);
    return ((size_t)address & 2) ? (unsigned short)(intOld >> 16) : (unsigned short)(intOld & 0xffff);
}

namespace cg = cooperative_groups;

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

__device__ void assignColor(float idx, uint8_t *pix);

__device__ void assignColor(float fx, float fy, uint8_t *pix);

template <typename T, int XBatch, int YBatch, bool Interpolate = true>
__device__ void copyToShared(T *shared, cv::cuda::PtrStepSz<T> values, const int yPadding, const int xPadding) {
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

    cg::thread_block tb = cg::this_thread_block();

    const size_t startX = (blockIdx.x * blockDim.x) * XBatch;
    const size_t startY = (blockIdx.y * blockDim.y) * YBatch;

    int paddedXStart = max(0, static_cast<int>(startX) - xPadding);
    int paddedSharedXStart = paddedXStart - startX;

    int paddedYStart = max(0, static_cast<int>(startY) - yPadding);
    int paddedSharedYStart = paddedYStart - startY;

    int xDim = min(static_cast<unsigned int>(width - paddedXStart), static_cast<unsigned int>(XBatch * blockDim.x + 2 * xPadding));
    int yDim = min(static_cast<unsigned int>(height - paddedYStart), static_cast<unsigned int>(YBatch * blockDim.y + 2 * yPadding));

    // Copy everything we can copy, and dont need to interpolate
    for (int i = 0; i < yDim; i++) {
        cg::memcpy_async(tb, &shared[SHARED_INDEX(paddedSharedXStart, paddedSharedYStart + i, xPadding, yPadding, sharedRowStep)],
                         &values[INDEX(paddedXStart, startY + i, inputRowStep)], xDim * sizeof(T));
    }

    cg::wait(tb);

    // Handle necessary interpolation

    if (static_cast<int>(startY) - yPadding < 0) {
        // Copy extra rows on top
        for (int i = 1; i <= yPadding; i++) {
            if constexpr (!Interpolate) {
                cg::memcpy_async(tb, &shared[SHARED_INDEX(paddedSharedXStart, -i, xPadding, yPadding, sharedRowStep)],
                                 &values[INDEX(paddedXStart, 0, inputRowStep)], xDim * sizeof(T));
            } else if (threadIdx.y == 0) {
                for (int j = 0; j < XBatch; j++) {
                    if (pixelX + j >= width) {
                        break;
                    }

                    T borderValue = shared[SHARED_INDEX(sharedPixelX + j, 0, xPadding, yPadding, sharedRowStep)];
                    T nextValue = shared[SHARED_INDEX(sharedPixelY + j, i, xPadding, yPadding, sharedRowStep)];
                    T value = borderValue + (nextValue - borderValue);

                    shared[SHARED_INDEX(sharedPixelX + j, sharedPixelY - i, xPadding, yPadding, sharedRowStep)] = value;
                }
            }
        }
    }

    if (startY + YBatch * blockDim.y + yPadding > height) {
        // Copy extra rows on bottom
        for (int i = 0; i < yPadding; i++) {
            if constexpr (!Interpolate) {
                cg::memcpy_async(tb, &shared[SHARED_INDEX(paddedSharedXStart, YBatch * blockDim.y + i, xPadding, yPadding, sharedRowStep)],
                                 &values[INDEX(paddedXStart, height - 1, inputRowStep)], xDim * sizeof(T));
            } else if (threadIdx.y == blockDim.y - 1) {
                for (int j = 0; j < XBatch; j++) {
                    if (pixelX + j >= width) {
                        break;
                    }

                    T borderValue = shared[SHARED_INDEX(sharedPixelX + j, YBatch * blockDim.y - 1, xPadding, yPadding, sharedRowStep)];
                    T prevValue = shared[SHARED_INDEX(sharedPixelX + j, YBatch * blockDim.y - 2 - i, xPadding, yPadding, sharedRowStep)];
                    T value = borderValue + (borderValue - prevValue);

                    shared[SHARED_INDEX(sharedPixelX + j, YBatch * blockDim.y + i, xPadding, yPadding, sharedRowStep)] = value;
                }
            }
        }
    }

    if (static_cast<int>(startX) - xPadding < 0) {
        for (int i = 1; i <= xPadding; i++) {
            if constexpr (!Interpolate) {
                for (int j = 0; j < yDim; j++) {
                    cg::memcpy_async(tb, &shared[SHARED_INDEX(-i, j, xPadding, yPadding, sharedRowStep)],
                                     &values[INDEX(0, startY + j, inputRowStep)], sizeof(T));
                }
            } else if (threadIdx.x == 0) {
                for (int j = 0; j < YBatch; j++) {
                    if (pixelY + j >= height) {
                        break;
                    }

                    T borderValue = shared[SHARED_INDEX(0, j, xPadding, yPadding, sharedRowStep)];
                    T nextValue = shared[SHARED_INDEX(i, j, xPadding, yPadding, sharedRowStep)];
                    T value = borderValue + (nextValue - borderValue);

                    shared[SHARED_INDEX(-i, j, xPadding, yPadding, sharedRowStep)] = value;
                }
            }
        }
    }

    if (startX + XBatch * blockDim.x + xPadding > width) {
        for (int i = 0; i < xPadding; i++) {
            if constexpr (!Interpolate) {
                for (int j = 0; j < yDim; j++) {
                    cg::memcpy_async(tb, &shared[SHARED_INDEX(XBatch * blockDim.x + i, j, xPadding, yPadding, sharedRowStep)],
                                     &values[INDEX(width - 1, startY + j, inputRowStep)], sizeof(T));
                }
            } else if (threadIdx.x == blockDim.x - 1) {
                for (int j = 0; j < YBatch; j++) {
                    if (pixelY + j >= height) {
                        break;
                    }

                    T borderValue = shared[SHARED_INDEX(XBatch * blockDim.x - 1, j, xPadding, yPadding, sharedRowStep)];
                    T prevValue = shared[SHARED_INDEX(XBatch * blockDim.x - 2 - i, j, xPadding, yPadding, sharedRowStep)];
                    T value = borderValue + (borderValue - prevValue);

                    shared[SHARED_INDEX(XBatch * blockDim.x + i, j, xPadding, yPadding, sharedRowStep)] = value;
                }
            }
        }
    }

    cg::wait(tb);
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
