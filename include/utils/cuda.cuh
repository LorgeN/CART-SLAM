#pragma once

#include <log4cxx/logger.h>

#include "modules/disparity.hpp"

// OpenCV uses row-major order
#define INDEX(x, y, rowStep) ((y) * (rowStep) + (x))
#define CLAMP(x, a, b) (max((a), min((b), (x))))
#define SHARED_INDEX(x, y, xPadding, yPadding, rowStep) (((y) + (yPadding)) * ((rowStep) + 2 * (xPadding)) + ((x) + (xPadding)))

#define CUDA_SAFE_CALL(logger, ans) \
    { cart::gpuAssert((logger), (ans), __FILE__, __LINE__); }

template <typename T>
__device__ void copyToShared(T *shared, cv::cuda::PtrStepSz<T> values, int xBatch, int yBatch, int yPadding, int xPadding, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int sharedPixelX = threadIdx.x * xBatch;
    int sharedPixelY = threadIdx.y * yBatch;

    int pixelX = x * xBatch;
    int pixelY = y * yBatch;

    size_t inputRowStep = values.step / sizeof(T);
    size_t sharedRowStep = xBatch * blockDim.x;

    for (int i = 0; i < yBatch; i++) {
        for (int j = 0; j < xBatch; j++) {
            if (pixelX + j >= width || pixelY + i >= height) {
                continue;
            }

            shared[SHARED_INDEX(sharedPixelX + j, sharedPixelY + i, xPadding, yPadding, sharedRowStep)] = values[INDEX(pixelX + j, pixelY + i, inputRowStep)];
        }
    }

    if (yPadding > 0) {
        if (threadIdx.y == 0) {
            // Copy extra rows on top
            for (int i = 1; i <= yPadding; i++) {
                for (int j = 0; j < xBatch; j++) {
                    if (pixelX + j >= width) {
                        break;
                    }

                    T value = values[INDEX(pixelX + j, max(0, pixelY - i), inputRowStep)];
                    shared[SHARED_INDEX(sharedPixelX + j, sharedPixelY - i, xPadding, yPadding, sharedRowStep)] = value;
                }
            }
        }

        if (threadIdx.y == blockDim.y - 1) {
            // Copy extra rows on bottom
            for (int i = 0; i < yPadding; i++) {
                for (int j = 0; j < xBatch; j++) {
                    if (pixelX + j >= width) {
                        break;
                    }

                    T value = values[INDEX(pixelX + j, min(height - 1, pixelY + yBatch + i), inputRowStep)];
                    shared[SHARED_INDEX(sharedPixelX + j, sharedPixelY + yBatch + i, xPadding, yPadding, sharedRowStep)] = value;
                }
            }
        }
    }

    if (xPadding > 0) {
        if (threadIdx.x == 0) {
            // Copy extra columns on left
            for (int i = 0; i < yBatch; i++) {
                if (pixelY + i >= height) {
                    break;
                }

                for (int j = 1; j <= xPadding; j++) {
                    T value = values[INDEX(max(0, pixelX - j), pixelY + i, inputRowStep)];
                    shared[SHARED_INDEX(sharedPixelX - j, sharedPixelY + i, xPadding, yPadding, sharedRowStep)] = value;
                }
            }
        }

        if (threadIdx.x == blockDim.x - 1) {
            // Copy extra columns on right
            for (int i = 0; i < yBatch; i++) {
                for (int j = 0; j < xPadding; j++) {
                    if (pixelY + i >= height) {
                        break;
                    }

                    T value = values[INDEX(min(width - 1, pixelX + xBatch + j), pixelY + i, inputRowStep)];
                    shared[SHARED_INDEX(sharedPixelX + xBatch + j, sharedPixelY + i, xPadding, yPadding, sharedRowStep)] = value;
                }
            }
        }
    }
}

namespace cart {
inline void gpuAssert(log4cxx::LoggerPtr logger, cudaError_t code, const char *file, int line, bool abort = true) {
    if (code == cudaSuccess) {
        return;
    }

    LOG4CXX_ERROR(logger, "An error occurred while performing CUDA operation: " << cudaGetErrorString(code) << " " << file << " " << line);
    if (abort) {
        exit(code);
    }
}
}  // namespace cart
