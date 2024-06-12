#include <opencv2/core/cuda_stream_accessor.hpp>

#include "cartslam.hpp"
#include "modules/disparity.hpp"
#include "modules/disparity/interpolation.cuh"
#include "utils/cuda.cuh"

#define THREADS_PER_BLOCK_X 16
#define THREADS_PER_BLOCK_Y 16
#define X_BATCH 4
#define Y_BATCH 4

#define SHARED_SIZE(radius) (((X_BATCH * THREADS_PER_BLOCK_X + (radius - 1) * 2) * (Y_BATCH * THREADS_PER_BLOCK_Y + (radius - 1) * 2)) * sizeof(cart::disparity_t))

#define LOCAL_INDEX(x, y) SHARED_INDEX(sharedPixelX + x, sharedPixelY + y, radius - 1, radius - 1, sharedRowStep)

__global__ void interpolateKernel(cv::cuda::PtrStepSz<cart::disparity_t> disparity, int radius, int width, int height, int iterations, int minDisparity, int maxDisparity) {
    extern __shared__ cart::disparity_t shared[];

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int sharedPixelX = threadIdx.x * X_BATCH;
    int sharedPixelY = threadIdx.y * Y_BATCH;

    int pixelX = x * X_BATCH;
    int pixelY = y * Y_BATCH;

    size_t sharedRowStep = X_BATCH * blockDim.x;

    cart::copyToShared<cart::disparity_t, X_BATCH, Y_BATCH>(shared, disparity, radius - 1, radius - 1);

    const unsigned int minCount = radius * radius + 1;  // Requires about 1 fourth of the pixels to be valid

    __syncthreads();

    // Average neighboring pixels
    for (int i = 0; i < iterations; i++) {
        for (int j = 0; j < X_BATCH; j++) {
            for (int i = 0; i < Y_BATCH; i++) {
                if (pixelX + j >= width || pixelY + i >= height) {
                    continue;
                }

                int sum = 0;
                int count = 0;

                for (int k = -radius + 1; k < radius; k++) {
                    for (int l = -radius + 1; l < radius; l++) {
                        cart::disparity_t value = shared[LOCAL_INDEX(j + k, i + l)];

                        if (value > minDisparity && value < maxDisparity) {
                            sum += value;
                            count++;
                        }
                    }
                }

                if (count > minCount) {
                    shared[LOCAL_INDEX(j, i)] = sum / count;
                } else {
                    shared[LOCAL_INDEX(j, i)] = CARTSLAM_DISPARITY_INVALID;
                }
            }
        }

        __syncthreads();
    }

    size_t disparityStep = disparity.step / sizeof(cart::disparity_t);

    // Write back to global memory
    for (int i = 0; i < Y_BATCH; i++) {
        for (int j = 0; j < X_BATCH; j++) {
            if (pixelX + j >= width || pixelY + i >= height) {
                continue;
            }

            disparity[INDEX(pixelX + j, pixelY + i, disparityStep)] = shared[LOCAL_INDEX(j, i)];
        }
    }
}

namespace cart::disparity {
void interpolate(log4cxx::LoggerPtr logger, cv::cuda::GpuMat& disparity, cv::cuda::Stream& stream, int radius, int iterations, int minDisparity, int maxDisparity) {
    int width = disparity.cols;
    int height = disparity.rows;

    dim3 threadsPerBlock(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y);
    dim3 numBlocks((disparity.cols + (threadsPerBlock.x * X_BATCH - 1)) / (threadsPerBlock.x * X_BATCH),
                   (disparity.rows + (threadsPerBlock.y * Y_BATCH - 1)) / (threadsPerBlock.y * Y_BATCH));

    int sharedSize = SHARED_SIZE(radius);

    cudaStream_t cudaStream = cv::cuda::StreamAccessor::getStream(stream);
    interpolateKernel<<<numBlocks, threadsPerBlock, sharedSize, cudaStream>>>(disparity, radius, width, height, iterations, minDisparity, maxDisparity);

    CUDA_SAFE_CALL(logger, cudaGetLastError());
}
}  // namespace cart::disparity