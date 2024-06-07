#include <cooperative_groups.h>

#include <opencv2/core.hpp>

#include "utils/cuda.cuh"
#include "utils/sanity_check.hpp"

#define THREADS_X 16
#define THREADS_Y 16
#define X_BATCH 4
#define Y_BATCH 4

#define PADDING 2

#define SHARED_SIZE ((PADDING * 2 + X_BATCH * THREADS_X) * (PADDING * 2 + Y_BATCH * THREADS_Y))

namespace cg = cooperative_groups;

__global__ void checkCopyFunction(cv::cuda::PtrStepSz<int> values, cv::cuda::PtrStepSz<uint8_t> output) {
    __shared__ int shared[SHARED_SIZE];

    cart::copyToShared<int, X_BATCH, Y_BATCH>(shared, values, PADDING, PADDING);

    cg::thread_block tb = cg::this_thread_block();

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int sharedPixelX = threadIdx.x * X_BATCH;
    int sharedPixelY = threadIdx.y * Y_BATCH;

    int pixelX = x * X_BATCH;
    int pixelY = y * Y_BATCH;

    size_t outputRowStep = output.step / sizeof(uint8_t);
    size_t sharedRowStep = X_BATCH * blockDim.x;

    tb.sync();

    for (int i = -PADDING; i < Y_BATCH + PADDING; i++) {
        for (int j = -PADDING; j < X_BATCH + PADDING; j++) {
            if (pixelX + j >= values.cols || pixelY + i >= values.rows) {
                continue;
            }

            // Compare to actual value and write to output if different
            int actualValue = values[INDEX(pixelX + j, pixelY + i, values.step / sizeof(int))];
            int sharedValue = shared[SHARED_INDEX(sharedPixelX + j, sharedPixelY + i, PADDING, PADDING, sharedRowStep)];
            if (actualValue != sharedValue) {
                output[INDEX(pixelX + j, pixelY + i, outputRowStep)] = 255;
            }
        }
    }
}

namespace cart::check {
void checkIfCopyWorks(log4cxx::LoggerPtr &logger) {
    cv::Mat values(720, 1280, CV_32SC1);

    // Set values to row * 1000 + col
    for (int y = 0; y < values.rows; y++) {
        for (int x = 0; x < values.cols; x++) {
            values.at<int>(y, x) = y * 1280 + x;
        }
    }

    cv::cuda::GpuMat valuesGpu;
    valuesGpu.upload(values);

    cv::cuda::GpuMat outputGpu(values.size(), CV_8UC1);

    dim3 threads(THREADS_X, THREADS_Y);
    dim3 blocks((values.cols + THREADS_X * X_BATCH - 1) / (THREADS_X * X_BATCH), (values.rows + THREADS_Y * Y_BATCH - 1) / (THREADS_Y * Y_BATCH));

    checkCopyFunction<<<blocks, threads>>>(valuesGpu, outputGpu);

    CUDA_SAFE_CALL(logger, cudaGetLastError());
    CUDA_SAFE_CALL(logger, cudaDeviceSynchronize());

    cv::Mat output;
    outputGpu.download(output);

    cv::imshow("Error", output);
    cv::waitKey(100000000);
    exit(1);
}
}  // namespace cart::check