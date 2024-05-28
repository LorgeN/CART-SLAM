#include <opencv2/core/cuda_stream_accessor.hpp>

#include "modules/superpixels/contourrelaxation/initialization.hpp"
#include "utils/cuda.cuh"

#define THREAD_BLOCK_SIZE 32
#define X_BATCH 8
#define Y_BATCH 8

namespace cart::contour {

__global__ void performBlockIntialization(cv::cuda::PtrStepSz<label_t> labelImage, const int blockWidth, const int blockHeight) {
    // Get the current pixel coordinates.
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    const int pixelX = x * X_BATCH;
    const int pixelY = y * Y_BATCH;

    const int blocksPerRow = ceil(static_cast<double>(labelImage.cols) / blockWidth);
    const size_t labelImageStep = labelImage.step / sizeof(label_t);

    // Set the pixel values
    for (int i = 0; i < X_BATCH; i++) {
        for (int j = 0; j < Y_BATCH; j++) {
            if (pixelX + i >= labelImage.cols || pixelY + j >= labelImage.rows) {
                continue;
            }

            int blockX = (pixelX + i) / blockWidth;
            int blockY = (pixelY + j) / blockHeight;
            int blockLabel = blockY * blocksPerRow + blockX;

            labelImage[INDEX(pixelX + i, pixelY + j, labelImageStep)] = blockLabel;
        }
    }
}

void createBlockInitialization(cv::Size const& imageSize, int const& blockWidth, int const& blockHeight, cv::cuda::GpuMat& labelImage, cart::contour::label_t& maxLabelId, cv::cuda::Stream& cvStream) {
    assert(imageSize.width > 0 && imageSize.height > 0);
    assert(blockWidth > 0 && blockHeight > 0);
    assert(imageSize.width >= blockWidth && imageSize.height >= blockHeight);

    labelImage.create(imageSize, cv::DataType<label_t>::type);

    // Find out how many blocks there will be in each direction. If image size is not a multiple of block size,
    // we need to round upwards because there will be one additional (smaller) block.
    int const numBlocksX = ceil(static_cast<double>(imageSize.width) / blockWidth);
    int const numBlocksY = ceil(static_cast<double>(imageSize.height) / blockHeight);

    maxLabelId = numBlocksX * numBlocksY;

    // Initialize the label image with block labels
    dim3 block(THREAD_BLOCK_SIZE, THREAD_BLOCK_SIZE);
    dim3 grid(ceil(static_cast<double>(imageSize.width) / (block.x * X_BATCH)), ceil(static_cast<double>(imageSize.height) / (block.y * Y_BATCH)));

    cudaStream_t stream = cv::cuda::StreamAccessor::getStream(cvStream);
    performBlockIntialization<<<grid, block, 0, stream>>>(labelImage, blockWidth, blockHeight);
}
}  // namespace cart::contour