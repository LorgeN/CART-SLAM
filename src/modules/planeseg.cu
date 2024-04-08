#include "modules/disparity.hpp"
#include "modules/planeseg.hpp"
#include "utils/cuda.cuh"

// OpenCV uses row-major order
#define INDEX(x, y, width) ((y) * (width) + (x))
// We expand the image by 1 pixel on each vertical to avoid boundary checks
#define SHARED_INDEX(x, y, width) ((y + 1) * (width) + (x))
#define CLAMP(x, a, b) (max((a), min((b), (x))))

#define X_BATCH 16
#define Y_BATCH 16
#define SHARED_SIZE(blockDim) ((X_BATCH * blockDim.x) * (Y_BATCH * (2 + blockDim.y)) * sizeof(uint8_t))

__global__ void calculateDerivatives(uint8_t* disparity, int16_t* output, int width, int height) {
    extern __shared__ uint8_t sharedDisparity[];

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int sharedPixelX = threadIdx.x * X_BATCH;
    int sharedPixelY = threadIdx.y * Y_BATCH;

    int pixelX = x * X_BATCH;
    int pixelY = y * Y_BATCH;

    for (int i = -1; i <= Y_BATCH; i++) {  // Go one row further in each direction
        for (int j = 0; j < X_BATCH; j++) {
            size_t index = INDEX(CLAMP(pixelX + j, 0, width - 1), CLAMP(pixelY + i, 0, height - 1), width);
            sharedDisparity[SHARED_INDEX(sharedPixelX + j, sharedPixelY + i, X_BATCH)] = disparity[index];
        }
    }

    __syncthreads();

    // Perform vertical low pass filter
    for (int i = 0; i < Y_BATCH; i++) {
        for (int j = 0; j < X_BATCH; j++) {
            int sum = 0;
#pragma unroll
            for (int k = -1; k <= 1; k++) {
                sum += sharedDisparity[SHARED_INDEX(sharedPixelX + j, sharedPixelY + i + k, X_BATCH)];
            }

            sharedDisparity[SHARED_INDEX(sharedPixelX + j, sharedPixelY + i, X_BATCH)] = sum / 3;
        }
    }

    __syncthreads();

    // Calculate vertical derivatives
    for (int i = 0; i < Y_BATCH; i++) {
        for (int j = 0; j < X_BATCH; j++) {
            if (pixelX + j >= width || pixelY + i >= height) {
                continue;
            }

            int16_t derivative =
                sharedDisparity[SHARED_INDEX(sharedPixelX + j, sharedPixelY + i + 1, X_BATCH)] -
                sharedDisparity[SHARED_INDEX(sharedPixelX + j, sharedPixelY + i - 1, X_BATCH)];

            output[INDEX(pixelX + j, pixelY + i, width)] = derivative;
        }
    }
}

namespace cart {
MODULE_RETURN_VALUE DisparityPlaneSegmentationModule::runInternal(System& system, SystemRunData& data) {
    LOG4CXX_DEBUG(this->logger, "Running disparity plane segmentation");
    auto disparity = data.getData<cv::cuda::GpuMat>(CARTSLAM_KEY_DISPARITY);

    if (disparity->empty()) {
        LOG4CXX_WARN(system.logger, "Disparity is empty");
        return MODULE_NO_RETURN_VALUE;
    }

    if (disparity->type() != CV_8UC1) {
        throw std::runtime_error("Disparity must be of type CV_8UC1");
    }

    cv::cuda::GpuMat derivatives;
    derivatives.create(disparity->size(), CV_16SC1);

    dim3 threadsPerBlock(16, 8);
    dim3 numBlocks((disparity->cols + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (disparity->rows + threadsPerBlock.y - 1) / threadsPerBlock.y);

    LOG4CXX_DEBUG(this->logger, "Launching kernel with " << numBlocks.x << "x" << numBlocks.y << " blocks and "
                                                         << threadsPerBlock.x << "x" << threadsPerBlock.y << " threads");
    LOG4CXX_DEBUG(this->logger, "Shared memory size: " << SHARED_SIZE(threadsPerBlock));

    calculateDerivatives<<<numBlocks, threadsPerBlock, SHARED_SIZE(threadsPerBlock)>>>((uint8_t*)disparity->cudaPtr(), (int16_t*)derivatives.cudaPtr(), disparity->cols, disparity->rows);

    CUDA_SAFE_CALL(this->logger, cudaPeekAtLastError());
    CUDA_SAFE_CALL(this->logger, cudaDeviceSynchronize());

    LOG4CXX_DEBUG(this->logger, "Derivatives calculated");
    return MODULE_RETURN(CARTSLAM_KEY_PLANES, boost::make_shared<cv::cuda::GpuMat>(boost::move(derivatives)));
}

boost::future<MODULE_RETURN_VALUE> DisparityPlaneSegmentationVisualizationModule::run(System& system, SystemRunData& data) {
    auto promise = boost::make_shared<boost::promise<MODULE_RETURN_VALUE>>();

    boost::asio::post(system.threadPool, [this, promise, &system, &data]() {
        auto planes = data.getData<cv::cuda::GpuMat>(CARTSLAM_KEY_PLANES);

        if (!planes->empty()) {
            // Show image
            cv::Mat image;

            planes->download(image);
            cv::normalize(image, image, -255, 255, cv::NORM_MINMAX);
            image.convertTo(image, CV_8UC1, 0.5, 128);
            
            this->imageThread.setImageIfLater(image, data.id);
        }

        promise->set_value(MODULE_NO_RETURN_VALUE);
    });

    return promise->get_future();
}
}  // namespace cart