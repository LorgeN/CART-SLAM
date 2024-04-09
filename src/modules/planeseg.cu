#include "modules/disparity.hpp"
#include "modules/planeseg.hpp"
#include "utils/cuda.cuh"

#define LOW_PASS_FILTER_SIZE 5
#define LOW_PASS_FILTER_PADDING (LOW_PASS_FILTER_SIZE / 2)

// OpenCV uses row-major order
#define INDEX(x, y, rowStep) ((y) * (rowStep) + (x))
// We expand the image by 1 pixel on each vertical to avoid boundary checks
#define SHARED_INDEX(x, y, rowStep) ((y + LOW_PASS_FILTER_PADDING) * (rowStep) + (x))
#define CLAMP(x, a, b) (max((a), min((b), (x))))

#define THREADS_PER_BLOCK_X 16
#define THREADS_PER_BLOCK_Y 16
#define X_BATCH 8
#define Y_BATCH 8
#define SHARED_SIZE ((X_BATCH * THREADS_PER_BLOCK_X) * (Y_BATCH * (LOW_PASS_FILTER_PADDING + THREADS_PER_BLOCK_Y)) * sizeof(uint8_t))

__global__ void calculateDerivatives(cv::cuda::PtrStepSz<uint8_t> disparity, cv::cuda::PtrStepSz<int16_t> output, int width, int height) {
    __shared__ uint8_t sharedDisparity[SHARED_SIZE];
    __shared__ uint8_t sharedSmoothed[SHARED_SIZE];

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int sharedPixelX = threadIdx.x * X_BATCH;
    int sharedPixelY = threadIdx.y * Y_BATCH;

    int pixelX = x * X_BATCH;
    int pixelY = y * Y_BATCH;

    size_t inputRowStep = disparity.step / sizeof(uint8_t);
    size_t outputRowStep = output.step / sizeof(int16_t);
    size_t sharedRowStep = X_BATCH * blockDim.x;

    for (int i = 0; i < Y_BATCH; i++) {
        for (int j = 0; j < X_BATCH; j++) {
            if (pixelX + j >= width || pixelY + i >= height) {
                continue;
            }

            sharedDisparity[SHARED_INDEX(sharedPixelX + j, sharedPixelY + i, sharedRowStep)] = disparity[INDEX(pixelX + j, pixelY + i, inputRowStep)];
        }
    }

    if (threadIdx.y == 0) {
        // Copy extra rows on top
        for (int i = 1; i <= LOW_PASS_FILTER_PADDING; i++) {
            for (int j = 0; j < X_BATCH; j++) {
                if (pixelX + j >= width) {
                    break;
                }

                // Pad with repeating value on boundaries
                uint8_t value = disparity[INDEX(pixelX + j, max(0, pixelY - i), inputRowStep)];
                sharedDisparity[SHARED_INDEX(sharedPixelX + j, sharedPixelY - i, sharedRowStep)] = value;
                // Trick to avoid having to share border values across thread blocks
                // Note that these values will not be smoothed, but that should be OK
                sharedSmoothed[SHARED_INDEX(sharedPixelX + j, sharedPixelY - i, sharedRowStep)] = value;
            }
        }
    }

    if (threadIdx.y == blockDim.y - 1) {
        // Copy extra rows on bottom
        for (int i = 0; i < LOW_PASS_FILTER_PADDING; i++) {
            for (int j = 0; j < X_BATCH; j++) {
                if (pixelX + j >= width) {
                    break;
                }

                uint8_t value = disparity[INDEX(pixelX + j, min(height - 1, pixelY + Y_BATCH + i), inputRowStep)];
                sharedDisparity[SHARED_INDEX(sharedPixelX + j, sharedPixelY + Y_BATCH + i, sharedRowStep)] = value;
                // Same trick as above
                sharedSmoothed[SHARED_INDEX(sharedPixelX + j, sharedPixelY + Y_BATCH + i, sharedRowStep)] = value;
            }
        }
    }

    __syncthreads();

    // Perform vertical low pass filter
    for (int j = 0; j < X_BATCH; j++) {
        // Sliding window sum
        uint16_t sum = 0;

#pragma unroll
        for (int i = -LOW_PASS_FILTER_PADDING; i < LOW_PASS_FILTER_PADDING; i++) {
            sum += sharedDisparity[SHARED_INDEX(sharedPixelX + j, sharedPixelY + i, sharedRowStep)];
        }

        for (int i = 0; i < Y_BATCH; i++) {
            sum += sharedDisparity[SHARED_INDEX(sharedPixelX + j, sharedPixelY + i + LOW_PASS_FILTER_PADDING, sharedRowStep)];

            sharedSmoothed[SHARED_INDEX(sharedPixelX + j, sharedPixelY + i, sharedRowStep)] = sum / LOW_PASS_FILTER_SIZE;

            sum -= sharedDisparity[SHARED_INDEX(sharedPixelX + j, sharedPixelY + i - LOW_PASS_FILTER_PADDING, sharedRowStep)];
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
                sharedSmoothed[SHARED_INDEX(sharedPixelX + j, sharedPixelY + i + 1, sharedRowStep)] -
                sharedSmoothed[SHARED_INDEX(sharedPixelX + j, sharedPixelY + i - 1, sharedRowStep)];

            output[INDEX(pixelX + j, pixelY + i, outputRowStep)] = derivative;
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

    dim3 threadsPerBlock(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y);
    dim3 numBlocks((disparity->cols + (threadsPerBlock.x * X_BATCH - 1)) / (threadsPerBlock.x * X_BATCH),
                   (disparity->rows + (threadsPerBlock.y * Y_BATCH - 1)) / (threadsPerBlock.y * Y_BATCH));

    LOG4CXX_DEBUG(this->logger, "Launching kernel with " << numBlocks.x << "x" << numBlocks.y << " blocks and "
                                                         << threadsPerBlock.x << "x" << threadsPerBlock.y << " threads");
    LOG4CXX_DEBUG(this->logger, "Shared memory size: " << SHARED_SIZE * 2);

    calculateDerivatives<<<numBlocks, threadsPerBlock>>>(*disparity, derivatives, disparity->cols, disparity->rows);

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
            image.convertTo(image, CV_8UC1, 1.0, 127);

            int histSize = 256;

            float range[] = {0, 256};  // the upper boundary is exclusive

            const float* histRange[] = {range};

            bool uniform = true, accumulate = false;

            cv::Mat hist;

            cv::calcHist(&image, 1, 0, cv::Mat(), hist, 1, &histSize, histRange, uniform, accumulate);

            int hist_w = 512, hist_h = 400;
            int bin_w = cvRound((double)hist_w / histSize);

            cv::Mat histImage(hist_h, hist_w, CV_8UC3, cv::Scalar(0, 0, 0));

            cv::normalize(hist, hist, 0, histImage.rows, cv::NORM_MINMAX, -1);

            for (int i = 1; i < histSize; i++) {
                cv::line(histImage, cv::Point(bin_w * (i - 1), hist_h - cvRound(hist.at<float>(i - 1))),
                         cv::Point(bin_w * (i), hist_h - cvRound(hist.at<float>(i))),
                         cv::Scalar(255, 0, 0), 2, 8, 0);
            }

            this->histThread.setImageIfLater(histImage, data.id);
            this->imageThread.setImageIfLater(image, data.id);
        }

        promise->set_value(MODULE_NO_RETURN_VALUE);
    });

    return promise->get_future();
}
}  // namespace cart