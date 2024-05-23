#include <opencv2/core/cuda_stream_accessor.hpp>
#include <opencv2/cudaimgproc.hpp>

#include "modules/disparity.hpp"
#include "utils/colors.hpp"
#include "utils/cuda.cuh"
#include "utils/cuda_colors.cuh"
#include "utils/modules.hpp"

#define CARTSLAM_DISPARITY_DERIVATIVE_INVALID (-32768)

#define IS_VALID(x) ((x) != CARTSLAM_DISPARITY_INVALID)

#define THREADS_PER_BLOCK_X 32
#define THREADS_PER_BLOCK_Y 32
#define X_BATCH 4
#define Y_BATCH 4

// Pad the shared memory by 1 in each direction
#define SHARED_SIZE ((2 + X_BATCH * THREADS_PER_BLOCK_X) * (2 + Y_BATCH * THREADS_PER_BLOCK_Y))

#define LOCAL_INDEX(x, y) SHARED_INDEX(sharedPixelX + x, sharedPixelY + y, 1, 1, sharedRowStep)

__global__ void calculateDirectionalDerivatives(cv::cuda::PtrStepSz<cart::disparity_t> disparity,
                                                cv::cuda::PtrStepSz<cart::derivative_t> output,
                                                cv::cuda::PtrStepSz<int> histogramOutput) {
    __shared__ cart::disparity_t sharedDisparity[SHARED_SIZE];
    __shared__ int localHistogram[256 * 2];  // 2 channels

    // Initialize histogram to 0
    for (int i = threadIdx.x + (blockDim.x * threadIdx.y); i < 512; i += blockDim.x * blockDim.y) {
        localHistogram[i] = 0;
    }

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int sharedPixelX = threadIdx.x * X_BATCH;
    int sharedPixelY = threadIdx.y * Y_BATCH;

    int pixelX = x * X_BATCH;
    int pixelY = y * Y_BATCH;

    size_t outputRowStep = output.step / sizeof(cart::derivative_t);
    size_t sharedRowStep = X_BATCH * blockDim.x;

    cart::copyToShared<cart::disparity_t, X_BATCH, Y_BATCH>(sharedDisparity, disparity, 1, 1);

    __syncthreads();

    // Calculate derivatives
    for (int i = 0; i < Y_BATCH; i++) {
        for (int j = 0; j < X_BATCH; j++) {
            if (pixelX + j >= disparity.cols || pixelY + i >= disparity.rows) {
                continue;
            }

            cart::derivative_t verticalDerivative =
                sharedDisparity[LOCAL_INDEX(j, i + 1)] -
                sharedDisparity[LOCAL_INDEX(j, i - 1)];

            cart::derivative_t horizontalDerivative =
                sharedDisparity[LOCAL_INDEX(j + 1, i)] -
                sharedDisparity[LOCAL_INDEX(j - 1, i)];

            bool verticalDerivativeValid = sharedDisparity[LOCAL_INDEX(j, i - 1)] != CARTSLAM_DISPARITY_INVALID &&
                                           sharedDisparity[LOCAL_INDEX(j, i + 1)] != CARTSLAM_DISPARITY_INVALID;

            bool horizontalDerivativeValid = sharedDisparity[LOCAL_INDEX(j - 1, i)] != CARTSLAM_DISPARITY_INVALID &&
                                             sharedDisparity[LOCAL_INDEX(j + 1, i)] != CARTSLAM_DISPARITY_INVALID;

            output[INDEX_CH(pixelX + j, pixelY + i, 2, 0, outputRowStep)] = verticalDerivativeValid ? verticalDerivative : CARTSLAM_DISPARITY_DERIVATIVE_INVALID;
            output[INDEX_CH(pixelX + j, pixelY + i, 2, 1, outputRowStep)] = horizontalDerivativeValid ? horizontalDerivative : CARTSLAM_DISPARITY_DERIVATIVE_INVALID;

            if (verticalDerivativeValid && verticalDerivative >= -128 && verticalDerivative <= 127) {
                atomicAdd(&localHistogram[2 * (verticalDerivative + 128)], 1);  // First channel is vertical
            }

            if (horizontalDerivativeValid && horizontalDerivative >= -128 && horizontalDerivative <= 127) {
                atomicAdd(&localHistogram[2 * (horizontalDerivative + 128) + 1], 1);  // Second channel is horizontal
            }
        }
    }

    __syncthreads();

    size_t index = blockIdx.x + blockIdx.y * gridDim.x;
    size_t histStep = histogramOutput.step / sizeof(int);

    for (int i = threadIdx.x + (blockDim.x * threadIdx.y); i < 256; i += blockDim.x * blockDim.y) {
        histogramOutput[INDEX_CH(index, i, 2, 0, histStep)] = localHistogram[2 * i];
        histogramOutput[INDEX_CH(index, i, 2, 1, histStep)] = localHistogram[2 * i + 1];
    }
}

__global__ void mergeDerivativeHistograms(cv::cuda::PtrStepSz<int> histogram, cv::cuda::PtrStepSz<int> output, int threadCount) {
    int channel = blockIdx.x * blockDim.x + threadIdx.x;

    int verticalSum = 0;
    int horizontalSum = 0;

    size_t histStep = histogram.step / sizeof(int);

    for (int i = 0; i < threadCount; i++) {
        verticalSum += histogram[INDEX_CH(i, channel, 2, 0, histStep)];
        horizontalSum += histogram[INDEX_CH(i, channel, 2, 1, histStep)];
    }

    size_t outputStep = output.step / sizeof(int);

    output[INDEX_CH(channel, 0, 2, 0, outputStep)] = verticalSum;
    output[INDEX_CH(channel, 0, 2, 1, outputStep)] = horizontalSum;
}

__global__ void applyFalseColors(cv::cuda::PtrStepSz<cart::derivative_t> derivatives, cv::cuda::PtrStepSz<uint8_t> output, double maxrad) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int pixelX = x * X_BATCH;
    int pixelY = y * Y_BATCH;

    size_t inputRowStep = derivatives.step / sizeof(cart::derivative_t);
    size_t outputRowStep = output.step / sizeof(uint8_t);

    for (int i = 0; i < Y_BATCH; i++) {
        for (int j = 0; j < X_BATCH; j++) {
            if (pixelX + j >= derivatives.cols || pixelY + i >= derivatives.rows) {
                continue;
            }

            cart::derivative_t dx = derivatives[INDEX_CH(pixelX + j, pixelY + i, 2, 0, inputRowStep)];
            cart::derivative_t dy = derivatives[INDEX_CH(pixelX + j, pixelY + i, 2, 1, inputRowStep)];

            if (!IS_VALID(dx) || !IS_VALID(dy)) {
                continue;
            }

            uint8_t* outputPixel = output + INDEX_BGR(pixelX + j, pixelY + i, 0, outputRowStep);
            cart::assignColor(static_cast<double>(dx) / maxrad, static_cast<double>(dy) / maxrad, outputPixel);
        }
    }
}

namespace cart {
system_data_t ImageDisparityDerivativeModule::runInternal(System& system, SystemRunData& data) {
    auto disparity = data.getData<cv::cuda::GpuMat>(CARTSLAM_KEY_DISPARITY);

    assert(!disparity->empty());
    assert(disparity->type() == CV_16SC1);

    cv::cuda::GpuMat derivatives(disparity->size(), CV_16SC2);

    LOG4CXX_DEBUG(this->logger, "Calculating directional derivatives");
    LOG4CXX_DEBUG(this->logger, "Disparity size: " << disparity->size());
    LOG4CXX_DEBUG(this->logger, "Derivatives size: " << derivatives.size());

    dim3 threadsPerBlock(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y);
    dim3 numBlocks((disparity->cols + (threadsPerBlock.x * X_BATCH - 1)) / (threadsPerBlock.x * X_BATCH),
                   (disparity->rows + (threadsPerBlock.y * Y_BATCH - 1)) / (threadsPerBlock.y * Y_BATCH));

    size_t totalBlocks = numBlocks.x * numBlocks.y;
    cv::cuda::GpuMat histogramTempStorage(256, totalBlocks, CV_32SC2);
    cv::cuda::GpuMat histogramOutput(1, 256, CV_32SC2);

    cudaStream_t stream;
    CUDA_SAFE_CALL(this->logger, cudaStreamCreate(&stream));

    calculateDirectionalDerivatives<<<numBlocks, threadsPerBlock, 0, stream>>>(*disparity, derivatives, histogramTempStorage);
    mergeDerivativeHistograms<<<1, 256, 0, stream>>>(histogramTempStorage, histogramOutput, totalBlocks);

    CUDA_SAFE_CALL(this->logger, cudaPeekAtLastError());
    CUDA_SAFE_CALL(this->logger, cudaStreamSynchronize(stream));
    CUDA_SAFE_CALL(this->logger, cudaStreamDestroy(stream));

    return MODULE_RETURN_ALL(
        MODULE_MAKE_PAIR(CARTSLAM_KEY_DISPARITY_DERIVATIVE, cv::cuda::GpuMat, boost::move(derivatives)),
        MODULE_MAKE_PAIR(CARTSLAM_KEY_DISPARITY_DERIVATIVE_HISTOGRAM, cv::cuda::GpuMat, boost::move(histogramOutput)));
}

system_data_t ImageDisparityDerivativeVisualizationModule::runInternal(System& system, SystemRunData& data) {
    auto derivatives = data.getData<cv::cuda::GpuMat>(CARTSLAM_KEY_DISPARITY_DERIVATIVE);

    cv::Mat cpuDerivatives;
    derivatives->download(cpuDerivatives);

    cv::Mat derivativeX, derivativeY, derivativeImage;
    cv::Mat derivativePlanes[2] = {derivativeX, derivativeY};

    cv::split(cpuDerivatives, derivativePlanes);

    derivativeX = derivativePlanes[0];
    derivativeY = derivativePlanes[1];

    derivativeImage.create(derivatives->size(), CV_8UC3);
    derivativeImage.setTo(cv::Scalar::all(0));

    // determine motion range:
    double maxrad = 1;

#pragma omp parallel for reduction(max : maxrad)
    for (int y = 0; y < derivativeX.rows; ++y) {
        double localMax = 0;
        for (int x = 0; x < derivativeX.cols; ++x) {
            derivative_t dx = derivativeX.at<derivative_t>(y, x);
            derivative_t dy = derivativeY.at<derivative_t>(y, x);

            if (!IS_VALID(dx) || !IS_VALID(dy)) {
                continue;
            }

            double value = dx * dx + dy * dy;
            localMax = max(localMax, value);
        }

        maxrad = localMax > maxrad ? localMax : maxrad;
    }

    maxrad = sqrt(maxrad);

    auto reference = getReferenceImage(data.dataElement);
    cv::Mat image;

    cv::cuda::GpuMat derivativeImageGpu(derivatives->size(), CV_8UC3);

    dim3 threadsPerBlock(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y);
    dim3 numBlocks((derivatives->cols + threadsPerBlock.x * X_BATCH - 1) / (threadsPerBlock.x * X_BATCH),
                   (derivatives->rows + threadsPerBlock.y * Y_BATCH - 1) / (threadsPerBlock.y * Y_BATCH));

    cv::cuda::Stream cvStream;

    cudaStream_t stream = cv::cuda::StreamAccessor::getStream(cvStream);

    copyColorWheelToDevice(stream);
    applyFalseColors<<<numBlocks, threadsPerBlock, 0, stream>>>(*derivatives, derivativeImageGpu, maxrad);
    cv::cuda::cvtColor(derivativeImageGpu, derivativeImageGpu, cv::COLOR_HSV2BGR, 0, cvStream);
    reference.download(image, cvStream);
    derivativeImageGpu.download(derivativeImage, cvStream);

    CUDA_SAFE_CALL(this->logger, cudaPeekAtLastError());

    cvStream.waitForCompletion();

    cv::Mat output;
    cv::vconcat(image, derivativeImage, output);

    this->imageThread->setImageIfLater(output, data.id);
    return MODULE_NO_RETURN_VALUE;
}
}  // namespace cart