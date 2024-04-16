#include <opencv2/core/cuda_stream_accessor.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/ximgproc/disparity_filter.hpp>

#include "modules/disparity.hpp"
#include "modules/disparity/interpolation.cuh"
#include "sources/zed.hpp"
#include "utils/cuda.cuh"

#define THREADS_PER_BLOCK_X 16
#define THREADS_PER_BLOCK_Y 16
#define X_BATCH 4
#define Y_BATCH 4

__global__ void processZedDisparity(cv::cuda::PtrStepSz<float> input, cv::cuda::PtrStepSz<int16_t> output) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int pixelX = x * X_BATCH;
    int pixelY = y * Y_BATCH;

    size_t inputRowStep = input.step / sizeof(float);
    size_t outputRowStep = output.step / sizeof(int16_t);

    for (int i = 0; i < Y_BATCH; i++) {
        for (int j = 0; j < X_BATCH; j++) {
            if (pixelX + j >= input.cols || pixelY + i >= input.rows) {
                continue;
            }

            float value = input[INDEX(pixelX + j, pixelY + i, inputRowStep)];

            int16_t result = CARTSLAM_DISPARITY_INVALID;
            // Check if value is infinite
            if (isfinite(value)) {
                result = static_cast<int16_t>(value * -16.0);
            }

            output[INDEX(pixelX + j, pixelY + i, outputRowStep)] = result;
        }
    }
}

namespace cart {
system_data_t ImageDisparityModule::runInternal(System& system, SystemRunData& data) {
    LOG4CXX_DEBUG(this->logger, "Running ImageDisparityModule");

    if (data.dataElement->type != DataElementType::STEREO) {
        throw std::runtime_error("ImageDisparityModule requires StereoDataElement");
    }

    cv::cuda::Stream stream;

    boost::shared_ptr<StereoDataElement> stereoData = boost::static_pointer_cast<StereoDataElement>(data.dataElement);

    cv::cuda::GpuMat left, right;

#ifdef CARTSLAM_IMAGE_MAKE_GRAYSCALE
    left = stereoData->left;
    right = stereoData->right;
#else
    cv::cuda::cvtColor(stereoData->left, left, cv::COLOR_BGR2GRAY, 0, stream);
    cv::cuda::cvtColor(stereoData->right, right, cv::COLOR_BGR2GRAY, 0, stream);
#endif

    cv::cuda::GpuMat disparity;
    this->stereoSGM->compute(left, right, disparity, stream);

    if (this->smoothingRadius > 0) {
        disparity::interpolate(this->logger, disparity, stream, this->smoothingRadius, this->smoothingIterations);
    }

    stream.waitForCompletion();

    return MODULE_RETURN(CARTSLAM_KEY_DISPARITY, boost::make_shared<cv::cuda::GpuMat>(boost::move(disparity)));
}

#ifdef CARTSLAM_ZED
system_data_t ZEDImageDisparityModule::runInternal(System& system, SystemRunData& data) {
    LOG4CXX_DEBUG(this->logger, "Running ZED ImageDisparityModule");

    if (data.dataElement->type != DataElementType::STEREO) {
        throw std::runtime_error("ImageDisparityModule requires StereoDataElement");
    }

    cv::cuda::Stream stream;

    boost::shared_ptr<sources::ZEDDataElement> stereoData = boost::static_pointer_cast<sources::ZEDDataElement>(data.dataElement);

    cv::cuda::GpuMat disparity = stereoData->disparityMeasure;
    if (disparity.empty()) {
        throw std::runtime_error("Disparity measure not available! Make sure to set extractDepthMeasure to true in ZEDDataSource constructor.");
    }

    cv::cuda::GpuMat disparityTransformed(disparity.size(), CV_16SC1);

    dim3 threadsPerBlock(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y);
    dim3 numBlocks((disparity.cols + (threadsPerBlock.x * X_BATCH - 1)) / (threadsPerBlock.x * X_BATCH),
                   (disparity.rows + (threadsPerBlock.y * Y_BATCH - 1)) / (threadsPerBlock.y * Y_BATCH));

    cudaStream_t cudaStream = cv::cuda::StreamAccessor::getStream(stream);
    processZedDisparity<<<numBlocks, threadsPerBlock, 0, cudaStream>>>(disparity, disparityTransformed);

    if (this->smoothingRadius > 0) {
        disparity::interpolate(this->logger, disparityTransformed, stream, this->smoothingRadius, this->smoothingIterations);
    }

    stream.waitForCompletion();

    stereoData->disparityMeasure.release();

    return MODULE_RETURN(CARTSLAM_KEY_DISPARITY, boost::make_shared<cv::cuda::GpuMat>(boost::move(disparityTransformed)));
}
#endif  // CARTSLAM_ZED

boost::future<system_data_t> ImageDisparityVisualizationModule::run(System& system, SystemRunData& data) {
    auto promise = boost::make_shared<boost::promise<system_data_t>>();

    boost::asio::post(system.getThreadPool(), [this, promise, &system, &data]() {
        auto disparity = data.getData<cv::cuda::GpuMat>(CARTSLAM_KEY_DISPARITY);

        cv::cuda::Stream stream;
        cv::Mat image, disparityData, disparityImage;

        boost::shared_ptr<cart::StereoDataElement> stereoData = boost::static_pointer_cast<cart::StereoDataElement>(data.dataElement);

        stereoData->left.download(image, stream);
        disparity->download(disparityData, stream);

        stream.waitForCompletion();

        cv::ximgproc::getDisparityVis(disparityData, disparityImage);

#ifndef CARTSLAM_IMAGE_MAKE_GRAYSCALE
        cv::cvtColor(disparityImage, disparityImage, cv::COLOR_GRAY2BGR);
#endif

        cv::Mat concatRes;
        cv::vconcat(image, disparityImage, concatRes);

        this->imageThread->setImageIfLater(concatRes, data.id);

        promise->set_value(MODULE_NO_RETURN_VALUE);
    });

    return promise->get_future();
}
}  // namespace cart