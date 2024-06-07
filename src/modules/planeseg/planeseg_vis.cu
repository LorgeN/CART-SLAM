#include <cuda_runtime.h>

#include <opencv2/core/cuda_stream_accessor.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/ximgproc/disparity_filter.hpp>

#include "cartslam.hpp"
#include "modules/disparity.hpp"
#include "modules/planeseg.hpp"
#include "utils/cuda.cuh"
#include "utils/modules.hpp"
#include "utils/peaks.hpp"

#define THREADS_PER_BLOCK_X 16
#define THREADS_PER_BLOCK_Y 16
#define X_BATCH 8
#define Y_BATCH 8

#define COLOR(plane) cart::PlaneColor<cart::Plane::plane>()

struct plane_colors_t {
    const int colors[3][3] = {
        {COLOR(HORIZONTAL).b / 2, COLOR(HORIZONTAL).g / 2, COLOR(HORIZONTAL).r / 2},
        {COLOR(VERTICAL).b / 2, COLOR(VERTICAL).g / 2, COLOR(VERTICAL).r / 2},
        {COLOR(UNKNOWN).b / 2, COLOR(UNKNOWN).g / 2, COLOR(UNKNOWN).r / 2}};
} planeColors;

__global__ void overlayPlanes(cv::cuda::PtrStepSz<uint8_t> image, cv::cuda::PtrStepSz<uint8_t> planes, cv::cuda::PtrStepSz<uint8_t> output, plane_colors_t colors) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int pixelX = x * X_BATCH;
    int pixelY = y * Y_BATCH;

    size_t outputRowStep = output.step / sizeof(uint8_t);
    size_t imageRowStep = image.step / sizeof(uint8_t);
    size_t planesRowStep = planes.step / sizeof(uint8_t);

    for (int i = 0; i < Y_BATCH; i++) {
        for (int j = 0; j < X_BATCH; j++) {
            if (pixelX + j >= image.cols || pixelY + i >= image.rows) {
                continue;
            }

            uint8_t plane = planes[INDEX(pixelX + j, pixelY + i, planesRowStep)];

            uint8_t b = image[INDEX_BGR(pixelX + j, pixelY + i, 0, imageRowStep)];
            uint8_t g = image[INDEX_BGR(pixelX + j, pixelY + i, 1, imageRowStep)];
            uint8_t r = image[INDEX_BGR(pixelX + j, pixelY + i, 2, imageRowStep)];

            output[INDEX_BGR(pixelX + j, pixelY + i, 0, outputRowStep)] = b / 2 + colors.colors[plane][0];
            output[INDEX_BGR(pixelX + j, pixelY + i, 1, outputRowStep)] = g / 2 + colors.colors[plane][1];
            output[INDEX_BGR(pixelX + j, pixelY + i, 2, outputRowStep)] = r / 2 + colors.colors[plane][2];
        }
    }
}

__global__ void paintBEVPlanes(cv::cuda::PtrStepSz<uint8_t> planes, cv::cuda::PtrStepSz<float> depth, cv::cuda::PtrStepSz<uint8_t> output, float maxDepth) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int pixelX = x * X_BATCH;
    int pixelY = y * Y_BATCH;

    size_t outputRowStep = output.step / sizeof(uint8_t);
    size_t depthRowStep = depth.step / sizeof(float);
    size_t planesRowStep = planes.step / sizeof(uint8_t);

    // Make sure we use the same scale along the x and y axis
    const float maxWidth = (maxDepth / output.rows) * (output.cols / 2);

    for (int i = 0; i < Y_BATCH; i++) {
        for (int j = 0; j < X_BATCH; j++) {
            if (pixelX + j >= depth.cols || pixelY + i >= depth.rows) {
                continue;
            }

            uint8_t plane = planes[INDEX(pixelX + j, pixelY + i, planesRowStep)];

            if (plane != cart::Plane::VERTICAL) {
                continue;
            }

            // depth is XYZ
            float x = depth[INDEX_CH(pixelX + j, pixelY + i, 3, 0, depthRowStep)];
            float y = depth[INDEX_CH(pixelX + j, pixelY + i, 3, 1, depthRowStep)];
            float z = depth[INDEX_CH(pixelX + j, pixelY + i, 3, 2, depthRowStep)];
            if (z > maxDepth || z < 0.0f || x < -10.0f || x > 10.0f) {
                continue;
            }

            // Normalize depth to 0 - rows (height) of the output image
            int row = output.rows - static_cast<int>(round((z / maxDepth) * output.rows)) - 1;

            int column = static_cast<int>(round((x / maxWidth) * output.cols)) + (output.cols / 2);

            int targetChannel = y > -0.5f ? 0 : 1;

            // This will result in race conditions, but it's fine for visualization. The idea is that
            // taller vertical planes will be more visible.
            uint8_t curr = output[INDEX_BGR(column, row, targetChannel, outputRowStep)];
            curr -= min(curr, static_cast<int>(ceil(1 * (z / 3 + 1))));
            output[INDEX_BGR(column, row, targetChannel, outputRowStep)] = curr;
            output[INDEX_BGR(column, row, 2, outputRowStep)] = curr;
        }
    }
}

namespace cart {

boost::future<system_data_t> DisparityPlaneSegmentationVisualizationModule::run(System& system, SystemRunData& data) {
    auto promise = boost::make_shared<boost::promise<system_data_t>>();

    boost::asio::post(system.getThreadPool(), [this, promise, &system, &data]() {
        auto planes = data.getData<cv::cuda::GpuMat>(CARTSLAM_KEY_PLANES);

        if (!planes->empty()) {
            // Show image
            auto referenceImage = getReferenceImage(data.dataElement);

            dim3 threadsPerBlock(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y);
            dim3 numBlocks((planes->cols + (threadsPerBlock.x * X_BATCH - 1)) / (threadsPerBlock.x * X_BATCH),
                           (planes->rows + (threadsPerBlock.y * Y_BATCH - 1)) / (threadsPerBlock.y * Y_BATCH));

            cv::cuda::GpuMat output(planes->size(), CV_8UC3);
            cv::Mat image, unsmoothedImage;

            cv::cuda::Stream cvStream;
            cudaStream_t stream = cv::cuda::StreamAccessor::getStream(cvStream);

            overlayPlanes<<<numBlocks, threadsPerBlock, 0, stream>>>(referenceImage, *planes, output, planeColors);
            CUDA_SAFE_CALL(this->logger, cudaPeekAtLastError());

            bool unsmoothed = this->showStacked && data.hasData(CARTSLAM_KEY_PLANES_UNSMOOTHED);
            if (unsmoothed) {
                auto unsmoothedPlanes = data.getData<cv::cuda::GpuMat>(CARTSLAM_KEY_PLANES_UNSMOOTHED);

                cv::cuda::GpuMat unsmoothedOutput(unsmoothedPlanes->size(), CV_8UC3);
                overlayPlanes<<<numBlocks, threadsPerBlock, 0, stream>>>(referenceImage, *unsmoothedPlanes, unsmoothedOutput, planeColors);
                CUDA_SAFE_CALL(this->logger, cudaPeekAtLastError());

                unsmoothedOutput.download(unsmoothedImage, cvStream);
            }

            output.download(image, cvStream);
            CUDA_SAFE_CALL(this->logger, cudaStreamSynchronize(stream));

            if (unsmoothed) {
                cv::Mat stacked;
                cv::vconcat(image, unsmoothedImage, stacked);
                this->imageThread->setImageIfLater(stacked, data.id);
            } else {
                this->imageThread->setImageIfLater(image, data.id);
            }
        }

        if (!system.hasData(CARTSLAM_KEY_DISPARITY_DERIVATIVE_HIST) || !this->showHistogram) {
            promise->set_value(MODULE_NO_RETURN_VALUE);
            return;
        }

        auto histSource = system.getData<cv::Mat>(CARTSLAM_KEY_DISPARITY_DERIVATIVE_HIST);

        int histSize = 256;
        int hist_w = 1024, hist_h = 800;
        int bin_w = cvRound((double)hist_w / histSize);

        cv::Mat histImage(hist_h, hist_w, CV_8UC3, cv::Scalar(0, 0, 0));

        cv::Mat hist;
        cv::normalize(*histSource, hist, 0, histImage.rows - 20, cv::NORM_MINMAX, -1);

        if (system.hasData(CARTSLAM_KEY_PLANE_PARAMETERS)) {
            LOG4CXX_DEBUG(this->logger, "Drawing plane parameters");
            auto parameters = system.getData<PlaneParameters>(CARTSLAM_KEY_PLANE_PARAMETERS);

            cv::circle(histImage, cv::Point((parameters->horizontalCenter + 128) * bin_w, hist_h - 5), 3, planeColor<Plane::HORIZONTAL>(), -1);
            cv::circle(histImage, cv::Point((parameters->verticalCenter + 128) * bin_w, hist_h - 5), 3, planeColor<Plane::VERTICAL>(), -1);

            int horizStart = parameters->horizontalRange.first + 128;
            int horizEnd = parameters->horizontalRange.second + 128;
            int vertStart = parameters->verticalRange.first + 128;
            int vertEnd = parameters->verticalRange.second + 128;

            for (int i = 1; i < histSize; i++) {
                cv::Scalar color = planeColor<Plane::UNKNOWN>();
                if (i >= horizStart && i < horizEnd) {
                    color = planeColor<Plane::HORIZONTAL>();
                } else if (i >= vertStart && i < vertEnd) {
                    color = planeColor<Plane::VERTICAL>();
                }

                cv::line(histImage, cv::Point(bin_w * (i - 1), hist_h - hist.at<int>(i - 1)),
                         cv::Point(bin_w * (i), hist_h - cvRound(hist.at<int>(i))),
                         color, 2, 8, 0);
            }
        } else {
            for (int i = 1; i < histSize; i++) {
                cv::line(histImage, cv::Point(bin_w * (i - 1), hist_h - hist.at<int>(i - 1)),
                         cv::Point(bin_w * (i), hist_h - cvRound(hist.at<int>(i))),
                         cv::Scalar(255, 0, 0), 2, 8, 0);
            }
        }

        this->histThread->setImageIfLater(histImage, data.id);

        promise->set_value(MODULE_NO_RETURN_VALUE);
    });

    return promise->get_future();
}

bool PlaneSegmentationBEVVisualizationModule::updateImage(System& system, SystemRunData& data, cv::Mat& image) {
    auto planes = data.getData<cv::cuda::GpuMat>(CARTSLAM_KEY_PLANES);
    auto depth = data.getData<cv::cuda::GpuMat>(CARTSLAM_KEY_DEPTH);

    if (planes->empty() || depth->empty()) {
        return false;
    }

    cv::cuda::GpuMat output(300, 600, CV_8UC3);

    cv::cuda::Stream cvStream;
    cudaStream_t stream = cv::cuda::StreamAccessor::getStream(cvStream);

    dim3 threadsPerBlock(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y);
    dim3 numBlocks((planes->cols + (threadsPerBlock.x * X_BATCH - 1)) / (threadsPerBlock.x * X_BATCH),
                   (planes->rows + (threadsPerBlock.y * Y_BATCH - 1)) / (threadsPerBlock.y * Y_BATCH));

    output.setTo(cv::Scalar(255, 255, 255), cvStream);
    paintBEVPlanes<<<numBlocks, threadsPerBlock, 0, stream>>>(*planes, *depth, output, 20.0);
    cv::cuda::resize(output, output, cv::Size(1200, 600), 0, 0, cv::INTER_NEAREST, cvStream);

    output.download(image, cvStream);

    cvStream.waitForCompletion();
    return true;
}
}  // namespace cart