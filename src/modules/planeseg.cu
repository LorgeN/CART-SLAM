#include <cuda_runtime.h>

#include <opencv2/core/cuda_stream_accessor.hpp>
#include <opencv2/ximgproc/disparity_filter.hpp>

#include "modules/disparity.hpp"
#include "modules/planeseg.hpp"
#include "timing.hpp"
#include "utils/cuda.cuh"
#include "utils/peaks.hpp"

#define LOW_PASS_FILTER_SIZE 5
#define LOW_PASS_FILTER_PADDING (LOW_PASS_FILTER_SIZE / 2)

#define THREADS_PER_BLOCK_X 16
#define THREADS_PER_BLOCK_Y 16
#define X_BATCH 8
#define Y_BATCH 8
#define SHARED_SIZE ((X_BATCH * THREADS_PER_BLOCK_X) * (Y_BATCH * (LOW_PASS_FILTER_PADDING * 2 + THREADS_PER_BLOCK_Y)))

#define LOCAL_INDEX(x, y) SHARED_INDEX(sharedPixelX + x, sharedPixelY + y, 0, LOW_PASS_FILTER_PADDING, sharedRowStep)

#define DISPARITY_SCALING (1.0 / 16.0)

#define ROUND_TO_INT(x) static_cast<int32_t>(round(x))

typedef int16_t derivative_t;

__global__ void calculateDerivatives(cv::cuda::PtrStepSz<cart::disparity_t> disparity, cv::cuda::PtrStepSz<derivative_t> output, cv::cuda::PtrStepSz<int> histogramOutput, int width, int height) {
    __shared__ cart::disparity_t sharedDisparity[SHARED_SIZE];
    __shared__ int localHistogram[256];

    // Initialize histogram to 0
    for (int i = threadIdx.x + (blockDim.x * threadIdx.y); i < 256; i += blockDim.x * blockDim.y) {
        localHistogram[i] = 0;
    }

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int sharedPixelX = threadIdx.x * X_BATCH;
    int sharedPixelY = threadIdx.y * Y_BATCH;

    int pixelX = x * X_BATCH;
    int pixelY = y * Y_BATCH;

    size_t outputRowStep = output.step / sizeof(derivative_t);
    size_t sharedRowStep = X_BATCH * blockDim.x;

    copyToShared<cart::disparity_t>(sharedDisparity, disparity, X_BATCH, Y_BATCH, LOW_PASS_FILTER_PADDING, 0, width, height);

    __syncthreads();

    // Perform vertical low pass filter
    for (int j = 0; j < X_BATCH; j++) {
        // Sliding window sum
        derivative_t sum = 0;

        cart::disparity_t previous[LOW_PASS_FILTER_PADDING] = {0};
        size_t previousIndex = 0;

        for (int i = -LOW_PASS_FILTER_PADDING; i < LOW_PASS_FILTER_PADDING; i++) {
            cart::disparity_t value = sharedDisparity[LOCAL_INDEX(j, i)];
            sum += value;

            if (i < 0) {
                previous[previousIndex] = value;
                previousIndex++;
            }
        }

        previousIndex = 0;

        for (int i = 0; i < Y_BATCH; i++) {
            sum += sharedDisparity[LOCAL_INDEX(j, i + LOW_PASS_FILTER_PADDING)];

            cart::disparity_t current = sharedDisparity[LOCAL_INDEX(j, i)];

            sharedDisparity[LOCAL_INDEX(j, i)] = sum / LOW_PASS_FILTER_SIZE;

            sum -= previous[previousIndex];
            previous[previousIndex] = current;
            previousIndex = (previousIndex + 1) % LOW_PASS_FILTER_PADDING;
        }
    }

    __syncthreads();

    // Calculate vertical derivatives
    for (int i = 0; i < Y_BATCH; i++) {
        for (int j = 0; j < X_BATCH; j++) {
            if (pixelX + j >= width || pixelY + i >= height) {
                continue;
            }

            derivative_t derivative =
                sharedDisparity[LOCAL_INDEX(j, i + 1)] -
                sharedDisparity[LOCAL_INDEX(j, i - 1)];

            output[INDEX(pixelX + j, pixelY + i, outputRowStep)] = derivative;

            // Only update histogram if the value is within the range of a signed char
            if (derivative >= -128 && derivative <= 127) {
                atomicAdd(&localHistogram[derivative + 128], 1);
            }
        }
    }

    __syncthreads();

    // We split this work over two kernels to avoid memory access conflicts
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        size_t index = blockIdx.x + blockIdx.y * gridDim.x;
        size_t histStep = histogramOutput.step / sizeof(int);

        for (int i = 0; i < 256; i++) {
            histogramOutput[INDEX(index, i, histStep)] = localHistogram[i];
        }
    }
}

__global__ void histogramMerge(cv::cuda::PtrStepSz<int> histogram, cv::cuda::PtrStepSz<int> output, int threadCount) {
    // Each thread handles one value, for a total of 256 threads
    int channel = blockIdx.x * blockDim.x + threadIdx.x;

    unsigned int sum = 0;

    size_t histStep = histogram.step / sizeof(int);

    for (int i = 0; i < threadCount; i++) {
        sum += histogram[INDEX(i, channel, histStep)];
    }

    // We keep a running total over time, so instead of setting the value we add it
    atomicAdd(output + channel, sum);
}

__global__ void classifyPlanes(cv::cuda::PtrStepSz<derivative_t> derivatives, cv::cuda::PtrStepSz<uint8_t> planes, int width, int height, int horizontalCenter, int horizontalVariance, int verticalCenter, int verticalVariance) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int pixelX = x * X_BATCH;
    int pixelY = y * Y_BATCH;

    size_t derivativesRowStep = derivatives.step / sizeof(derivative_t);
    size_t planesRowStep = planes.step / sizeof(uint8_t);

    int horizontalStart = horizontalCenter - horizontalVariance;
    int horizontalEnd = horizontalCenter + horizontalVariance;

    int verticalStart = verticalCenter - verticalVariance;
    int verticalEnd = verticalCenter + verticalVariance;

    for (int i = 0; i < Y_BATCH; i++) {
        for (int j = 0; j < X_BATCH; j++) {
            if (pixelX + j >= width || pixelY + i >= height) {
                continue;
            }

            derivative_t derivative = derivatives[INDEX(pixelX + j, pixelY + i, derivativesRowStep)];

            int plane = cart::Plane::UNKNOWN;

            if (derivative >= horizontalStart && derivative <= horizontalEnd) {
                plane = cart::Plane::HORIZONTAL;
            } else if (derivative >= verticalStart && derivative <= verticalEnd) {
                plane = cart::Plane::VERTICAL;
            }

            planes[INDEX(pixelX + j, pixelY + i, planesRowStep)] = plane;
        }
    }
}

__global__ void overlayPlanes(cv::cuda::PtrStepSz<uint8_t> image, cv::cuda::PtrStepSz<uint8_t> planes, cv::cuda::PtrStepSz<uint8_t> output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int pixelX = x * X_BATCH;
    int pixelY = y * Y_BATCH;

    size_t outputRowStep = output.step / sizeof(uint8_t);
    size_t imageRowStep = image.step / sizeof(uint8_t);
    size_t planesRowStep = planes.step / sizeof(uint8_t);

    for (int i = 0; i < Y_BATCH; i++) {
        for (int j = 0; j < X_BATCH; j++) {
            if (pixelX + j >= width || pixelY + i >= height) {
                continue;
            }

            uint8_t plane = planes[INDEX(pixelX + j, pixelY + i, planesRowStep)];

            uint8_t b = image[INDEX_BGR(pixelX + j, pixelY + i, 0, imageRowStep)];
            uint8_t g = image[INDEX_BGR(pixelX + j, pixelY + i, 1, imageRowStep)];
            uint8_t r = image[INDEX_BGR(pixelX + j, pixelY + i, 2, imageRowStep)];

            if (plane == cart::Plane::HORIZONTAL) {
                b = b / 2 + cart::PlaneColor<cart::Plane::HORIZONTAL>().b / 2;
                g = g / 2 + cart::PlaneColor<cart::Plane::HORIZONTAL>().g / 2;
                r = r / 2 + cart::PlaneColor<cart::Plane::HORIZONTAL>().r / 2;
            } else if (plane == cart::Plane::VERTICAL) {
                b = b / 2 + cart::PlaneColor<cart::Plane::VERTICAL>().b / 2;
                g = g / 2 + cart::PlaneColor<cart::Plane::VERTICAL>().g / 2;
                r = r / 2 + cart::PlaneColor<cart::Plane::VERTICAL>().r / 2;
            }

            output[INDEX_BGR(pixelX + j, pixelY + i, 0, outputRowStep)] = b;
            output[INDEX_BGR(pixelX + j, pixelY + i, 1, outputRowStep)] = g;
            output[INDEX_BGR(pixelX + j, pixelY + i, 2, outputRowStep)] = r;
        }
    }
}

namespace cart {
system_data_t DisparityPlaneSegmentationModule::runInternal(System& system, SystemRunData& data) {
    LOG4CXX_DEBUG(this->logger, "Running disparity plane segmentation");
    auto disparity = data.getData<cv::cuda::GpuMat>(CARTSLAM_KEY_DISPARITY);

    if (disparity->empty()) {
        LOG4CXX_WARN(this->logger, "Disparity is empty");
        return MODULE_NO_RETURN_VALUE;
    }

    if (disparity->type() != CV_16SC1) {
        LOG4CXX_WARN(this->logger, "Disparity must be of type CV_16SC1, was " << disparity->type() << " (Depth: " << disparity->depth() << ", channels: " << disparity->channels() << ")");
        throw std::runtime_error("Disparity must be of type CV_16SC1");
    }

    cv::cuda::GpuMat derivatives(disparity->size(), CV_16SC1);

    dim3 threadsPerBlock(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y);
    dim3 numBlocks((disparity->cols + (threadsPerBlock.x * X_BATCH - 1)) / (threadsPerBlock.x * X_BATCH),
                   (disparity->rows + (threadsPerBlock.y * Y_BATCH - 1)) / (threadsPerBlock.y * Y_BATCH));

    size_t totalBlocks = numBlocks.x * numBlocks.y;
    cv::cuda::GpuMat histogramTempStorage(256, totalBlocks, CV_32SC1);

    {
        // Read-lock critical section
        boost::shared_lock<boost::shared_mutex> lock(this->derivativeHistogramMutex);

        if (this->derivativeHistogram.empty()) {
            LOG4CXX_DEBUG(this->logger, "Creating histogram");
            this->derivativeHistogram.create(1, 256, CV_32SC1);
            this->derivativeHistogram.setTo(0);
        }

        cv::cuda::Stream cvStream;
        cudaStream_t stream = cv::cuda::StreamAccessor::getStream(cvStream);

        calculateDerivatives<<<numBlocks, threadsPerBlock, 0, stream>>>(*disparity, derivatives, histogramTempStorage, disparity->cols, disparity->rows);
        histogramMerge<<<1, 256, 0, stream>>>(histogramTempStorage, this->derivativeHistogram, totalBlocks);

        CUDA_SAFE_CALL(this->logger, cudaPeekAtLastError());
        CUDA_SAFE_CALL(this->logger, cudaStreamSynchronize(stream));
    }

    LOG4CXX_DEBUG(this->logger, "Derivatives calculated");
    if (data.id % this->updateInterval == 1) {
        this->updatePlaneParameters(system, data);
    }

    cv::cuda::GpuMat planes(disparity->size(), CV_8UC1);

    {
        cudaStream_t stream;
        cudaStreamCreate(&stream);

        classifyPlanes<<<numBlocks, threadsPerBlock, 0, stream>>>(derivatives, planes, disparity->cols, disparity->rows, this->horizontalCenter, this->horizontalVariance, this->verticalCenter, this->verticalVariance);

        CUDA_SAFE_CALL(this->logger, cudaPeekAtLastError());
        CUDA_SAFE_CALL(this->logger, cudaStreamSynchronize(stream));
    }

    return MODULE_RETURN(CARTSLAM_KEY_PLANES, boost::make_shared<cv::cuda::GpuMat>(boost::move(planes)));
}

void DisparityPlaneSegmentationModule::updatePlaneParameters(System& system, SystemRunData& data) {
    // TODO: Inspiration for alternative: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7926705&tag=1
    this->lastUpdatedFrame = data.id;
    // TODO: Find peeks and minimum between peeks

    cv::Mat histogram;

    {
        boost::unique_lock<boost::shared_mutex> lock(this->derivativeHistogramMutex);
        this->derivativeHistogram.download(histogram);
        this->derivativeHistogram.setTo(0);
    }

    std::vector<util::Peak> peaks = util::findPeaks(histogram);

    if (peaks.size() < 2) {
        LOG4CXX_DEBUG(this->logger, "Not enough peaks found");
        return;
    }

    // Set vertical to the peak closest to 0
    if (abs(peaks[0].born - 128) > abs(peaks[1].born - 128)) {
        std::swap(peaks[0], peaks[1]);
    }

    this->verticalCenter = peaks[0].born - 128;
    this->horizontalCenter = peaks[1].born - 128;

    // Find the minimum between the peaks
    int minIndex = min(peaks[0].born, peaks[1].born);

    for (int i = minIndex; i < max(peaks[0].born, peaks[1].born); i++) {
        if (histogram.at<int>(i) < histogram.at<int>(minIndex)) {
            minIndex = i;
        }
    }

    LOG4CXX_DEBUG(this->logger, "Peaks: " << peaks[0].born << ", " << peaks[1].born << ", Min: " << minIndex);

    // Set variance as distance from peak to minimum
    this->verticalVariance = abs(minIndex - peaks[0].born);
    this->horizontalVariance = abs(minIndex - peaks[1].born);

    LOG4CXX_DEBUG(this->logger, "Vertical center: " << this->verticalCenter << ", Horizontal center: " << this->horizontalCenter);
    LOG4CXX_DEBUG(this->logger, "Vertical variance: " << this->verticalVariance << ", Horizontal variance: " << this->horizontalVariance);

    system.insertGlobalData(CARTSLAM_KEY_PLANE_PARAMETERS, boost::make_shared<PlaneParameters>(this->horizontalCenter, this->horizontalVariance, this->verticalCenter, this->verticalVariance));

    auto histShared = boost::make_shared<cv::Mat>(boost::move(histogram));
    system.insertGlobalData(CARTSLAM_KEY_DISPARITY_DERIVATIVE_HIST, histShared);
}

boost::future<system_data_t> DisparityPlaneSegmentationVisualizationModule::run(System& system, SystemRunData& data) {
    auto promise = boost::make_shared<boost::promise<system_data_t>>();

    boost::asio::post(system.getThreadPool(), [this, promise, &system, &data]() {
        auto planes = data.getData<cv::cuda::GpuMat>(CARTSLAM_KEY_PLANES);

        if (!planes->empty()) {
            // Show image
            {
                boost::shared_ptr<cart::StereoDataElement> stereoData = boost::static_pointer_cast<cart::StereoDataElement>(data.dataElement);

                dim3 threadsPerBlock(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y);
                dim3 numBlocks((planes->cols + (threadsPerBlock.x * X_BATCH - 1)) / (threadsPerBlock.x * X_BATCH),
                               (planes->rows + (threadsPerBlock.y * Y_BATCH - 1)) / (threadsPerBlock.y * Y_BATCH));

                cv::cuda::GpuMat output(planes->size(), CV_8UC3);
                overlayPlanes<<<numBlocks, threadsPerBlock>>>(stereoData->left, *planes, output, planes->cols, planes->rows);

                CUDA_SAFE_CALL(this->logger, cudaPeekAtLastError());
                CUDA_SAFE_CALL(this->logger, cudaDeviceSynchronize());

                cv::Mat image;
                output.download(image);
                this->imageThread->setImageIfLater(image, data.id);
            }
        }

        if (!system.hasData(CARTSLAM_KEY_DISPARITY_DERIVATIVE_HIST)) {
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

            int horizStart = parameters->horizontalCenter + 128 - parameters->horizontalVariance;
            int horizEnd = parameters->horizontalCenter + 128 + parameters->horizontalVariance;
            int vertStart = parameters->verticalCenter + 128 - parameters->verticalVariance;
            int vertEnd = parameters->verticalCenter + 128 + parameters->verticalVariance;

            for (int i = 1; i < histSize; i++) {
                cv::Scalar color = planeColor<Plane::UNKNOWN>();
                if (i >= horizStart && i <= horizEnd) {
                    color = planeColor<Plane::HORIZONTAL>();
                } else if (i >= vertStart && i <= vertEnd) {
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
}  // namespace cart