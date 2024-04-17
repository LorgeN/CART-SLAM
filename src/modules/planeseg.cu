#include <cuda_runtime.h>

#include <opencv2/core/cuda_stream_accessor.hpp>
#include <opencv2/ximgproc/disparity_filter.hpp>

#include "modules/disparity.hpp"
#include "modules/planeseg.hpp"
#include "timing.hpp"
#include "utils/cuda.cuh"
#include "utils/modules.hpp"
#include "utils/peaks.hpp"

#define LOW_PASS_FILTER_SIZE 5
#define LOW_PASS_FILTER_PADDING (LOW_PASS_FILTER_SIZE / 2)

#define CARTSLAM_DISPARITY_DERIVATIVE_INVALID (-32768)
#define CARTSLAM_PLANE_TEMPORAL_DISTANCE (CARTSLAM_RUN_RETENTION - CARTSLAM_CONCURRENT_RUN_LIMIT)

#define THREADS_PER_BLOCK_X 16
#define THREADS_PER_BLOCK_Y 16
#define X_BATCH 8
#define Y_BATCH 8
#define SHARED_SIZE ((X_BATCH * THREADS_PER_BLOCK_X) * (Y_BATCH * (LOW_PASS_FILTER_PADDING * 2 + THREADS_PER_BLOCK_Y)))

#define LOCAL_INDEX(x, y) SHARED_INDEX(sharedPixelX + x, sharedPixelY + y, 0, LOW_PASS_FILTER_PADDING, sharedRowStep)

#define DISPARITY_SCALING (1.0 / 16.0)

#define ROUND_TO_INT(x) static_cast<int32_t>(round(x))

typedef int16_t derivative_t;

// TODO: Change disparity values to float?
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
        int count = 0;

        cart::disparity_t previous[LOW_PASS_FILTER_PADDING] = {0};
        size_t previousIndex = 0;

#pragma unroll
        for (int i = -LOW_PASS_FILTER_PADDING; i < LOW_PASS_FILTER_PADDING; i++) {
            cart::disparity_t value = sharedDisparity[LOCAL_INDEX(j, i)];
            if (value != CARTSLAM_DISPARITY_INVALID) {
                sum += value;
                count++;
            }

            if (i < 0) {
                previous[previousIndex] = value;
                previousIndex++;
            }
        }

        previousIndex = 0;

        for (int i = 0; i < Y_BATCH; i++) {
            cart::disparity_t value = sharedDisparity[LOCAL_INDEX(j, i + LOW_PASS_FILTER_PADDING)];
            if (value != CARTSLAM_DISPARITY_INVALID) {
                sum += value;
                count++;
            }

            cart::disparity_t current = sharedDisparity[LOCAL_INDEX(j, i)];

            sharedDisparity[LOCAL_INDEX(j, i)] = sum / count;

            value = previous[previousIndex];
            if (value != CARTSLAM_DISPARITY_INVALID) {
                sum -= value;
                count--;
            }

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

            bool valid = sharedDisparity[LOCAL_INDEX(j, i)] != CARTSLAM_DISPARITY_INVALID &&
                         sharedDisparity[LOCAL_INDEX(j, i + 1)] != CARTSLAM_DISPARITY_INVALID &&
                         sharedDisparity[LOCAL_INDEX(j, i - 1)] != CARTSLAM_DISPARITY_INVALID;
            output[INDEX(pixelX + j, pixelY + i, outputRowStep)] = valid ? derivative : CARTSLAM_DISPARITY_DERIVATIVE_INVALID;

            // Only update histogram if the value is within the range of a signed char
            if (derivative >= -128 && derivative <= 127 && valid) {
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

struct PreviousPlaneAssignment {
    uint8_t* data;
    size_t step;
};

__global__ void classifyPlanes(cv::cuda::PtrStepSz<derivative_t> derivatives, cv::cuda::PtrStepSz<uint8_t> planes, int width, int height,
                               cart::PlaneParameters params, cv::cuda::PtrStepSz<uint8_t> smoothedPlanes, PreviousPlaneAssignment* previousPlanes, int previousPlanesCount) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int pixelX = x * X_BATCH;
    int pixelY = y * Y_BATCH;

    size_t derivativesRowStep = derivatives.step / sizeof(derivative_t);
    size_t planesRowStep = planes.step / sizeof(uint8_t);

    int horizontalStart = params.horizontalRange.first;
    int horizontalEnd = params.horizontalRange.second;

    int verticalStart = params.verticalRange.first;
    int verticalEnd = params.verticalRange.second;

    for (int i = 0; i < Y_BATCH; i++) {
        for (int j = 0; j < X_BATCH; j++) {
            if (pixelX + j >= width || pixelY + i >= height) {
                continue;
            }

            derivative_t derivative = derivatives[INDEX(pixelX + j, pixelY + i, derivativesRowStep)];
            int plane = cart::Plane::UNKNOWN;

            if (derivative != CARTSLAM_DISPARITY_DERIVATIVE_INVALID && derivative >= horizontalStart && derivative < horizontalEnd) {
                plane = cart::Plane::HORIZONTAL;
            } else if (derivative != CARTSLAM_DISPARITY_DERIVATIVE_INVALID && derivative >= verticalStart && derivative < verticalEnd) {
                plane = cart::Plane::VERTICAL;
            }

            planes[INDEX(pixelX + j, pixelY + i, planesRowStep)] = plane;

            // Very naive temporal smoothing, assuming no movement in the image which is usually not the case
            // TODO: Integrate optical flow
            if (previousPlanesCount <= 0) {
                continue;
            }

            int votes[CARTSLAM_PLANE_COUNT] = {0};
            votes[plane]++;

            for (int k = 0; k < previousPlanesCount; k++) {
                PreviousPlaneAssignment previousPlane = previousPlanes[k];
                int previousPlaneValue = previousPlane.data[INDEX(pixelX + j, pixelY + i, previousPlane.step / sizeof(uint8_t))];
                votes[previousPlaneValue]++;
            }

            int max = cart::Plane::UNKNOWN;
            if (votes[cart::Plane::HORIZONTAL] > votes[max]) {
                max = cart::Plane::HORIZONTAL;
            }

            if (votes[cart::Plane::VERTICAL] > votes[max]) {
                max = cart::Plane::VERTICAL;
            }

            smoothedPlanes[INDEX(pixelX + j, pixelY + i, planesRowStep)] = max;
        }
    }
}

#define COLOR(plane) cart::PlaneColor<cart::Plane::plane>()

struct plane_colors_t {
    const int colors[3][3] = {
        {COLOR(HORIZONTAL).b / 2, COLOR(HORIZONTAL).g / 2, COLOR(HORIZONTAL).r / 2},
        {COLOR(VERTICAL).b / 2, COLOR(VERTICAL).g / 2, COLOR(VERTICAL).r / 2},
        {COLOR(UNKNOWN).b / 2, COLOR(UNKNOWN).g / 2, COLOR(UNKNOWN).r / 2}};
} planeColors;

__global__ void overlayPlanes(cv::cuda::PtrStepSz<uint8_t> image, cv::cuda::PtrStepSz<uint8_t> planes, cv::cuda::PtrStepSz<uint8_t> output, int width, int height, plane_colors_t colors) {
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

            output[INDEX_BGR(pixelX + j, pixelY + i, 0, outputRowStep)] = b / 2 + colors.colors[plane][0];
            output[INDEX_BGR(pixelX + j, pixelY + i, 1, outputRowStep)] = g / 2 + colors.colors[plane][1];
            output[INDEX_BGR(pixelX + j, pixelY + i, 2, outputRowStep)] = r / 2 + colors.colors[plane][2];
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
    this->updatePlaneParameters(system, data);

    cv::cuda::GpuMat planes(disparity->size(), CV_8UC1);
    cv::cuda::GpuMat smoothed;

    int previousPlaneCount = 0;
    PreviousPlaneAssignment* devicePrevPlanes = nullptr;

    if (this->useTemporalSmoothing && data.id > 1) {
        smoothed.create(disparity->size(), CV_8UC1);

        PreviousPlaneAssignment previousPlanesHost[CARTSLAM_PLANE_TEMPORAL_DISTANCE];

        // Copy previous planes to constant memory
        for (int i = 1; i <= CARTSLAM_PLANE_TEMPORAL_DISTANCE; i++) {
            if (data.id - i <= 0) {
                break;
            }

            auto relativeRun = data.getRelativeRun(-i);
            LOG4CXX_DEBUG(this->logger, "Getting previous planes from " << relativeRun->id << " (id: " << data.id << ", i: " << i << ")");
            
            // Block until available
            auto prev = *relativeRun->getDataAsync<cv::cuda::GpuMat>(CARTSLAM_KEY_PLANES_UNSMOOTHED).get();
            previousPlanesHost[previousPlaneCount] = {
                static_cast<uint8_t*>(prev.cudaPtr()),
                prev.step,
            };

            previousPlaneCount++;
        }

        CUDA_SAFE_CALL(this->logger, cudaMalloc(&devicePrevPlanes, previousPlaneCount * sizeof(PreviousPlaneAssignment)));
        CUDA_SAFE_CALL(this->logger, cudaMemcpy(devicePrevPlanes, previousPlanesHost, previousPlaneCount * sizeof(PreviousPlaneAssignment), cudaMemcpyHostToDevice));
    }

    {
        cudaStream_t stream;
        cudaStreamCreate(&stream);

        PlaneParameters params = this->planeParameterProvider->getPlaneParameters();
        classifyPlanes<<<numBlocks, threadsPerBlock, 0, stream>>>(derivatives, planes, disparity->cols, disparity->rows, params, smoothed, devicePrevPlanes, previousPlaneCount);

        CUDA_SAFE_CALL(this->logger, cudaPeekAtLastError());
        CUDA_SAFE_CALL(this->logger, cudaStreamSynchronize(stream));
    }

    if (devicePrevPlanes != nullptr) {
        CUDA_SAFE_CALL(this->logger, cudaFree(devicePrevPlanes));
    }

    if (this->useTemporalSmoothing) {
        return MODULE_RETURN_ALL(
            MODULE_MAKE_PAIR(CARTSLAM_KEY_PLANES, cv::cuda::GpuMat, boost::move(smoothed)),
            MODULE_MAKE_PAIR(CARTSLAM_KEY_PLANES_UNSMOOTHED, cv::cuda::GpuMat, boost::move(planes)));
    }

    return MODULE_RETURN_SHARED(CARTSLAM_KEY_PLANES, cv::cuda::GpuMat, boost::move(planes));
}

void DisparityPlaneSegmentationModule::updatePlaneParameters(System& system, SystemRunData& data) {
    // TODO: Inspiration for alternative: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7926705&tag=1
    if (data.id % this->updateInterval != 1) {
        return;
    }

    cv::Mat histogram;

    {
        boost::unique_lock<boost::shared_mutex> lock(this->derivativeHistogramMutex);
        this->derivativeHistogram.download(histogram);

        if (data.id % (this->updateInterval * this->resetInterval) == 1) {
            // Reset to avoid overflow
            this->derivativeHistogram.setTo(0);
        }
    }

    this->planeParameterProvider->updatePlaneParameters(this->logger, system, data, histogram);

    system.insertGlobalData(CARTSLAM_KEY_PLANE_PARAMETERS, boost::make_shared<PlaneParameters>(this->planeParameterProvider->getPlaneParameters()));

    auto histShared = boost::make_shared<cv::Mat>(boost::move(histogram));
    system.insertGlobalData(CARTSLAM_KEY_DISPARITY_DERIVATIVE_HIST, histShared);
}

void HistogramPeakPlaneParameterProvider::updatePlaneParameters(log4cxx::LoggerPtr logger, System& system, SystemRunData& data, cv::Mat& histogram) {
    std::vector<util::Peak> peaks = util::findPeaks(histogram);

    if (peaks.size() < 2) {
        LOG4CXX_DEBUG(logger, "Not enough peaks found");
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

    LOG4CXX_DEBUG(logger, "Peaks: " << peaks[0].born << ", " << peaks[1].born << ", Min: " << minIndex);

    // Set variance as distance from peak to minimum
    int verticalMinDist = abs(minIndex - peaks[0].born);
    int horizontalMinDist = abs(minIndex - peaks[1].born);

    int verticalDerivative = (histogram.at<int>(peaks[0].born) - histogram.at<int>(minIndex)) / verticalMinDist;
    int horizontalDerivative = (histogram.at<int>(peaks[1].born) - histogram.at<int>(minIndex)) / horizontalMinDist;

    int verticalWidth = histogram.at<int>(peaks[0].born) / verticalDerivative;
    int horizontalWidth = histogram.at<int>(peaks[1].born) / horizontalDerivative;

    this->verticalRange = std::make_pair(peaks[0].born - verticalWidth - 128, minIndex - 127);
    this->horizontalRange = std::make_pair(minIndex - 127, peaks[1].born + horizontalWidth - 127);

    LOG4CXX_DEBUG(logger, "Vertical center: " << this->verticalCenter << ", Horizontal center: " << this->horizontalCenter);
    LOG4CXX_DEBUG(logger, "Vertical range: " << this->verticalRange.first << " - " << this->verticalRange.second);
    LOG4CXX_DEBUG(logger, "Horizontal range: " << this->horizontalRange.first << " - " << this->horizontalRange.second);
}

boost::future<system_data_t> DisparityPlaneSegmentationVisualizationModule::run(System& system, SystemRunData& data) {
    auto promise = boost::make_shared<boost::promise<system_data_t>>();

    boost::asio::post(system.getThreadPool(), [this, promise, &system, &data]() {
        auto planes = data.getData<cv::cuda::GpuMat>(CARTSLAM_KEY_PLANES);

        if (!planes->empty()) {
            // Show image
            boost::shared_ptr<cart::StereoDataElement> stereoData = boost::static_pointer_cast<cart::StereoDataElement>(data.dataElement);

            dim3 threadsPerBlock(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y);
            dim3 numBlocks((planes->cols + (threadsPerBlock.x * X_BATCH - 1)) / (threadsPerBlock.x * X_BATCH),
                           (planes->rows + (threadsPerBlock.y * Y_BATCH - 1)) / (threadsPerBlock.y * Y_BATCH));

            cv::cuda::GpuMat output(planes->size(), CV_8UC3);
            cv::Mat image;

            cv::cuda::Stream cvStream;
            cudaStream_t stream = cv::cuda::StreamAccessor::getStream(cvStream);

            overlayPlanes<<<numBlocks, threadsPerBlock, 0, stream>>>(stereoData->left, *planes, output, planes->cols, planes->rows, planeColors);
            CUDA_SAFE_CALL(this->logger, cudaPeekAtLastError());

            output.download(image, cvStream);
            CUDA_SAFE_CALL(this->logger, cudaStreamSynchronize(stream));

            this->imageThread->setImageIfLater(image, data.id);
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
}  // namespace cart