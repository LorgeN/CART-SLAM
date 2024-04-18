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
typedef int16_t optical_flow_t;

// TODO: Change disparity values to float?
__global__ void calculateDerivatives(cv::cuda::PtrStepSz<cart::disparity_t> disparity,
                                     cv::cuda::PtrStepSz<derivative_t> output,
                                     cv::cuda::PtrStepSz<int> histogramOutput, int width, int height) {
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

__global__ void classifyPlanes(cv::cuda::PtrStepSz<derivative_t> derivatives,
                               cv::cuda::PtrStepSz<uint8_t> planes,
                               cart::PlaneParameters params,
                               cv::cuda::PtrStepSz<uint8_t> smoothedPlanes,
                               cv_mat_ptr_t<uint8_t>* previousPlanes,
                               cv_mat_ptr_t<optical_flow_t>* previousOpticalFlow,
                               int previousPlanesCount) {
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
            if (pixelX + j >= derivatives.cols || pixelY + i >= derivatives.rows) {
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

            if (previousPlanesCount <= 0) {
                continue;
            }

            int votes[CARTSLAM_PLANE_COUNT] = {0};
            votes[plane]++;

            for (int k = 0; k < previousPlanesCount; k++) {
                cv_mat_ptr_t previosOptFlow = previousOpticalFlow[k];
                optical_flow_t flowX = previosOptFlow.data[INDEX_CH(pixelX + j, pixelY + i, 2, 0, previosOptFlow.step / sizeof(optical_flow_t))];
                optical_flow_t flowY = previosOptFlow.data[INDEX_CH(pixelX + j, pixelY + i, 2, 1, previosOptFlow.step / sizeof(optical_flow_t))];

                // Flow values are in S10.5 format. We are only interested in whole integer values
                flowX >>= 5;
                flowY >>= 5;

                int x = pixelX + j - flowX;
                int y = pixelY + i - flowY;
                // It is possible for these values to go beyond the image bounds, so we need to check
                if (x < 0 || y < 0 || x >= derivatives.cols || y >= derivatives.rows) {
                    continue;
                }

                cv_mat_ptr_t previousPlane = previousPlanes[k];
                int previousPlaneValue = previousPlane.data[INDEX(x, y, previousPlane.step / sizeof(uint8_t))];
                votes[previousPlaneValue]++;
            }

            int max = votes[cart::Plane::HORIZONTAL] > votes[cart::Plane::VERTICAL] ? cart::Plane::HORIZONTAL : cart::Plane::VERTICAL;
            if (votes[max] == 0) {
                max = cart::Plane::UNKNOWN;
            }

            smoothedPlanes[INDEX(pixelX + j, pixelY + i, planesRowStep)] = max;
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
    cv_mat_ptr_t<uint8_t>* devicePrevPlanes = nullptr;
    cv_mat_ptr_t<optical_flow_t>* devicePrevOpticalFlow = nullptr;

    if (this->useTemporalSmoothing && data.id > 1) {
        LOG4CXX_DEBUG(this->logger, "Using temporal smoothing");
        smoothed.create(disparity->size(), CV_8UC1);

        cv_mat_ptr_t<uint8_t> previousPlanesHost[CARTSLAM_PLANE_TEMPORAL_DISTANCE];

        auto optFlowCurr = data.getData<image_optical_flow_t>(CARTSLAM_KEY_OPTFLOW);

        cv_mat_ptr_t<optical_flow_t> previousOpticalFlowHost[CARTSLAM_PLANE_TEMPORAL_DISTANCE] = {{static_cast<optical_flow_t*>(optFlowCurr->flow.cudaPtr()),
                                                                                                   optFlowCurr->flow.step}};

        // Copy previous planes to constant memory
        for (int i = 1; i <= CARTSLAM_PLANE_TEMPORAL_DISTANCE; i++) {
            if (data.id - i <= 0) {
                break;
            }

            LOG4CXX_DEBUG(this->logger, "Getting previous planes from relative run " << (-i) << " (id: " << data.id << ")");
            auto relativeRun = data.getRelativeRun(-i);
            LOG4CXX_DEBUG(this->logger, "Getting previous planes from " << relativeRun->id << " (id: " << data.id << ", i: " << i << ")");

            // Block until available
            auto prev = relativeRun->getDataAsync<cv::cuda::GpuMat>(CARTSLAM_KEY_PLANES_UNSMOOTHED).get();
            previousPlanesHost[previousPlaneCount] = {
                static_cast<uint8_t*>(prev->cudaPtr()),
                prev->step,
            };

            previousPlaneCount++;

            if (relativeRun->id > 1 && previousPlaneCount < CARTSLAM_PLANE_TEMPORAL_DISTANCE) {
                LOG4CXX_DEBUG(this->logger, "Getting optical flow from " << relativeRun->id);
                auto optFlow = relativeRun->getDataAsync<image_optical_flow_t>(CARTSLAM_KEY_OPTFLOW).get();
                previousOpticalFlowHost[previousPlaneCount] = {
                    static_cast<optical_flow_t*>(optFlow->flow.cudaPtr()),
                    optFlow->flow.step};
            }
        }

        CUDA_SAFE_CALL(this->logger, cudaMalloc(&devicePrevPlanes, previousPlaneCount * sizeof(cv_mat_ptr_t<uint8_t>)));
        CUDA_SAFE_CALL(this->logger, cudaMemcpy(devicePrevPlanes, previousPlanesHost, previousPlaneCount * sizeof(cv_mat_ptr_t<uint8_t>), cudaMemcpyHostToDevice));

        CUDA_SAFE_CALL(this->logger, cudaMalloc(&devicePrevOpticalFlow, previousPlaneCount * sizeof(cv_mat_ptr_t<optical_flow_t>)));
        CUDA_SAFE_CALL(this->logger, cudaMemcpy(devicePrevOpticalFlow, previousOpticalFlowHost, previousPlaneCount * sizeof(cv_mat_ptr_t<optical_flow_t>), cudaMemcpyHostToDevice));
    }

    {
        cudaStream_t stream;
        cudaStreamCreate(&stream);

        PlaneParameters params = this->planeParameterProvider->getPlaneParameters();
        classifyPlanes<<<numBlocks, threadsPerBlock, 0, stream>>>(derivatives, planes, params, smoothed, devicePrevPlanes, devicePrevOpticalFlow, previousPlaneCount);

        CUDA_SAFE_CALL(this->logger, cudaPeekAtLastError());
        CUDA_SAFE_CALL(this->logger, cudaStreamSynchronize(stream));
    }

    if (this->useTemporalSmoothing) {
        if (data.id > 1) {
            CUDA_SAFE_CALL(this->logger, cudaFree(devicePrevPlanes));
            CUDA_SAFE_CALL(this->logger, cudaFree(devicePrevOpticalFlow));
        } else {
            // Return both as the same, since this is the first run
            LOG4CXX_DEBUG(this->logger, "Returning planes as both smoothed and unsmoothed");
            auto ptr = boost::make_shared<cv::cuda::GpuMat>(boost::move(planes));
            return MODULE_RETURN_ALL(
                std::make_pair(CARTSLAM_KEY_PLANES, ptr),
                std::make_pair(CARTSLAM_KEY_PLANES_UNSMOOTHED, ptr));
        }

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
}  // namespace cart