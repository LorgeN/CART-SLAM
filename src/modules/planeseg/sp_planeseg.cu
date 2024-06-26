#include <cooperative_groups.h>

#include <opencv2/core/cuda_stream_accessor.hpp>

#include "cartslam.hpp"
#include "modules/disparity.hpp"
#include "modules/planeseg.hpp"
#include "modules/superpixels/contourrelaxation/contourrelaxation.hpp"
#include "utils/cuda.cuh"
#include "utils/modules.hpp"

#define MAX_SHARED_MEMORY 32768
#define CARTSLAM_DISPARITY_DERIVATIVE_INVALID (-32768)

#define THREADS_PER_BLOCK_X 32
#define THREADS_PER_BLOCK_Y 32
#define X_BATCH 4
#define Y_BATCH 4
#define SHARED_SIZE ((X_BATCH * THREADS_PER_BLOCK_X) * (Y_BATCH * (LOW_PASS_FILTER_PADDING * 2 + THREADS_PER_BLOCK_Y)))

#define LABEL_INDEX(label, plane) ((label) * 3 + (plane))

namespace cg = cooperative_groups;

__global__ void performSuperPixelClassifications(const cv::cuda::PtrStepSz<cart::derivative_t> derivatives,
                                                 const cv::cuda::PtrStepSz<cart::contour::label_t> labels,
                                                 cv::cuda::PtrStepSz<uint8_t> planes,
                                                 const cart::PlaneParameters params,
                                                 cv::cuda::PtrStepSz<uint16_t> globalLabelData,
                                                 const cart::cv_mat_ptr_t<uint8_t>* previousPlanes,
                                                 const cart::cv_mat_ptr_t<cart::optical_flow_t>* previousOpticalFlow,
                                                 const int previousPlanesCount,
                                                 const int maxLabel) {
    extern __shared__ uint16_t sharedLabelData[];  // For each label; Track vertical, horizontal and unknown votes. Total these for total pixels

    for (int i = threadIdx.x + (blockDim.x * threadIdx.y); i < maxLabel; i += blockDim.x * blockDim.y) {
#pragma unroll
        for (int plane = 0; plane < 3; plane++) {
            sharedLabelData[LABEL_INDEX(i, plane)] = 0;
        }
    }

    cg::thread_block tb = cg::this_thread_block();

    tb.sync();

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int pixelX = x * X_BATCH;
    int pixelY = y * Y_BATCH;

    size_t derivativesRowStep = derivatives.step / sizeof(cart::derivative_t);
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

            cart::derivative_t derivative = derivatives[INDEX_CH(pixelX + j, pixelY + i, 2, 0, derivativesRowStep)];
            int plane = cart::Plane::UNKNOWN;

            if (derivative != CARTSLAM_DISPARITY_DERIVATIVE_INVALID && derivative >= horizontalStart && derivative < horizontalEnd) {
                plane = cart::Plane::HORIZONTAL;
            } else if (derivative != CARTSLAM_DISPARITY_DERIVATIVE_INVALID && derivative >= verticalStart && derivative < verticalEnd) {
                plane = cart::Plane::VERTICAL;
            }

            planes[INDEX(pixelX + j, pixelY + i, planesRowStep)] = plane;

            // Will not diverge since this will be the same across all threads
            if (previousPlanesCount > 0) {
                int votes[CARTSLAM_PLANE_COUNT] = {0};
                votes[plane] += 2;  // Assign more weight to current value

                // Need to be signed ints because we can go negative
                int x = pixelX + j;
                int y = pixelY + i;

                for (int k = 0; k < previousPlanesCount; k++) {
                    cart::cv_mat_ptr_t previosOptFlow = previousOpticalFlow[k];
                    cart::optical_flow_t flowX = previosOptFlow.data[INDEX_CH(pixelX + j, pixelY + i, 2, 0, previosOptFlow.step / sizeof(cart::optical_flow_t))];
                    cart::optical_flow_t flowY = previosOptFlow.data[INDEX_CH(pixelX + j, pixelY + i, 2, 1, previosOptFlow.step / sizeof(cart::optical_flow_t))];

                    // Flow values are in S10.5 format. We are only interested in whole integer values
                    flowX >>= 5;
                    flowY >>= 5;

                    // We subtract the flow values to get the previous pixel position. For each opt flow matrix, these reference the previous frame,
                    // so we need to subtract all the flow values between current and the target frame to get the right position
                    x -= flowX;
                    y -= flowY;

                    // It is possible for these values to go beyond the image bounds, so we need to check
                    if (x < 0 || y < 0 || x >= derivatives.cols || y >= derivatives.rows) {
                        continue;
                    }

                    cart::cv_mat_ptr_t previousPlane = previousPlanes[k];
                    int previousPlaneValue = previousPlane.data[INDEX(x, y, previousPlane.step / sizeof(uint8_t))];
                    votes[previousPlaneValue]++;
                }

                // Find the plane with the most votes, or unknown if there are no votes
                plane = votes[cart::Plane::HORIZONTAL] > votes[cart::Plane::VERTICAL] ? cart::Plane::HORIZONTAL : cart::Plane::VERTICAL;
                if (votes[plane] < votes[cart::Plane::UNKNOWN]) {
                    plane = cart::Plane::UNKNOWN;
                }
            }

            cart::contour::label_t label = labels[INDEX(pixelX + j, pixelY + i, labels.step / sizeof(cart::contour::label_t))];
            unsafeAtomicAdd(&sharedLabelData[LABEL_INDEX(label, plane)], 1);
        }
    }

    tb.sync();

    size_t globalStep = globalLabelData.step / sizeof(uint16_t);

    for (int i = threadIdx.x + (blockDim.x * threadIdx.y); i < maxLabel; i += blockDim.x * blockDim.y) {
#pragma unroll
        for (int plane = 0; plane < 3; plane++) {
            unsafeAtomicAdd(&globalLabelData[INDEX(plane, i, globalStep)], sharedLabelData[LABEL_INDEX(i, plane)]);
        }
    }
}

__global__ void classifyPlanes(const cv::cuda::PtrStepSz<cart::contour::label_t> labels,
                               const cv::cuda::PtrStepSz<uint16_t> globalLabelData,
                               cv::cuda::PtrStepSz<uint8_t> resultingPlanes,
                               int maxLabel) {
    extern __shared__ cart::Plane sharedPlaneAssignments[];  // For each label; Track vertical, horizontal and unknown votes. Total these for total pixels

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int pixelX = x * X_BATCH;
    int pixelY = y * Y_BATCH;

    size_t planesRowStep = resultingPlanes.step / sizeof(uint8_t);
    size_t globalStep = globalLabelData.step / sizeof(uint16_t);

    for (int label = threadIdx.x + (blockDim.x * threadIdx.y); label < maxLabel; label += blockDim.x * blockDim.y) {
        int maxVotes = globalLabelData[INDEX(cart::Plane::UNKNOWN, label, globalStep)];
        cart::Plane max = cart::Plane::UNKNOWN;

        int verticalVotes = globalLabelData[INDEX(cart::Plane::VERTICAL, label, globalStep)];
        int horizontalVotes = globalLabelData[INDEX(cart::Plane::HORIZONTAL, label, globalStep)];

        if (verticalVotes > maxVotes) {
            maxVotes = verticalVotes;
            max = cart::Plane::VERTICAL;
        }

        if (horizontalVotes > maxVotes) {
            max = cart::Plane::HORIZONTAL;
        }

        sharedPlaneAssignments[label] = max;
    }

    cg::thread_block tb = cg::this_thread_block();
    tb.sync();

    for (int i = 0; i < Y_BATCH; i++) {
        for (int j = 0; j < X_BATCH; j++) {
            if (pixelX + j >= labels.cols || pixelY + i >= labels.rows) {
                continue;
            }

            cart::contour::label_t label = labels[INDEX(pixelX + j, pixelY + i, labels.step / sizeof(cart::contour::label_t))];
            int max = sharedPlaneAssignments[label];
            resultingPlanes[INDEX(pixelX + j, pixelY + i, planesRowStep)] = max;
        }
    }
}

namespace cart {

SuperPixelDisparityPlaneSegmentationModule::SuperPixelDisparityPlaneSegmentationModule(
    boost::shared_ptr<PlaneParameterProvider> planeParameterProvider,
    const int updateInterval, const int resetInterval, const bool useTemporalSmoothing, const unsigned int temporalSmoothingDistance)
    // Need optical flow for temporal smoothing
    : SyncWrapperSystemModule("SPPlaneSegmentation"),
      planeParameterProvider(planeParameterProvider),
      updateInterval(updateInterval),
      useTemporalSmoothing(useTemporalSmoothing),
      resetInterval(resetInterval),
      temporalSmoothingDistance(temporalSmoothingDistance) {
    this->requiresData.push_back(module_dependency_t(CARTSLAM_KEY_SUPERPIXELS));
    this->requiresData.push_back(module_dependency_t(CARTSLAM_KEY_SUPERPIXELS_MAX_LABEL));
    this->requiresData.push_back(module_dependency_t(CARTSLAM_KEY_DISPARITY_DERIVATIVE));
    this->requiresData.push_back(module_dependency_t(CARTSLAM_KEY_DISPARITY_DERIVATIVE_HISTOGRAM));

    if (useTemporalSmoothing) {
        this->requiresData.push_back(module_dependency_t(CARTSLAM_KEY_OPTFLOW));
        for (size_t i = 1; i <= this->temporalSmoothingDistance; i++) {
            this->requiresData.push_back(module_dependency_t(CARTSLAM_KEY_PLANES_UNSMOOTHED, -i));

            if ((i + 1) <= this->temporalSmoothingDistance) {
                this->requiresData.push_back(module_dependency_t(CARTSLAM_KEY_OPTFLOW, -i));
            }
        }
    }

    this->providesData.push_back(CARTSLAM_KEY_PLANES);

    if (useTemporalSmoothing) {
        this->providesData.push_back(CARTSLAM_KEY_PLANES_UNSMOOTHED);
    }

    CUDA_SAFE_CALL(this->logger, cudaFuncSetAttribute(performSuperPixelClassifications, cudaFuncAttributeMaxDynamicSharedMemorySize, MAX_SHARED_MEMORY));
    CUDA_SAFE_CALL(this->logger, cudaFuncSetAttribute(classifyPlanes, cudaFuncAttributeMaxDynamicSharedMemorySize, MAX_SHARED_MEMORY));
};

system_data_t SuperPixelDisparityPlaneSegmentationModule::runInternal(System& system, SystemRunData& data) {
    auto derivatives = data.getData<cv::cuda::GpuMat>(CARTSLAM_KEY_DISPARITY_DERIVATIVE);

    if (derivatives->empty()) {
        LOG4CXX_WARN(this->logger, "Disparity derivatives are empty");
        return MODULE_NO_RETURN_VALUE;
    }

    if (derivatives->type() != CV_16SC2) {
        LOG4CXX_WARN(this->logger, "Disparity derivatives must be of type CV_16SC2, was " << derivatives->type() << " (Depth: " << derivatives->depth() << ", channels: " << derivatives->channels() << ")");
        throw std::runtime_error("Disparity must be of type CV_16SC2");
    }

    dim3 threadsPerBlock(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y);
    dim3 numBlocks((derivatives->cols + (threadsPerBlock.x * X_BATCH - 1)) / (threadsPerBlock.x * X_BATCH),
                   (derivatives->rows + (threadsPerBlock.y * Y_BATCH - 1)) / (threadsPerBlock.y * Y_BATCH));

    this->updatePlaneParameters(system, data);
    LOG4CXX_DEBUG(this->logger, "Updated plane parameters");

    cv::cuda::GpuMat planes(derivatives->size(), CV_8UC1);
    cv::cuda::GpuMat smoothed(derivatives->size(), CV_8UC1);

    int previousPlaneCount = 0;
    cv_mat_ptr_t<uint8_t>* devicePrevPlanes = nullptr;
    cv_mat_ptr_t<optical_flow_t>* devicePrevOpticalFlow = nullptr;

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // TODO: Make this a function to avoid code duplication
    if (this->useTemporalSmoothing && data.id > 1) {
        LOG4CXX_DEBUG(this->logger, "Using temporal smoothing");

        cv_mat_ptr_t<uint8_t> previousPlanesHost[this->temporalSmoothingDistance];

        auto optFlowCurr = data.getData<cv::cuda::GpuMat>(CARTSLAM_KEY_OPTFLOW);

        cv_mat_ptr_t<optical_flow_t> previousOpticalFlowHost[this->temporalSmoothingDistance] = {{static_cast<optical_flow_t*>(optFlowCurr->cudaPtr()),
                                                                                                  optFlowCurr->step}};

        // Copy previous planes to constant memory
        for (int i = 1; i <= this->temporalSmoothingDistance; i++) {
            if (data.id - i <= 0) {
                break;
            }

            LOG4CXX_DEBUG(this->logger, "Getting previous planes from relative run " << (-i) << " (id: " << data.id << ")");
            auto relativeRun = data.getRelativeRun(-i);
            LOG4CXX_DEBUG(this->logger, "Getting previous planes from " << relativeRun->id << " (id: " << data.id << ", i: " << i << ")");

            // Block until available
            boost::shared_ptr<cv::cuda::GpuMat> prev;
            try {
                prev = relativeRun->getData<cv::cuda::GpuMat>(CARTSLAM_KEY_PLANES_UNSMOOTHED);
            } catch (const std::exception& e) {
                LOG4CXX_ERROR(this->logger, "Could not get previous planes from " << relativeRun->id << ": " << e.what());
                break;
            }

            LOG4CXX_DEBUG(this->logger, "Got previous planes from " << relativeRun->id << " (id: " << data.id << ", i: " << i << ")");

            previousPlanesHost[previousPlaneCount] = {
                static_cast<uint8_t*>(prev->cudaPtr()),
                prev->step,
            };

            previousPlaneCount++;

            if (relativeRun->id > 1 && previousPlaneCount < this->temporalSmoothingDistance) {
                LOG4CXX_DEBUG(this->logger, "Getting optical flow from " << relativeRun->id);

                boost::shared_ptr<cv::cuda::GpuMat> optFlow;

                try {
                    optFlow = relativeRun->getData<cv::cuda::GpuMat>(CARTSLAM_KEY_OPTFLOW);
                } catch (const std::exception& e) {
                    LOG4CXX_ERROR(this->logger, "Could not get optical flow from " << relativeRun->id << ": " << e.what());
                    break;
                }

                previousOpticalFlowHost[previousPlaneCount] = {
                    static_cast<optical_flow_t*>(optFlow->cudaPtr()),
                    optFlow->step};
            }
        }

        CUDA_SAFE_CALL(this->logger, cudaMallocAsync(&devicePrevPlanes, previousPlaneCount * sizeof(cv_mat_ptr_t<uint8_t>), stream));
        CUDA_SAFE_CALL(this->logger, cudaMemcpyAsync(devicePrevPlanes, previousPlanesHost, previousPlaneCount * sizeof(cv_mat_ptr_t<uint8_t>), cudaMemcpyHostToDevice, stream));

        CUDA_SAFE_CALL(this->logger, cudaMallocAsync(&devicePrevOpticalFlow, previousPlaneCount * sizeof(cv_mat_ptr_t<optical_flow_t>), stream));
        CUDA_SAFE_CALL(this->logger, cudaMemcpyAsync(devicePrevOpticalFlow, previousOpticalFlowHost, previousPlaneCount * sizeof(cv_mat_ptr_t<optical_flow_t>), cudaMemcpyHostToDevice, stream));
    }

    auto labels = data.getData<cv::cuda::GpuMat>(CARTSLAM_KEY_SUPERPIXELS);
    auto maxLabel = *data.getData<contour::label_t>(CARTSLAM_KEY_SUPERPIXELS_MAX_LABEL);

    cv::cuda::Stream cvStream = cv::cuda::StreamAccessor::wrapStream(stream);

    cv::cuda::GpuMat globalLabelData(maxLabel + 1, 3, CV_16UC1);
    globalLabelData.setTo(0, cvStream);
    PlaneParameters params = this->planeParameterProvider->getPlaneParameters();

    size_t sharedSize = (maxLabel + 1) * 3 * sizeof(uint16_t);
    if (sharedSize > MAX_SHARED_MEMORY) {
        LOG4CXX_WARN(this->logger, "Shared memory size " << sharedSize << " exceeds maximum " << MAX_SHARED_MEMORY);
        throw std::runtime_error("Shared memory size exceeds maximum. Reduce image size or increase block size.");
    }

    size_t sharedSizeClassify = (maxLabel + 1) * sizeof(cart::Plane);

    performSuperPixelClassifications<<<numBlocks, threadsPerBlock, sharedSize, stream>>>(*derivatives, *labels, planes, params, globalLabelData, devicePrevPlanes, devicePrevOpticalFlow, previousPlaneCount, maxLabel);
    classifyPlanes<<<numBlocks, threadsPerBlock, sharedSizeClassify, stream>>>(*labels, globalLabelData, smoothed, maxLabel);

    if (data.id > 1 && this->useTemporalSmoothing) {
        CUDA_SAFE_CALL(this->logger, cudaFreeAsync(devicePrevPlanes, stream));
        CUDA_SAFE_CALL(this->logger, cudaFreeAsync(devicePrevOpticalFlow, stream));
    }

    CUDA_SAFE_CALL(this->logger, cudaGetLastError());
    CUDA_SAFE_CALL(this->logger, cudaStreamSynchronize(stream));
    CUDA_SAFE_CALL(this->logger, cudaStreamDestroy(stream));

    return MODULE_RETURN_ALL(
        MODULE_MAKE_PAIR(CARTSLAM_KEY_PLANES, cv::cuda::GpuMat, boost::move(smoothed)),
        MODULE_MAKE_PAIR(CARTSLAM_KEY_PLANES_UNSMOOTHED, cv::cuda::GpuMat, boost::move(planes)));
}

void SuperPixelDisparityPlaneSegmentationModule::updatePlaneParameters(System& system, SystemRunData& data) {
    cv::Mat channels[2];

    cv::Mat histogram;
    data.getData<cv::cuda::GpuMat>(CARTSLAM_KEY_DISPARITY_DERIVATIVE_HISTOGRAM)->download(histogram);

    cv::split(histogram, channels);
    histogram = channels[0];

    {
        boost::unique_lock<boost::shared_mutex> lock(this->derivativeHistogramMutex);

        if (this->derivativeHistogram.empty()) {
            this->derivativeHistogram = cv::Mat::zeros(histogram.size(), histogram.type());
        } else {
            // Add the values from histogram to the running total
            this->derivativeHistogram += histogram;
            histogram = this->derivativeHistogram.clone();
        }

        if (data.id % (this->updateInterval * this->resetInterval) == 1) {
            // Reset to avoid overflow
            this->derivativeHistogram.setTo(0);
        }
    }

    if (data.id % this->updateInterval != 1) {
        return;
    }

    this->planeParameterProvider->updatePlaneParameters(this->logger, system, data, histogram);

    system.insertGlobalData(CARTSLAM_KEY_PLANE_PARAMETERS, boost::make_shared<PlaneParameters>(this->planeParameterProvider->getPlaneParameters()));

    auto histShared = boost::make_shared<cv::Mat>(boost::move(histogram));
    system.insertGlobalData(CARTSLAM_KEY_DISPARITY_DERIVATIVE_HIST, histShared);
}
}  // namespace cart