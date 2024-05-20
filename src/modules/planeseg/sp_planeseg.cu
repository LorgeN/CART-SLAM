#include <opencv2/core/cuda_stream_accessor.hpp>

#include "modules/disparity.hpp"
#include "modules/planeseg.hpp"
#include "modules/superpixels/contourrelaxation/contourrelaxation.hpp"
#include "utils/cuda.cuh"
#include "utils/modules.hpp"

#define CARTSLAM_DISPARITY_DERIVATIVE_INVALID (-32768)
#define CARTSLAM_PLANE_TEMPORAL_DISTANCE (CARTSLAM_RUN_RETENTION - CARTSLAM_CONCURRENT_RUN_LIMIT)

#define THREADS_PER_BLOCK_X 32
#define THREADS_PER_BLOCK_Y 32
#define X_BATCH 4
#define Y_BATCH 4
#define SHARED_SIZE ((X_BATCH * THREADS_PER_BLOCK_X) * (Y_BATCH * (LOW_PASS_FILTER_PADDING * 2 + THREADS_PER_BLOCK_Y)))

__global__ void performSuperPixelClassifications(cv::cuda::PtrStepSz<cart::derivative_t> derivatives,
                                                 cv::cuda::PtrStepSz<cart::contour::label_t> labels,
                                                 cv::cuda::PtrStepSz<uint8_t> planes,
                                                 cart::PlaneParameters params,
                                                 cv::cuda::PtrStepSz<int> globalLabelData,
                                                 cv::cuda::PtrStepSz<uint8_t> resultingPlanes,
                                                 cart::cv_mat_ptr_t<uint8_t>* previousPlanes,
                                                 cart::cv_mat_ptr_t<cart::optical_flow_t>* previousOpticalFlow,
                                                 int previousPlanesCount,
                                                 int maxLabel) {
    extern __shared__ int sharedLabelData[];  // For each label; Track vertical, horizontal and unknown votes. Total these for total pixels

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

            if (previousPlanesCount <= 0) {
                continue;
            }

            int votes[CARTSLAM_PLANE_COUNT] = {0};
            votes[plane]++;

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
            int max = votes[cart::Plane::HORIZONTAL] > votes[cart::Plane::VERTICAL] ? cart::Plane::HORIZONTAL : cart::Plane::VERTICAL;
            if (votes[max] == 0) {
                max = cart::Plane::UNKNOWN;
            }

            cart::contour::label_t label = labels[INDEX(pixelX + j, pixelY + i, labels.step / sizeof(cart::contour::label_t))];
            atomicAdd(&sharedLabelData[3 * label + max], 1);
        }
    }

    __syncthreads();

    if (threadIdx.x == 0 && threadIdx.y == 0) {
        for (int i = 0; i < maxLabel; i++) {
            atomicAdd(&globalLabelData[INDEX(i, 0, globalLabelData.step / sizeof(int))], sharedLabelData[3 * i]);
            atomicAdd(&globalLabelData[INDEX(i, 1, globalLabelData.step / sizeof(int))], sharedLabelData[3 * i + 1]);
            atomicAdd(&globalLabelData[INDEX(i, 2, globalLabelData.step / sizeof(int))], sharedLabelData[3 * i + 2]);
        }
    }

    __syncthreads();

    for (int i = 0; i < Y_BATCH; i++) {
        for (int j = 0; j < X_BATCH; j++) {
            if (pixelX + j >= derivatives.cols || pixelY + i >= derivatives.rows) {
                continue;
            }

            cart::contour::label_t label = labels[INDEX(pixelX + j, pixelY + i, labels.step / sizeof(cart::contour::label_t))];
            int verticalVotes = globalLabelData[INDEX(label, 0, globalLabelData.step / sizeof(int))];
            int horizontalVotes = globalLabelData[INDEX(label, 1, globalLabelData.step / sizeof(int))];
            int unknownVotes = globalLabelData[INDEX(label, 2, globalLabelData.step / sizeof(int))];

            int max = verticalVotes > horizontalVotes ? cart::Plane::VERTICAL : cart::Plane::HORIZONTAL;
            if (verticalVotes == 0 && horizontalVotes == 0) {
                max = cart::Plane::UNKNOWN;
            }

            resultingPlanes[INDEX(pixelX + j, pixelY + i, planesRowStep)] = max;
        }
    }
}

namespace cart {

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

        CUDA_SAFE_CALL(this->logger, cudaMallocAsync(&devicePrevPlanes, previousPlaneCount * sizeof(cv_mat_ptr_t<uint8_t>), stream));
        CUDA_SAFE_CALL(this->logger, cudaMemcpyAsync(devicePrevPlanes, previousPlanesHost, previousPlaneCount * sizeof(cv_mat_ptr_t<uint8_t>), cudaMemcpyHostToDevice, stream));

        CUDA_SAFE_CALL(this->logger, cudaMallocAsync(&devicePrevOpticalFlow, previousPlaneCount * sizeof(cv_mat_ptr_t<optical_flow_t>), stream));
        CUDA_SAFE_CALL(this->logger, cudaMemcpyAsync(devicePrevOpticalFlow, previousOpticalFlowHost, previousPlaneCount * sizeof(cv_mat_ptr_t<optical_flow_t>), cudaMemcpyHostToDevice, stream));
    }

    auto labels = data.getData<cv::Mat>(CARTSLAM_KEY_SUPERPIXELS);
    double maxLabelDbl;
    cv::minMaxLoc(*labels, nullptr, &maxLabelDbl, nullptr, nullptr);
    cart::contour::label_t maxLabel = static_cast<cart::contour::label_t>(maxLabelDbl);

    cv::cuda::Stream cvStream = cv::cuda::StreamAccessor::wrapStream(stream);

    cv::cuda::GpuMat globalLabelData(3, maxLabel + 1, CV_32SC1);
    globalLabelData.setTo(0, cvStream);
    PlaneParameters params = this->planeParameterProvider->getPlaneParameters();

    cv::cuda::GpuMat labelsGpu;
    labelsGpu.upload(*labels, cvStream);

    size_t sharedSize = (maxLabel + 1) * 3 * sizeof(int);

    /*
    cv::cuda::PtrStepSz<cart::derivative_t> derivatives,
                                                 cv::cuda::PtrStepSz<cart::contour::label_t> labels,
                                                 cv::cuda::PtrStepSz<uint8_t> planes,
                                                 cart::PlaneParameters params,
                                                 cv::cuda::PtrStepSz<int> globalLabelData,
                                                 cv::cuda::PtrStepSz<uint8_t> resultingPlanes,
                                                 cart::cv_mat_ptr_t<uint8_t>* previousPlanes,
                                                 cart::cv_mat_ptr_t<cart::optical_flow_t>* previousOpticalFlow,
                                                 int previousPlanesCount,
                                                 int maxLabel*/

    performSuperPixelClassifications<<<numBlocks, threadsPerBlock, sharedSize, stream>>>(*derivatives, labelsGpu, planes, params, globalLabelData, smoothed, devicePrevPlanes, devicePrevOpticalFlow, previousPlaneCount, maxLabel);

    if (data.id > 1 && this->useTemporalSmoothing) {
        CUDA_SAFE_CALL(this->logger, cudaFreeAsync(devicePrevPlanes, stream));
        CUDA_SAFE_CALL(this->logger, cudaFreeAsync(devicePrevOpticalFlow, stream));
    }

    CUDA_SAFE_CALL(this->logger, cudaPeekAtLastError());
    CUDA_SAFE_CALL(this->logger, cudaStreamSynchronize(stream));
    CUDA_SAFE_CALL(this->logger, cudaStreamDestroy(stream));

    if (this->useTemporalSmoothing) {
        if (data.id == 1) {
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

void SuperPixelDisparityPlaneSegmentationModule::updatePlaneParameters(System& system, SystemRunData& data) {
    // TODO: Inspiration for alternative: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7926705&tag=1
    if (data.id % this->updateInterval != 1) {
        return;
    }

    // TODO: Sum up the histograms from all the recent runs
    cv::Mat histogram;
    data.getData<cv::cuda::GpuMat>(CARTSLAM_KEY_DISPARITY_DERIVATIVE_HISTOGRAM)->download(histogram);

    this->planeParameterProvider->updatePlaneParameters(this->logger, system, data, histogram);

    system.insertGlobalData(CARTSLAM_KEY_PLANE_PARAMETERS, boost::make_shared<PlaneParameters>(this->planeParameterProvider->getPlaneParameters()));

    auto histShared = boost::make_shared<cv::Mat>(boost::move(histogram));
    system.insertGlobalData(CARTSLAM_KEY_DISPARITY_DERIVATIVE_HIST, histShared);
}
}  // namespace cart