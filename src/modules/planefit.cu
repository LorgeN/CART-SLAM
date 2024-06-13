#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>

#include <opencv2/core/cuda_stream_accessor.hpp>

#include "cartslam.hpp"
#include "modules/planefit.hpp"
#include "utils/colors.hpp"
#include "utils/cuda.cuh"
#include "utils/plane.hpp"
#include "utils/random.hpp"

#define THREADS_X 32
#define THREADS_Y 32
#define X_BATCH 4
#define Y_BATCH 4

#define MAX_SHARED_MEMORY 32768
#define IS_VALID_DEPTH(depth) (std::isfinite(depth) && (depth) <= 40.0 && (depth) > 0.0)
#define IS_VALID_REGION(stats) (stats.pixelCountInvalid < 0.5 * stats.pixelCount)

#define CALC_COLOR(ch) std::min(static_cast<uint8_t>(255), static_cast<uint8_t>(std::abs(normal[ch]) / 2.0 * 255))

namespace cg = cooperative_groups;

struct plane_t {
    double a;
    double b;
    double c;
    double d;
};

__device__ double calculateDistanceFromPlane(const plane_t plane, const cv::Point3f point) {
    return std::abs(plane.a * point.x + plane.b * point.y + plane.c * point.z + plane.d) / std::sqrt(plane.a * plane.a + plane.b * plane.b + plane.c * plane.c);
}

__global__ void countPixels(
    const cv::cuda::PtrStepSz<cart::contour::label_t> superpixels,
    const cv::cuda::PtrStepSz<cv::Point3f> depth,
    cart::label_statistics_t* labelStats,
    const cart::contour::label_t maxLabelId) {
    extern __shared__ cart::label_statistics_t sharedStats[];
    for (int i = threadIdx.x + (blockDim.x * threadIdx.y); i <= maxLabelId; i += blockDim.x * blockDim.y) {
        sharedStats[i].pixelCount = 0;
        sharedStats[i].pixelCountInvalid = 0;
    }

    cg::thread_block tb = cg::this_thread_block();

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int pixelX = x * X_BATCH;
    int pixelY = y * Y_BATCH;

    tb.sync();

    for (int i = 0; i < X_BATCH; i++) {
        for (int j = 0; j < Y_BATCH; j++) {
            if (pixelX + i >= superpixels.cols || pixelY + j >= superpixels.rows) {
                continue;
            }

            const auto label = superpixels(pixelY + j, pixelX + i);
            const auto point = depth(pixelY + j, pixelX + i);

            if (!IS_VALID_DEPTH(point.z)) {
                unsafeAtomicAdd(&sharedStats[label].pixelCountInvalid, 1);
            }

            unsafeAtomicAdd(&sharedStats[label].pixelCount, 1);
        }
    }

    tb.sync();

    for (int i = threadIdx.x + (blockDim.x * threadIdx.y); i <= maxLabelId; i += blockDim.x * blockDim.y) {
        unsafeAtomicAdd(&labelStats[i].pixelCount, sharedStats[i].pixelCount);
        unsafeAtomicAdd(&labelStats[i].pixelCountInvalid, sharedStats[i].pixelCountInvalid);
    }
}

__global__ void calculateRegionDistance(
    const cv::cuda::PtrStepSz<cart::contour::label_t> superpixels,
    const cv::cuda::PtrStepSz<cv::Point3f> depth,
    const plane_t* planes,
    cart::label_inliers_t** labelStats,
    const cart::contour::label_t maxLabelId,
    const double threshold,
    const size_t planeCount) {
    extern __shared__ cart::label_inliers_t sharedInliers[];

    cg::thread_block tb = cg::this_thread_block();

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int pixelX = x * X_BATCH;
    int pixelY = y * Y_BATCH;

    for (size_t planeId = 0; planeId < planeCount; planeId++) {
        for (int i = threadIdx.x + (blockDim.x * threadIdx.y); i <= maxLabelId; i += blockDim.x * blockDim.y) {
            sharedInliers[i] = 0;
        }

        tb.sync();

        const auto plane = planes[planeId];

        for (int i = 0; i < X_BATCH; i++) {
            for (int j = 0; j < Y_BATCH; j++) {
                if (pixelX + i >= superpixels.cols || pixelY + j >= superpixels.rows) {
                    continue;
                }

                const auto label = superpixels(pixelY + j, pixelX + i);
                const auto point = depth(pixelY + j, pixelX + i);

                if (!IS_VALID_DEPTH(point.z)) {
                    continue;
                }

                const auto distance = calculateDistanceFromPlane(plane, point);

                if (distance < threshold) {
                    atomicAdd(&sharedInliers[label], 1);
                }
            }
        }

        tb.sync();

        for (int i = threadIdx.x + (blockDim.x * threadIdx.y); i <= maxLabelId; i += blockDim.x * blockDim.y) {
            atomicAdd(&labelStats[planeId][i], sharedInliers[i]);
        }
    }
}

__global__ void overlayPlanesByIdx(cv::cuda::PtrStepSz<uint8_t> image, cv::cuda::PtrStepSz<cart::contour::label_t> labels, size_t* labelAssignments, cv::cuda::PtrStepSz<uint8_t> output, float planeCount) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int pixelX = x * X_BATCH;
    int pixelY = y * Y_BATCH;

    size_t outputRowStep = output.step / sizeof(uint8_t);
    size_t imageRowStep = image.step / sizeof(uint8_t);
    size_t labelsRowStep = labels.step / sizeof(cart::contour::label_t);

    uint8_t colors[3] = {0, 0, 0};

    for (int i = 0; i < Y_BATCH; i++) {
        for (int j = 0; j < X_BATCH; j++) {
            if (pixelX + j >= image.cols || pixelY + i >= image.rows) {
                continue;
            }

            cart::contour::label_t label = labels[INDEX(pixelX + j, pixelY + i, labelsRowStep)];
            size_t plane = labelAssignments[label];

            if (plane != 0) {
                cart::assignColor(static_cast<float>(plane) / planeCount, colors);
            } else {
                colors[0] = 0;
                colors[1] = 0;
                colors[2] = 0;
            }

            uint8_t b = image[INDEX_BGR(pixelX + j, pixelY + i, 0, imageRowStep)];
            uint8_t g = image[INDEX_BGR(pixelX + j, pixelY + i, 1, imageRowStep)];
            uint8_t r = image[INDEX_BGR(pixelX + j, pixelY + i, 2, imageRowStep)];

            output[INDEX_BGR(pixelX + j, pixelY + i, 0, outputRowStep)] = b / 2 + colors[0] / 2;
            output[INDEX_BGR(pixelX + j, pixelY + i, 1, outputRowStep)] = g / 2 + colors[1] / 2;
            output[INDEX_BGR(pixelX + j, pixelY + i, 2, outputRowStep)] = r / 2 + colors[2] / 2;
        }
    }
}

namespace cart {
SuperPixelPlaneFitModule::SuperPixelPlaneFitModule() : SyncWrapperSystemModule("PlaneFit") {
    this->requiresData.push_back(module_dependency_t(CARTSLAM_KEY_DEPTH));
    this->requiresData.push_back(module_dependency_t(CARTSLAM_KEY_SUPERPIXELS));
    this->requiresData.push_back(module_dependency_t(CARTSLAM_KEY_SUPERPIXELS_MAX_LABEL));
    this->requiresData.push_back(module_dependency_t(CARTSLAM_KEY_DISPARITY_DERIVATIVE));

    this->providesData.push_back(CARTSLAM_KEY_PLANES_EQ);

    CUDA_SAFE_CALL(this->logger, cudaFuncSetAttribute(countPixels, cudaFuncAttributeMaxDynamicSharedMemorySize, MAX_SHARED_MEMORY));
    CUDA_SAFE_CALL(this->logger, cudaFuncSetAttribute(calculateRegionDistance, cudaFuncAttributeMaxDynamicSharedMemorySize, MAX_SHARED_MEMORY));
};

void SuperPixelPlaneFitModule::generateLabelStatistics(const cv::cuda::GpuMat& superpixels,
                                                       const cv::cuda::GpuMat& depth,
                                                       label_statistics_t* labelStatsHost,
                                                       const cart::contour::label_t maxLabelId) {
    cudaStream_t stream;
    CUDA_SAFE_CALL(this->logger, cudaStreamCreate(&stream));

    label_statistics_t* labelStats;
    CUDA_SAFE_CALL(this->logger, cudaMallocAsync(&labelStats, sizeof(label_statistics_t) * (maxLabelId + 1), stream));
    CUDA_SAFE_CALL(this->logger, cudaMemsetAsync(labelStats, 0, sizeof(label_statistics_t) * (maxLabelId + 1), stream));

    dim3 threads(THREADS_X, THREADS_Y);
    dim3 blocks((superpixels.cols + THREADS_X * X_BATCH - 1) / (THREADS_X * X_BATCH), (superpixels.rows + THREADS_Y * Y_BATCH - 1) / (THREADS_Y * Y_BATCH));

    size_t sharedMemSize = sizeof(label_statistics_t) * (maxLabelId + 1);
    if (sharedMemSize > MAX_SHARED_MEMORY) {
        throw std::runtime_error("Shared memory size exceeds maximum. Was " + std::to_string(sharedMemSize) + " but maximum is " + std::to_string(MAX_SHARED_MEMORY) + ".");
    }

    countPixels<<<blocks, threads, sharedMemSize, stream>>>(superpixels, depth, labelStats, maxLabelId);

    CUDA_SAFE_CALL(this->logger, cudaMemcpyAsync(labelStatsHost, labelStats, sizeof(label_statistics_t) * (maxLabelId + 1), cudaMemcpyDeviceToHost, stream));
    CUDA_SAFE_CALL(this->logger, cudaFreeAsync(labelStats, stream));

    CUDA_SAFE_CALL(this->logger, cudaGetLastError());
    CUDA_SAFE_CALL(this->logger, cudaStreamSynchronize(stream));
    CUDA_SAFE_CALL(this->logger, cudaStreamDestroy(stream));
}

std::pair<cv::Vec4d, std::vector<cart::contour::label_t>> SuperPixelPlaneFitModule::attemptAssignment(const cv::cuda::GpuMat& superpixels,
                                                                                                      const cv::cuda::GpuMat& depth,
                                                                                                      const std::vector<cv::Vec4d>& planes,
                                                                                                      const cart::contour::label_t maxLabelId,
                                                                                                      const std::vector<size_t>& planeAssignments,
                                                                                                      const label_statistics_t* labelStatsHost,
                                                                                                      const double threshold) {
    cudaStream_t stream;
    CUDA_SAFE_CALL(this->logger, cudaStreamCreate(&stream));

    label_inliers_t** labelInliersHost[planes.size()];

    for (size_t i = 0; i < planes.size(); i++) {
        CUDA_SAFE_CALL(this->logger, cudaMallocAsync(&labelInliersHost[i], sizeof(label_inliers_t) * (maxLabelId + 1), stream));
        CUDA_SAFE_CALL(this->logger, cudaMemsetAsync(labelInliersHost[i], 0, sizeof(label_inliers_t) * (maxLabelId + 1), stream));
    }

    CUDA_SAFE_CALL(this->logger, cudaStreamSynchronize(stream));

    label_inliers_t** labelInliers;
    CUDA_SAFE_CALL(this->logger, cudaMallocAsync(&labelInliers, sizeof(label_inliers_t*) * planes.size(), stream));
    CUDA_SAFE_CALL(this->logger, cudaMemcpyAsync(labelInliers, labelInliersHost, sizeof(label_inliers_t*) * planes.size(), cudaMemcpyHostToDevice, stream));

    plane_t planesHost[planes.size()];
    for (size_t i = 0; i < planes.size(); i++) {
        planesHost[i] = {planes[i][0], planes[i][1], planes[i][2], planes[i][3]};
    }

    plane_t* planesDevice;
    CUDA_SAFE_CALL(this->logger, cudaMallocAsync(&planesDevice, sizeof(plane_t) * planes.size(), stream));
    CUDA_SAFE_CALL(this->logger, cudaMemcpyAsync(planesDevice, planesHost, sizeof(plane_t) * planes.size(), cudaMemcpyHostToDevice, stream));

    dim3 threads(THREADS_X, THREADS_Y);
    dim3 blocks((superpixels.cols + THREADS_X * X_BATCH - 1) / (THREADS_X * X_BATCH), (superpixels.rows + THREADS_Y * Y_BATCH - 1) / (THREADS_Y * Y_BATCH));

    size_t sharedMemSize = sizeof(label_inliers_t) * (maxLabelId + 1);
    if (sharedMemSize > MAX_SHARED_MEMORY) {
        throw std::runtime_error("Shared memory size exceeds maximum. Was " + std::to_string(sharedMemSize) + " but maximum is " + std::to_string(MAX_SHARED_MEMORY) + ".");
    }

    calculateRegionDistance<<<blocks, threads, sharedMemSize, stream>>>(superpixels, depth, planesDevice, labelInliers, maxLabelId, threshold, planes.size());

    label_inliers_t** labelInliersResultHost = new label_inliers_t*[planes.size()];

    for (size_t i = 0; i < planes.size(); i++) {
        labelInliersResultHost[i] = new label_inliers_t[maxLabelId + 1];

        CUDA_SAFE_CALL(this->logger, cudaMemcpyAsync(labelInliersResultHost[i], labelInliersHost[i], sizeof(label_inliers_t) * (maxLabelId + 1), cudaMemcpyDeviceToHost, stream));
    }

    for (size_t i = 0; i < planes.size(); i++) {
        CUDA_SAFE_CALL(this->logger, cudaFreeAsync(labelInliersHost[i], stream));
    }

    CUDA_SAFE_CALL(this->logger, cudaFreeAsync(labelInliers, stream));
    CUDA_SAFE_CALL(this->logger, cudaFreeAsync(planesDevice, stream));

    CUDA_SAFE_CALL(this->logger, cudaGetLastError());
    CUDA_SAFE_CALL(this->logger, cudaStreamSynchronize(stream));
    CUDA_SAFE_CALL(this->logger, cudaStreamDestroy(stream));

    std::vector<std::vector<cart::contour::label_t>> acceptablePlaneLabels(planes.size());
    size_t planeAssignmentCount[planes.size()] = {0};

    // Check each label and assign it to the plane with the most inliers
    for (size_t label = 0; label < maxLabelId + 1; label++) {
        size_t pixelCount = labelStatsHost[label].pixelCount;

        // Check if percentage of valid pixels is high enough
        if (!IS_VALID_REGION(labelStatsHost[label])) {
            continue;
        }

        if (planeAssignments[label] != 0) {
            continue;
        }

        for (size_t plane = 0; plane < planes.size(); plane++) {
            const auto inliers = labelInliersResultHost[plane][label];
            if (inliers > 0.5 * pixelCount) {
                acceptablePlaneLabels[plane].push_back(label);
                planeAssignmentCount[plane]++;
            }
        }
    }

    // Find the plane with the most inliers and assign that as the initial plane
    size_t maxPlane = 0;
    size_t maxPlaneCount = 0;

    for (size_t plane = 0; plane < planes.size(); plane++) {
        if (planeAssignmentCount[plane] > maxPlaneCount) {
            maxPlane = plane;
            maxPlaneCount = planeAssignmentCount[plane];
        }
    }

    for (size_t i = 0; i < planes.size(); i++) {
        delete[] labelInliersResultHost[i];
    }

    delete[] labelInliersResultHost;

    return std::make_pair(planes[maxPlane], acceptablePlaneLabels[maxPlane]);
}

std::vector<size_t> selectRandomSuperpixels(const cv::Mat superpixels, int xCount, int yCount) {
    // Select points in a grid pattern, with some randomness

    std::vector<size_t> selectedSuperpixels;

    int yStep = superpixels.rows / (yCount + 2);
    int xStep = superpixels.cols / (xCount + 2);

    std::random_device dev;
    std::mt19937 rng(dev());
    std::uniform_int_distribution<std::mt19937::result_type> distY(-yStep / 2, yStep / 2);
    std::uniform_int_distribution<std::mt19937::result_type> distX(-xStep / 2, xStep / 2);

    for (int y = yStep; y < superpixels.rows; y += yStep) {
        for (int x = xStep; x < superpixels.cols; x += xStep) {
            const int xOffset = x + distX(rng);
            const int yOffset = y + distY(rng);
            if (xOffset < 0 || xOffset >= superpixels.cols || yOffset < 0 || yOffset >= superpixels.rows) {
                continue;
            }

            selectedSuperpixels.push_back(superpixels.at<contour::label_t>(yOffset, xOffset));
        }
    }

    return selectedSuperpixels;
}

system_data_t SuperPixelPlaneFitModule::runInternal(System& system, SystemRunData& data) {
    const auto maxLabel = data.getData<contour::label_t>(CARTSLAM_KEY_SUPERPIXELS_MAX_LABEL);
    const auto superpixels = data.getData<cv::cuda::GpuMat>(CARTSLAM_KEY_SUPERPIXELS);
    const auto depth = data.getData<cv::cuda::GpuMat>(CARTSLAM_KEY_DEPTH);

    std::vector<std::vector<cv::Point3d>> superpixelPoints(*maxLabel + 1);

    cv::Mat superpixelsHost;
    cv::Mat depthHost;
    superpixels->download(superpixelsHost);
    depth->download(depthHost);

    for (int y = 0; y < superpixelsHost.rows; y++) {
        for (int x = 0; x < superpixelsHost.cols; x++) {
            const auto label = superpixelsHost.at<contour::label_t>(y, x);
            const auto point = depthHost.at<cv::Point3f>(y, x);

            if (!IS_VALID_DEPTH(point.z)) {
                continue;
            }

            const cv::Point3d point3d(point.x, point.y, point.z);
            superpixelPoints[label].push_back(point3d);
        }
    }

    std::vector<cv::Vec4d> planes;
    std::vector<size_t> planeAssignments(*maxLabel + 1, 0);
    size_t assignmentCount = 0;

    label_statistics_t* labelStats = new label_statistics_t[*maxLabel + 1];
    generateLabelStatistics(*superpixels, *depth, labelStats, *maxLabel);

    // Exclude invalid regions
    for (int i = 0; i < *maxLabel + 1; i++) {
        if (IS_VALID_REGION(labelStats[i])) {
            planeAssignments[i] = 0;
            assignmentCount++;
        }
    }

    util::RandomSampler sampler(*maxLabel + 1);

    size_t iter = 0;
    while (static_cast<double>(assignmentCount) / static_cast<double>(*maxLabel + 1) < 0.9 && iter++ < 100) {
        auto sample = selectRandomSuperpixels(superpixelsHost, 4, 3);

        std::vector<cv::Vec4d> localPlanes;
        for (const auto& label : sample) {
            if (planeAssignments[label] != 0 || !IS_VALID_REGION(labelStats[label])) {
                continue;
            }

            if (superpixelPoints[label].size() < 16) {
                continue;
            }

            auto plane = cart::util::segmentPlane(superpixelPoints[label], 0.01);
            localPlanes.push_back(plane);
        }

        if (localPlanes.size() <= 3) {
            continue;
        }

        auto result = attemptAssignment(*superpixels, *depth, localPlanes, *maxLabel, planeAssignments, labelStats, 0.02);
        const auto plane = result.first;
        const auto acceptableLabels = result.second;

        if (acceptableLabels.size() < 16) {
            continue;
        }

        planes.push_back(plane);
        for (const auto& label : acceptableLabels) {
            planeAssignments[label] = planes.size();
        }

        assignmentCount += acceptableLabels.size();
    }

    delete[] labelStats;

    plane_fit_data_t planeFitData;
    planeFitData.planes = planes;
    planeFitData.planeAssignments = planeAssignments;

    return MODULE_RETURN_SHARED(CARTSLAM_KEY_PLANES_EQ, plane_fit_data_t, boost::move(planeFitData));
}

bool SuperPixelPlaneFitVisualizationModule::updateImage(System& system, SystemRunData& data, cv::Mat& image) {
    const auto maxLabel = data.getData<contour::label_t>(CARTSLAM_KEY_SUPERPIXELS_MAX_LABEL);
    const auto superpixels = data.getData<cv::cuda::GpuMat>(CARTSLAM_KEY_SUPERPIXELS);
    const auto planesFit = data.getData<plane_fit_data_t>(CARTSLAM_KEY_PLANES_EQ);

    const float planeCount = static_cast<float>(planesFit->planes.size());

    cudaStream_t stream;
    CUDA_SAFE_CALL(this->logger, cudaStreamCreate(&stream));

    cart::copyColorWheelToDevice(stream);

    cv::cuda::GpuMat output(superpixels->size(), CV_8UC3);

    const auto referenceImage = getReferenceImage(data.dataElement);

    size_t* planeAssignments = new size_t[*maxLabel + 1];
    for (size_t i = 0; i < *maxLabel + 1; i++) {
        planeAssignments[i] = planesFit->planeAssignments[i];
    }

    size_t* planeAssignmentsDevice;
    CUDA_SAFE_CALL(this->logger, cudaMallocAsync(&planeAssignmentsDevice, sizeof(size_t) * (*maxLabel + 1), stream));
    CUDA_SAFE_CALL(this->logger, cudaMemcpyAsync(planeAssignmentsDevice, planeAssignments, sizeof(size_t) * (*maxLabel + 1), cudaMemcpyHostToDevice, stream));

    dim3 threads(THREADS_X, THREADS_Y);
    dim3 blocks((superpixels->cols + THREADS_X * X_BATCH - 1) / (THREADS_X * X_BATCH), (superpixels->rows + THREADS_Y * Y_BATCH - 1) / (THREADS_Y * Y_BATCH));

    overlayPlanesByIdx<<<blocks, threads, 0, stream>>>(referenceImage, *superpixels, planeAssignmentsDevice, output, planeCount);

    cv::cuda::Stream cvStream = cv::cuda::StreamAccessor::wrapStream(stream);
    output.download(image, cvStream);

    CUDA_SAFE_CALL(this->logger, cudaFreeAsync(planeAssignmentsDevice, stream));

    CUDA_SAFE_CALL(this->logger, cudaGetLastError());
    CUDA_SAFE_CALL(this->logger, cudaStreamSynchronize(stream));
    CUDA_SAFE_CALL(this->logger, cudaStreamDestroy(stream));

    delete[] planeAssignments;

    return true;
}
}  // namespace cart