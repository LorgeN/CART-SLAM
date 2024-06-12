#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>

#include "cartslam.hpp"
#include "modules/planefit.hpp"
#include "utils/cuda.cuh"
#include "utils/plane.hpp"
#include "utils/random.hpp"

#define THREADS_X 32
#define THREADS_Y 32
#define X_BATCH 4
#define Y_BATCH 4

#define MAX_SHARED_MEMORY 32768

#define CALC_COLOR(ch) std::min(static_cast<uint8_t>(255), static_cast<uint8_t>(std::abs(normal[ch]) / 2.0 * 255))

namespace cg = cooperative_groups;

struct plane_t {
    double a;
    double b;
    double c;
    double d;
};

typedef unsigned int label_inliers_t;

struct label_statistics_t {
    uint16_t pixelCount;
    uint16_t pixelCountInvalid;
};

__device__ double calculateDistanceFromPlane(const plane_t plane, const cv::Point3f point) {
    return std::abs(plane.a * point.x + plane.b * point.y + plane.c * point.z + plane.d) / std::sqrt(plane.a * plane.a + plane.b * plane.b + plane.c * plane.c);
}

__global__ void countPixels(
    const cv::cuda::PtrStepSz<cart::contour::label_t> superpixels,
    const cv::cuda::PtrStepSz<cv::Point3f> depth,
    label_statistics_t* labelStats,
    const cart::contour::label_t maxLabelId) {
    extern __shared__ label_statistics_t sharedStats[];
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

            if (!std::isfinite(point.z)) {
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
    label_inliers_t** labelStats,
    const cart::contour::label_t maxLabelId,
    const double threshold,
    const size_t planeCount) {
    extern __shared__ label_inliers_t sharedInliers[];

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

                if (!std::isfinite(point.z)) {
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

void SuperPixelPlaneFitModule::attemptAssignment(
    const cv::cuda::GpuMat& superpixels,
    const cv::cuda::GpuMat& depth,
    const std::vector<cv::Vec4d>& planes,
    size_t& assignedPlanes,
    const cart::contour::label_t maxLabelId,
    const double threshold,
    std::vector<size_t>& planeAssignments) {
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

    sharedMemSize = sizeof(label_inliers_t) * (maxLabelId + 1);
    calculateRegionDistance<<<blocks, threads, sharedMemSize, stream>>>(superpixels, depth, planesDevice, labelInliers, maxLabelId, threshold, planes.size());

    label_inliers_t** labelInliersResultHost = new label_inliers_t*[planes.size()];
    label_statistics_t* labelStatsHost = new label_statistics_t[maxLabelId + 1];
    CUDA_SAFE_CALL(this->logger, cudaMemcpyAsync(labelStatsHost, labelStats, sizeof(label_statistics_t) * (maxLabelId + 1), cudaMemcpyDeviceToHost, stream));
    CUDA_SAFE_CALL(this->logger, cudaFreeAsync(labelStats, stream));

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

    // Check each label and assign it to the plane with the most inliers
    for (size_t label = 0; label < maxLabelId + 1; label++) {
        size_t maxInliers = 0;
        size_t maxPlane = 0;
        size_t pixelCount = labelStatsHost[label].pixelCount;
        size_t pixelCountInvalid = labelStatsHost[label].pixelCountInvalid;

        // Check if percentage of valid pixels is high enough
        if (pixelCountInvalid > 0.5 * pixelCount) {
            planeAssignments[label] = 0;
            assignedPlanes++;
            continue;
        }

        for (size_t plane = 0; plane < planes.size(); plane++) {
            if (labelInliersResultHost[plane][label] > maxInliers) {
                maxInliers = labelInliersResultHost[plane][label];
                maxPlane = plane;
            }
        }

        if (maxInliers > 0.5 * pixelCount) {
            planeAssignments[label] = maxPlane + 1;
            assignedPlanes++;
        }
    }

    for (size_t i = 0; i < planes.size(); i++) {
        delete[] labelInliersResultHost[i];
    }

    delete[] labelStatsHost;
    delete[] labelInliersResultHost;
}

std::vector<size_t> selectRandomSuperpixels(const cv::Mat superpixels, int xCount, int yCount) {
    // Select points in a grid pattern, with some randomness

    std::vector<size_t> selectedSuperpixels;

    int yStep = superpixels.rows / (yCount + 1);
    int xStep = superpixels.cols / (xCount + 1);

    std::random_device dev;
    std::mt19937 rng(dev());
    std::uniform_int_distribution<std::mt19937::result_type> distY(-yStep / 2, yStep / 2);
    std::uniform_int_distribution<std::mt19937::result_type> distX(-xStep / 2, xStep / 2);

    for (int y = 0; y < superpixels.rows; y += yStep) {
        for (int x = 0; x < superpixels.cols; x += xStep) {
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

            if (!std::isfinite(point.z)) {
                continue;
            }

            const cv::Point3d point3d(point.x, point.y, point.z);
            superpixelPoints[label].push_back(point3d);
        }
    }

    std::vector<cv::Vec4d> planes;
    std::vector<size_t> planeAssignments(*maxLabel + 1, 0);
    size_t assignmentCount = 0;

    auto sampler = util::RandomSampler(*maxLabel + 1);

    while (static_cast<double>(assignmentCount) / static_cast<double>(*maxLabel + 1) < 0.9) {
        planes.clear();
        planeAssignments.assign(*maxLabel + 1, 0);

        auto sample = selectRandomSuperpixels(superpixelsHost, 4, 3);

        for (const auto& label : sample) {
            if (superpixelPoints[label].size() < 16) {
                continue;
            }

            auto plane = cart::util::segmentPlane(superpixelPoints[label]);
            planes.push_back(plane);
        }

        if (planes.size() < 3) {
            continue;
        }

        attemptAssignment(*superpixels, *depth, planes, assignmentCount, *maxLabel, 0.1, planeAssignments);
    }

    plane_fit_data_t planeFitData;
    planeFitData.planes = planes;
    planeFitData.planeAssignments = planeAssignments;

    return MODULE_RETURN_SHARED(CARTSLAM_KEY_PLANES_EQ, plane_fit_data_t, boost::move(planeFitData));
}

bool SuperPixelPlaneFitVisualizationModule::updateImage(System& system, SystemRunData& data, cv::Mat& image) {
    const auto maxLabel = data.getData<contour::label_t>(CARTSLAM_KEY_SUPERPIXELS_MAX_LABEL);
    const auto superpixels = data.getData<cv::cuda::GpuMat>(CARTSLAM_KEY_SUPERPIXELS);
    const auto planesFit = data.getData<plane_fit_data_t>(CARTSLAM_KEY_PLANES_EQ);

    cv::Mat superpixelsHost;
    superpixels->download(superpixelsHost);

    cv::Mat vectorColoring(superpixelsHost.size(), CV_8UC3);

    for (int y = 0; y < superpixelsHost.rows; y++) {
        for (int x = 0; x < superpixelsHost.cols; x++) {
            const auto label = superpixelsHost.at<contour::label_t>(y, x);
            const auto planeId = planesFit->planeAssignments[label];

            if (planeId == 0) {
                vectorColoring.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 0, 0);
            } else {
                const auto plane = planesFit->planes[planeId - 1];
                const auto normal = cv::Vec3d(plane[0], plane[1], plane[2]);

                vectorColoring.at<cv::Vec3b>(y, x) = cv::Vec3b(CALC_COLOR(0), CALC_COLOR(1), CALC_COLOR(2));
            }
        }
    }

    auto referenceImage = getReferenceImage(data.dataElement);

    cv::Mat imageHost;
    referenceImage.download(imageHost);

    cv::vconcat(vectorColoring, imageHost, image);
    return true;
}
}  // namespace cart