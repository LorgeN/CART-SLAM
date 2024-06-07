#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>

#include "cartslam.hpp"
#include "modules/planefit.hpp"
#include "utils/plane.hpp"
#include "utils/random.hpp"

#define X_BATCH 4
#define Y_BATCH 4

namespace cg = cooperative_groups;

struct __align__(8) label_statistics_t {
    unsigned int pixelCount;
    unsigned int inliers;
};

__global__ void calculateRegionDistance(
    const cv::cuda::PtrStepSz<cart::contour::label_t> superpixels,
    const cv::cuda::PtrStepSz<cv::Point3f> depth,
    const cv::Vec4d plane,
    label_statistics_t* labelStats) {
    /*
extern __shared__ label_statistics_t stats[];

cg::thread_block tb = cg::this_thread_block();

int x = blockIdx.x * blockDim.x + threadIdx.x;
int y = blockIdx.y * blockDim.y + threadIdx.y;

int pixelX = x * X_BATCH;
int pixelY = y * Y_BATCH;
*/

    // Set the shared memory to zero

    // TODO: Implement this function
}

namespace cart {

system_data_t SuperPixelPlaneFitModule::runInternal(System& system, SystemRunData& data) {
    const auto maxLabel = data.getData<contour::label_t>(CARTSLAM_KEY_SUPERPIXELS_MAX_LABEL);
    const auto superpixels = data.getData<cv::cuda::GpuMat>(CARTSLAM_KEY_SUPERPIXELS);
    const auto depth = data.getData<cv::cuda::GpuMat>(CARTSLAM_KEY_DEPTH);

    std::vector<std::vector<cv::Point3d>> superpixelPoints(*maxLabel + 1);
    util::RandomSampler sampler(*maxLabel);

    cv::Mat superpixelsHost;
    cv::Mat depthHost;
    superpixels->download(superpixelsHost);
    depth->download(depthHost);

    for (int y = 0; y < superpixelsHost.rows; y++) {
        for (int x = 0; x < superpixelsHost.cols; x++) {
            const auto label = superpixelsHost.at<contour::label_t>(y, x);
            const auto point = depthHost.at<cv::Point3f>(y, x);
            const cv::Point3d point3d(point.x, point.y, point.z);
            superpixelPoints[label].push_back(point3d);
        }
    }

    return MODULE_NO_RETURN_VALUE;
}
}  // namespace cart