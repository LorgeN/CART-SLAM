#include "cartslam.hpp"
#include "modules/planefit.hpp"
#include "utils/plane.hpp"

struct __align__(16) label_statistics_t {
    unsigned int pixelCount;
    unsigned int inliers;
};

__global__ void calculateRegionDistance(const cv::cuda::PtrStepSz<cart::contour::label_t> superpixels, const cv::cuda::PtrStepSz<cv::Point3f> depth, const cv::Vec4d plane, label_statistics_t* labelStats) {
   // TODO: Implement this function
}

namespace cart {

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
            const cv::Point3d point3d(point.x, point.y, point.z);
            superpixelPoints[label].push_back(point3d);
        }
    }

    std::vector<cv::Vec4d> planes(*maxLabel + 1);

#pragma omp parallel for schedule(static)
    for (contour::label_t label = 0; label <= *maxLabel; label++) {
        auto inliers = superpixelPoints[label];
        if (inliers.size() < 4) {
            planes[label] = cv::Vec4d(0, 0, 0, 0);
            continue;
        }

        planes[label] = util::segmentPlane(inliers);
        LOG4CXX_DEBUG(logger, "Label " << label << ": " << planes[label]);
    }

    return MODULE_RETURN_SHARED(CARTSLAM_KEY_PLANES_EQ, std::vector<cv::Vec4d>, boost::move(planes));
}
}  // namespace cart