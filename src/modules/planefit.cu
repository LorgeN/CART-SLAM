#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>

#include "cartslam.hpp"
#include "modules/planefit.hpp"
#include "utils/plane.hpp"
#include "utils/random.hpp"

#define X_BATCH 4
#define Y_BATCH 4

#define CALC_COLOR(ch) std::min(static_cast<uint8_t>(255), static_cast<uint8_t>(std::abs(normal[ch]) / 2.0 * 255))

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

            if (!std::isfinite(point.z)) {
                continue;
            }

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
    }

    return MODULE_RETURN_SHARED(CARTSLAM_KEY_PLANES_EQ, std::vector<cv::Vec4d>, boost::move(planes));
}

bool SuperPixelPlaneFitVisualizationModule::updateImage(System& system, SystemRunData& data, cv::Mat& image) {
    const auto maxLabel = data.getData<contour::label_t>(CARTSLAM_KEY_SUPERPIXELS_MAX_LABEL);
    const auto superpixels = data.getData<cv::cuda::GpuMat>(CARTSLAM_KEY_SUPERPIXELS);
    const auto planesFit = data.getData<std::vector<cv::Vec4d>>(CARTSLAM_KEY_PLANES_EQ);

    cv::Mat superpixelsHost;
    superpixels->download(superpixelsHost);

    cv::Mat vectorColoring(superpixelsHost.size(), CV_8UC3);

    for (int y = 0; y < superpixelsHost.rows; y++) {
        for (int x = 0; x < superpixelsHost.cols; x++) {
            const auto label = superpixelsHost.at<contour::label_t>(y, x);

            const auto plane = (*planesFit)[label];

            if (plane[0] == 0 && plane[1] == 0 && plane[2] == 0) {
                continue;
            }

            auto normal = cv::Vec3d(plane[0], plane[1], plane[2]);
            normal = normal / cv::norm(normal);

            const auto color = cv::Vec3b(CALC_COLOR(0), CALC_COLOR(1), CALC_COLOR(2));
            vectorColoring.at<cv::Vec3b>(y, x) = color;
        }
    }

    auto referenceImage = getReferenceImage(data.dataElement);

    cv::Mat imageHost;
    referenceImage.download(imageHost);

    cv::vconcat(vectorColoring, imageHost, image);
    return true;
}
}  // namespace cart