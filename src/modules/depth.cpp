#include "modules/depth.hpp"

#include <opencv2/cudastereo.hpp>

#include "cartslam.hpp"
#include "utils/modules.hpp"

namespace cart {
system_data_t DepthModule::runInternal(System& system, SystemRunData& data) {
    auto disparityData = data.getData<cv::cuda::GpuMat>(CARTSLAM_KEY_DISPARITY);

    auto Q = system.getDataSource()->getCameraIntrinsics().Q;

    cv::cuda::GpuMat depth;
    cv::cuda::Stream stream;

    // Scale disparity by / 16.0 and convert to float
    cv::cuda::GpuMat disparityFloat;
    disparityData->convertTo(disparityFloat, CV_32F, 1.0 / 16.0, stream);
    cv::cuda::reprojectImageTo3D(disparityFloat, depth, Q, 3, stream);

    stream.waitForCompletion();

    return MODULE_RETURN_SHARED(CARTSLAM_KEY_DEPTH, cv::cuda::GpuMat, boost::move(depth));
}

bool DepthVisualizationModule::updateImage(System& system, SystemRunData& data, cv::Mat& image) {
    auto depth = data.getData<cv::cuda::GpuMat>(CARTSLAM_KEY_DEPTH);
    cv::Mat localImage;
    depth->download(localImage);

    cv::Mat channels[3];
    cv::split(localImage, channels);

    // Only show the Z (depth) channel
    image = channels[2];

    // Find the minimum and maximum values
    double min, max;
    cv::minMaxLoc(image, &min, &max, NULL, NULL);
    LOG4CXX_DEBUG(logger, "Depth min: " << min << ", max: " << max);

    image.convertTo(image, CV_8U, 255.0 / 10.0);
    return true;
}
}  // namespace cart