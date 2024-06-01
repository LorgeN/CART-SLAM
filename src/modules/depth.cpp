#include "modules/depth.hpp"

#include <opencv2/cudastereo.hpp>

#include "cartslam.hpp"
#include "utils/modules.hpp"

namespace cart {
system_data_t DepthModule::runInternal(System& system, SystemRunData& data) {
    auto disparityData = data.getData<cv::cuda::GpuMat>(CARTSLAM_KEY_DISPARITY);

    static auto Q = system.getDataSource()->getCameraIntrinsics().Q;

    cv::cuda::GpuMat depth;
    cv::cuda::Stream stream;

    // Print Q matrix
    LOG4CXX_DEBUG(this->logger, "Q matrix: ");
    for (int i = 0; i < 4; i++) {
        LOG4CXX_DEBUG(this->logger, Q.at<float>(i, 0) << " " << Q.at<float>(i, 1) << " " << Q.at<float>(i, 2) << " " << Q.at<float>(i, 3));
    }

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
    image.convertTo(image, CV_8U, 255.0 / 30.0);
    return true;
}
}  // namespace cart