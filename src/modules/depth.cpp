#include "modules/depth.hpp"

#include <opencv2/cudastereo.hpp>

#include "utils/modules.hpp"

namespace cart {
system_data_t DepthModule::runInternal(System& system, SystemRunData& data) {
    auto disparityData = data.getData<cv::cuda::GpuMat>(CARTSLAM_KEY_DISPARITY);

    static auto Q = system.getDataSource()->getCameraIntrinsics().Q;

    cv::cuda::GpuMat depth;
    cv::cuda::Stream stream;

    cv::cuda::reprojectImageTo3D(*disparityData, depth, Q, 3, stream);

    stream.waitForCompletion();

    return MODULE_RETURN_SHARED(CARTSLAM_KEY_DEPTH, cv::cuda::GpuMat, boost::move(depth));
}

boost::future<system_data_t> DepthVisualizationModule::run(System& system, SystemRunData& data) {
    auto promise = boost::make_shared<boost::promise<system_data_t>>();

    boost::asio::post(system.getThreadPool(), [this, promise, &system, &data]() {
        auto disparity = data.getData<cv::cuda::GpuMat>(CARTSLAM_KEY_DEPTH);

        cv::Mat image;
        disparity->download(image);
        this->imageThread->setImageIfLater(image, data.id);
        promise->set_value(MODULE_NO_RETURN_VALUE);
    });

    return promise->get_future();
}
}  // namespace cart