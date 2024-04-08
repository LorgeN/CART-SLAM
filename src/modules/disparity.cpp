#include "modules/disparity.hpp"

#include <opencv2/ximgproc/disparity_filter.hpp>

namespace cart {
MODULE_RETURN_VALUE ImageDisparityModule::runInternal(System& system, SystemRunData& data) {
    LOG4CXX_DEBUG(this->logger, "Running ImageDisparityModule");

    if (data.dataElement->type != DataElementType::STEREO) {
        throw std::runtime_error("ImageDisparityModule requires StereoDataElement");
    }

    cv::cuda::Stream stream;

    boost::shared_ptr<StereoDataElement> stereoData = boost::static_pointer_cast<StereoDataElement>(data.dataElement);

    cv::cuda::GpuMat disparity;
    this->stereoBM->compute(stereoData->left, stereoData->right, disparity, stream);

    stream.waitForCompletion();

    return MODULE_RETURN(CARTSLAM_KEY_DISPARITY, boost::make_shared<cv::cuda::GpuMat>(boost::move(disparity)));
}

boost::future<MODULE_RETURN_VALUE> ImageDisparityVisualizationModule::run(System& system, SystemRunData& data) {
    auto promise = boost::make_shared<boost::promise<MODULE_RETURN_VALUE>>();

    boost::asio::post(system.threadPool, [this, promise, &system, &data]() {
        auto disparity = data.getData<cv::cuda::GpuMat>(CARTSLAM_KEY_DISPARITY);

        cv::cuda::Stream stream;
        cv::Mat image, disparityData, disparityImage;

        boost::shared_ptr<cart::StereoDataElement> stereoData = boost::static_pointer_cast<cart::StereoDataElement>(data.dataElement);

        stereoData->left.download(image, stream);
        disparity->download(disparityData, stream);

        stream.waitForCompletion();

        disparityData.convertTo(disparityData, CV_16S, 16.0);
        cv::ximgproc::getDisparityVis(disparityData, disparityImage);

        cv::Mat concatRes;
        cv::vconcat(image, disparityImage, concatRes);

        this->imageThread.setImageIfLater(concatRes, data.id);

        promise->set_value(MODULE_NO_RETURN_VALUE);
    });

    return promise->get_future();
}
}  // namespace cart