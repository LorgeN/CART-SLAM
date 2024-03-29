#include "modules/features.hpp"

#include <opencv2/imgproc/imgproc.hpp>

#define CARTSLAM_OPTION_KEYPOINTS 10000

const cv::Ptr<cv::cuda::ORB> orb = cv::cuda::ORB::create(CARTSLAM_OPTION_KEYPOINTS);

namespace cart {
MODULE_RETURN_VALUE ImageFeatureDetectorModule::runInternal(System &system, SystemRunData &data) {
    LOG4CXX_DEBUG(this->logger, "Running ImageFeatureDetectorModule");
    cv::cuda::Stream stream;
    ImageFeatureDetectorVisitor visitor(this->detector, stream, this->logger);
    return MODULE_RETURN_VALUE_PAIR("features", visitor(data.dataElement));
}

void *ImageFeatureDetectorVisitor::visitStereo(StereoDataElement *element) {
    auto leftFeatures = this->detector(element->left, this->stream, this->logger);
    auto rightFeatures = this->detector(element->right, this->stream, this->logger);

    return new std::pair<ImageFeatures, ImageFeatures>(leftFeatures, rightFeatures);
}

boost::future<MODULE_RETURN_VALUE> ImageFeatureVisualizationModule::run(System &system, SystemRunData &data) {
    auto promise = boost::make_shared<boost::promise<MODULE_RETURN_VALUE>>();

    boost::asio::post(system.threadPool, [this, promise, &system, &data]() {
        auto features = data.getData<std::pair<ImageFeatures, ImageFeatures>>("features");

        cv::cuda::Stream stream;
        cv::Mat images[2];

        // TODO: Support for non-stereo images
        cart::StereoDataElement *stereoData = static_cast<cart::StereoDataElement *>(data.dataElement);

        stereoData->left.download(images[0], stream);
        stereoData->right.download(images[1], stream);

        stream.waitForCompletion();

        cv::drawKeypoints(images[0], features->first.keypoints, images[0]);
        cv::drawKeypoints(images[1], features->second.keypoints, images[1]);

        cv::Mat concatRes;
        cv::hconcat(images[0], images[1], concatRes);

        this->imageThread.setImageIfLater(concatRes, data.id);

        promise->set_value(MODULE_NO_RETURN_VALUE);
    });

    return promise->get_future();
}

ImageFeatures detectOrbFeatures(const CARTSLAM_IMAGE_TYPE image, cv::cuda::Stream &stream, log4cxx::LoggerPtr logger) {
    cv::cuda::GpuMat descriptors;
    cv::cuda::GpuMat keypoints;

    LOG4CXX_DEBUG(logger, "Detecting ORB features");
    orb->detectAndComputeAsync(image, cv::noArray(), keypoints, descriptors, false, stream);

    stream.waitForCompletion();
    LOG4CXX_DEBUG(logger, "Detected ORB features");

    std::vector<cv::KeyPoint> keypointsHost;
    LOG4CXX_DEBUG(logger, "Converting keypoints to host");
    orb->convert(keypoints, keypointsHost);
    LOG4CXX_DEBUG(logger, "Converted keypoints to host");

    LOG4CXX_INFO(logger, "Found " << keypointsHost.size() << " keypoints in image");

    return ImageFeatures(keypointsHost, descriptors);
}
}  // namespace cart