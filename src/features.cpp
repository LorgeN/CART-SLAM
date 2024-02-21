#include "features.hpp"


#define CARTSLAM_OPTION_KEYPOINTS 10000

const cv::Ptr<cv::cuda::ORB> orb = cv::cuda::ORB::create(CARTSLAM_OPTION_KEYPOINTS);

namespace cart {
MODULE_RETURN_VALUE ImageFeatureDetectorModule::runInternal(System &system, SystemRunData &data) {
    cv::cuda::Stream stream;
    ImageFeatureDetectorVisitor visitor(this->detector, stream);
    return MODULE_RETURN_VALUE_PAIR("features", visitor(data.getDataElement()));
}

void *ImageFeatureDetectorVisitor::visitStereo(StereoDataElement *element) {
    auto leftFeatures = this->detector(element->left, this->stream);
    auto rightFeatures = this->detector(element->right, this->stream);

    return new std::pair<ImageFeatures, ImageFeatures>(leftFeatures, rightFeatures);
}

ImageFeatures detectOrbFeatures(const CARTSLAM_IMAGE_TYPE image, cv::cuda::Stream &stream) {
    cv::cuda::GpuMat descriptors;
    cv::cuda::GpuMat keypoints;

    orb->detectAndComputeAsync(image, cv::noArray(), keypoints, descriptors, false, stream);

    stream.waitForCompletion();

    std::vector<cv::KeyPoint> keypointsHost;
    orb->convert(keypoints, keypointsHost);

    return ImageFeatures(keypointsHost, descriptors);
}
}  // namespace cart