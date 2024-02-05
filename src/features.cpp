#include "features.hpp"

#define CARTSLAM_OPTION_KEYPOINTS 10000

const cv::Ptr<cv::cuda::ORB> orb = cv::cuda::ORB::create(CARTSLAM_OPTION_KEYPOINTS);

namespace cart {
ImageFeatures detectOrbFeatures(const CARTSLAM_IMAGE_TYPE image, cv::cuda::Stream &stream) {
    std::promise<ImageFeatures*> promise;

    cv::cuda::GpuMat descriptors;
    cv::cuda::GpuMat keypoints;

    orb->detectAndComputeAsync(image, cv::noArray(), keypoints, descriptors, false, stream);

    stream.waitForCompletion();

    std::vector<cv::KeyPoint> keypointsHost;
    orb->convert(keypoints, keypointsHost);

    return ImageFeatures(keypointsHost, descriptors);
}
}  // namespace cart