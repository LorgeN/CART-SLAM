#include "features.hpp"

#include "opencv2/cudaimgproc.hpp"

#define CARTSLAM_OPTION_KEYPOINTS 10000

const cv::Ptr<cv::cuda::ORB> orb = cv::cuda::ORB::create(CARTSLAM_OPTION_KEYPOINTS);

namespace cart {
ImageFeatures detectOrbFeatures(const CARTSLAM_IMAGE_TYPE image, cv::cuda::Stream stream) {
    CARTSLAM_IMAGE_TYPE imageCopy;

    std::promise<ImageFeatures*> promise;

    if (image.type() != CV_8UC1) {
        cv::cuda::cvtColor(image, imageCopy, cv::COLOR_BGR2GRAY, 0, stream);
    } else {
        imageCopy = image;
    }

    cv::cuda::GpuMat descriptors;
    cv::cuda::GpuMat keypoints;

    orb->detectAndComputeAsync(imageCopy, cv::noArray(), keypoints, descriptors, false, stream);

    stream.waitForCompletion();

    std::vector<cv::KeyPoint> keypointsHost;
    orb->convert(keypoints, keypointsHost);

    return ImageFeatures(keypointsHost, descriptors);
}
}  // namespace cart