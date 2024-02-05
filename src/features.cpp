#include "features.hpp"
#include "opencv2/cudaimgproc.hpp"

#define CARTSLAM_OPTION_KEYPOINTS 3000

namespace cart {
std::vector<cv::KeyPoint> detectOrbFeatures(const CARTSLAM_IMAGE_TYPE image) {
    std::vector<cv::KeyPoint> keypoints;

    CARTSLAM_IMAGE_TYPE imageCopy;

    try {
        if (image.type() != CV_8UC1) {
            cv::cuda::cvtColor(image, imageCopy, cv::COLOR_BGR2GRAY);
        } else {
            imageCopy = image;
        }
    } catch (const cv::Exception& ex) {
        std::cout << "Error while converting image to grayscale: " << ex.what() << std::endl;
        throw ex;
    }

    cv::Ptr<cv::cuda::ORB> orb = cv::cuda::ORB::create(CARTSLAM_OPTION_KEYPOINTS);

    try {
        orb->detect(imageCopy, keypoints, cv::noArray());
    } catch (const cv::Exception& ex) {
        std::cout << "Error while detecting features: " << ex.what() << std::endl;
        throw ex;
    }

    return keypoints;
}
}  // namespace cart