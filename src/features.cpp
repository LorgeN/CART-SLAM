#include "features.hpp"

#ifdef CARTSLAM_USE_GPU
#include "opencv2/cudaimgproc.hpp"
#endif

namespace cart {
std::vector<cv::KeyPoint> detectOrbFeatures(const CARTSLAM_IMAGE_TYPE image) {
    std::vector<cv::KeyPoint> keypoints;

    CARTSLAM_IMAGE_TYPE imageCopy;

    try {
        if (image.type() != CV_8UC1) {
#ifdef CARTSLAM_USE_GPU
            cv::cuda::cvtColor(image, imageCopy, cv::COLOR_BGR2GRAY);
#else
            cv::cvtColor(image, imageCopy, cv::COLOR_BGR2GRAY);
#endif
        } else {
            imageCopy = image;
        }
    } catch (const cv::Exception& ex) {
        std::cout << "Error while converting image to grayscale: " << ex.what() << std::endl;
        throw ex;
    }

#ifdef CARTSLAM_USE_GPU
    cv::Ptr<cv::cuda::ORB> orb = cv::cuda::ORB::create(3000);
#else
    cv::Ptr<cv::ORB> orb = cv::ORB::create(3000);
#endif

    try {
        orb->detect(imageCopy, keypoints, cv::noArray());
    } catch (const cv::Exception& ex) {
        std::cout << "Error while detecting features: " << ex.what() << std::endl;
        throw ex;
    }

    return keypoints;
}
}  // namespace cart