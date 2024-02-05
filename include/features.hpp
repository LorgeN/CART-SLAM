#ifndef CARTSLAM_FEATURES_HPP
#define CARTSLAM_FEATURES_HPP

#include <future>

#include "datasource.hpp"
#include "opencv2/cudafeatures2d.hpp"

namespace cart {
class ImageFeatures {
   public:
    ImageFeatures(std::vector<cv::KeyPoint> keypoints, cv::cuda::GpuMat descriptors) {
        this->keypoints = keypoints;
        this->descriptors = descriptors;
    }

    std::vector<cv::KeyPoint> keypoints;
    cv::cuda::GpuMat descriptors;
};

typedef std::function<ImageFeatures(const CARTSLAM_IMAGE_TYPE, cv::cuda::Stream)> FeatureDetector;

ImageFeatures detectOrbFeatures(const CARTSLAM_IMAGE_TYPE image, cv::cuda::Stream stream);
}  // namespace cart

#endif