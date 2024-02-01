#ifndef CARTSLAM_FEATURES_HPP
#define CARTSLAM_FEATURES_HPP

#include "datasource.hpp"

#ifdef CARTSLAM_USE_GPU
#include "opencv2/cudafeatures2d.hpp"
#else
#include "opencv2/features2d.hpp"
#endif

namespace cart {
typedef std::function<std::vector<cv::KeyPoint>(const CARTSLAM_IMAGE_TYPE)> FeatureDetector;

std::vector<cv::KeyPoint> detectOrbFeatures(const CARTSLAM_IMAGE_TYPE image);
}  // namespace cart

#endif