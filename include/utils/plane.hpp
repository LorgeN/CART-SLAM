#pragma once

#include <opencv2/core.hpp>
#include <vector>

namespace cart::util {
cv::Vec4d segmentPlane(
    const std::vector<cv::Point3d> &points,
    const double distThreshold = 0.01,
    const int ransacN = 4,
    const int iters = 100,
    const double probability = 0.99999999);
}