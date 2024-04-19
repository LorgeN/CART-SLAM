#pragma once

#include "opencv2/opencv.hpp"

namespace cart::util {

cv::Vec3b computeColor(float fx, float fy);

// TODO: CUDA __device__ function of the above

}  // namespace cart::util
