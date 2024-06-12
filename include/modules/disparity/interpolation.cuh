#pragma once

#include <log4cxx/logger.h>

#include <opencv2/core.hpp>

namespace cart::disparity {

void interpolate(log4cxx::LoggerPtr logger, cv::cuda::GpuMat& disparity, cv::cuda::Stream& stream, int radius, int iterations, int minDisparity, int maxDisparity);

}  // namespace cart::disparity