#pragma once

#include <opencv2/opencv.hpp>
#include <vector>

#include "contourrelaxation/constants.hpp"

namespace cart::contour {

void computeBoundaryOverlay(log4cxx::LoggerPtr logger, cv::cuda::GpuMat bgrImage, cv::cuda::GpuMat labelImage, cv::cuda::GpuMat &out_boundaryOverlay);

}  // namespace cart::contour