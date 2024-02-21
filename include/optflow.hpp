#ifndef CARTSLAM_OPTFLOW_HPP
#define CARTSLAM_OPTFLOW_HPP

#include "datasource.hpp"
#include "opencv2/cudaimgproc.hpp"
#include "opencv2/cudaoptflow.hpp"
#include "opencv2/cudastereo.hpp"
#include "opencv2/opencv.hpp"

namespace cart {
class ImageOpticalFlow {
   public:
    ImageOpticalFlow(cv::cuda::GpuMat flow, cv::cuda::GpuMat cost) : flow(flow), cost(cost) {}

    cv::cuda::GpuMat flow;
    cv::cuda::GpuMat cost;
};

cv::Ptr<cv::cuda::NvidiaOpticalFlow_2_0> createOpticalFlow(cv::cuda::Stream &stream);

ImageOpticalFlow detectOpticalFlow(const CARTSLAM_IMAGE_TYPE input, const CARTSLAM_IMAGE_TYPE reference, cv::Ptr<cv::cuda::NvidiaOpticalFlow_2_0> &flow);

cv::Mat drawOpticalFlow(const ImageOpticalFlow &imageFlow, cv::Ptr<cv::cuda::NvidiaOpticalFlow_2_0> &opticalFlow, cv::cuda::Stream &stream);

}  // namespace cart

#endif  // CARTSLAM_OPTFLOW_HPP