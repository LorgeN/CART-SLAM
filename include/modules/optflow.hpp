#pragma once

#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaoptflow.hpp>
#include <opencv2/cudastereo.hpp>
#include <opencv2/opencv.hpp>

#include "cartslam.hpp"
#include "datasource.hpp"
#include "utils/ui.hpp"

#define CARTSLAM_KEY_OPTFLOW "optflow"

namespace cart {
cv::Ptr<cv::cuda::NvidiaOpticalFlow_2_0> createOpticalFlow(cv::cuda::Stream &stream);

struct image_optical_flow_t {
    cv::cuda::GpuMat flow;
    cv::cuda::GpuMat cost;
};

cv::Mat drawOpticalFlow(const image_optical_flow_t &imageFlow, cv::Ptr<cv::cuda::NvidiaOpticalFlow_2_0> &flow, cv::cuda::Stream &stream);

class ImageOpticalFlowModule : public SyncWrapperSystemModule {
   public:
    ImageOpticalFlowModule() : SyncWrapperSystemModule("ImageOpticalFlow"){};

    system_data_t runInternal(System &system, SystemRunData &data) override;

   private:
    image_optical_flow_t detectOpticalFlow(
        const CARTSLAM_IMAGE_TYPE input,
        const CARTSLAM_IMAGE_TYPE reference,
        cv::InputArray hint,
        cv::Ptr<cv::cuda::NvidiaOpticalFlow_2_0> opticalFlow,
        cv::cuda::Stream &stream);

    // HW-accel optical flow is not thread-safe, ref; https://docs.opencv.org/4.x/d5/d26/classcv_1_1cuda_1_1NvidiaHWOpticalFlow.html#a9c065a6ed6ed6d9a5531f191ac7a366d
    boost::shared_mutex flowMutex;
};

class ImageOpticalFlowVisualizationModule : public SystemModule {
   public:
    ImageOpticalFlowVisualizationModule(int points = 10) : SystemModule("ImageOpticalFlowVisualization", {CARTSLAM_KEY_OPTFLOW}) {
        this->imageThread = ImageProvider::create("Optical Flow");
        this->visualizationPoints = getRandomPoints(points, cv::Size(CARTSLAM_IMAGE_RES_X, CARTSLAM_IMAGE_RES_Y));
    };

    boost::future<system_data_t> run(System &system, SystemRunData &data) override;

   private:
    std::vector<cv::Point2i> visualizationPoints;
    boost::shared_ptr<ImageProvider> imageThread;
};
}  // namespace cart
