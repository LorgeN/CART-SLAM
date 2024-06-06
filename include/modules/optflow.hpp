#pragma once

#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaoptflow.hpp>
#include <opencv2/cudastereo.hpp>
#include <opencv2/opencv.hpp>

#include "datasource.hpp"
#include "module.hpp"
#include "utils/ui.hpp"
#include "visualization.hpp"

#define CARTSLAM_KEY_OPTFLOW "optflow"

namespace cart {

typedef int16_t optical_flow_t;

cv::Ptr<cv::cuda::NvidiaOpticalFlow_2_0> createOpticalFlow(const cv::Size imageRes, cv::cuda::Stream &stream);

cv::Mat drawOpticalFlow(const cv::cuda::GpuMat &imageFlow, cv::Ptr<cv::cuda::NvidiaOpticalFlow_2_0> &flow, cv::cuda::Stream &stream);

class ImageOpticalFlowModule : public SyncWrapperSystemModule {
   public:
    ImageOpticalFlowModule(const cv::Size imageRes);

    system_data_t runInternal(System &system, SystemRunData &data) override;

   private:
    void detectOpticalFlow(const image_t input,
                           const image_t reference,
                           cv::Ptr<cv::cuda::NvidiaOpticalFlow_2_0> opticalFlow,
                           cv::cuda::Stream &stream,
                           cv::cuda::GpuMat &flow);

    // HW-accel optical flow is not thread-safe, ref; https://docs.opencv.org/4.x/d5/d26/classcv_1_1cuda_1_1NvidiaHWOpticalFlow.html#a9c065a6ed6ed6d9a5531f191ac7a366d
    boost::shared_mutex flowMutex;
    cv::Ptr<cv::cuda::NvidiaOpticalFlow_2_0> opticalFlow;
    cv::cuda::Stream stream;
};

class ImageOpticalFlowVisualizationModule : public VisualizationModule {
   public:
    ImageOpticalFlowVisualizationModule(const cv::Size imageRes, int points = 10) : VisualizationModule("ImageOpticalFlowVisualization") {
        this->requiresData.push_back(module_dependency_t(CARTSLAM_KEY_OPTFLOW));
        this->visualizationPoints = getRandomPoints(points, imageRes);
    };

    bool updateImage(System &system, SystemRunData &data, cv::Mat &image) override;

   private:
    std::vector<cv::Point2i> visualizationPoints;
};
}  // namespace cart
