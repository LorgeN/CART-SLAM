#ifndef CARTSLAM_OPTFLOW_HPP
#define CARTSLAM_OPTFLOW_HPP

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

class ImageOpticalFlow {
   public:
    ImageOpticalFlow(cv::cuda::GpuMat flow, cv::cuda::GpuMat cost) : flow(flow), cost(cost){};

    cv::cuda::GpuMat flow;
    cv::cuda::GpuMat cost;
};

ImageOpticalFlow detectOpticalFlow(const CARTSLAM_IMAGE_TYPE input, const CARTSLAM_IMAGE_TYPE reference, const cv::Ptr<cv::cuda::NvidiaOpticalFlow_2_0> &flow);

cv::Mat drawOpticalFlow(const ImageOpticalFlow &imageFlow, cv::Ptr<cv::cuda::NvidiaOpticalFlow_2_0> &flow, cv::cuda::Stream &stream);

class ImageOpticalFlowModule : public SyncWrapperSystemModule {
   public:
    ImageOpticalFlowModule() : SyncWrapperSystemModule("ImageOpticalFlow"){};

    MODULE_RETURN_VALUE runInternal(System &system, SystemRunData &data) override;
};

class ImageOpticalFlowVisualizationModule : public SystemModule {
   public:
    ImageOpticalFlowVisualizationModule() : SystemModule("ImageOpticalFlowVisualization", {CARTSLAM_KEY_OPTFLOW}), imageThread("Optical Flow"){};

    boost::future<MODULE_RETURN_VALUE> run(System &system, SystemRunData &data) override;

   private:
    cart::ImageThread imageThread;
};

class ImageOpticalFlowVisitor : public DataElementVisitor<void *> {
   public:
    ImageOpticalFlowVisitor(SystemRunData &data, cv::Ptr<cv::cuda::NvidiaOpticalFlow_2_0> &flow, log4cxx::LoggerPtr logger) : data(data), logger(logger), flow(flow){};
    void *visitStereo(StereoDataElement *element) override;

   private:
    SystemRunData &data;
    const log4cxx::LoggerPtr logger;
    const cv::Ptr<cv::cuda::NvidiaOpticalFlow_2_0> &flow;
};

}  // namespace cart

#endif  // CARTSLAM_OPTFLOW_HPP