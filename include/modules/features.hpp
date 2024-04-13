#pragma once

#include <log4cxx/logger.h>

#include <opencv2/cudafeatures2d.hpp>

#include "cartslam.hpp"
#include "datasource.hpp"
#include "utils/ui.hpp"

#define CARTSLAM_OPTION_KEYPOINTS 5000

#define CARTSLAM_KEY_FEATURES "features"

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

typedef std::function<ImageFeatures(const CARTSLAM_IMAGE_TYPE, cv::cuda::Stream&, log4cxx::LoggerPtr)> FeatureDetector;

ImageFeatures detectOrbFeatures(const CARTSLAM_IMAGE_TYPE image, cv::cuda::Stream& stream, log4cxx::LoggerPtr logger);

class ImageFeatureDetectorModule : public SyncWrapperSystemModule {
   public:
    ImageFeatureDetectorModule(FeatureDetector detector) : SyncWrapperSystemModule("ImageFeatureDetector"), detector(detector){};

    module_result_t runInternal(System& system, SystemRunData& data) override;

   private:
    const FeatureDetector detector;
};

class ImageFeatureVisualizationModule : public SystemModule {
   public:
    ImageFeatureVisualizationModule() : SystemModule("ImageFeatureVisualization", {CARTSLAM_KEY_FEATURES}) {
        this->imageThread = ImageProvider::create("Features");
    };

    boost::future<module_result_t> run(System& system, SystemRunData& data) override;

   private:
    boost::shared_ptr<ImageProvider> imageThread;
};

class ImageFeatureDetectorVisitor : public DataElementVisitor<void*> {
   public:
    ImageFeatureDetectorVisitor(const FeatureDetector& detector, cv::cuda::Stream& stream, log4cxx::LoggerPtr logger) : detector(detector), stream(stream), logger(logger){};
    void* visitStereo(boost::shared_ptr<StereoDataElement> element) override;

   private:
    cv::cuda::Stream& stream;
    const FeatureDetector detector;
    log4cxx::LoggerPtr logger;
};
}  // namespace cart