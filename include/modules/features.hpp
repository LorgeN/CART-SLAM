#pragma once

#include <log4cxx/logger.h>

#include <opencv2/cudafeatures2d.hpp>

#include "datasource.hpp"
#include "module.hpp"
#include "utils/ui.hpp"
#include "visualization.hpp"

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

typedef std::function<ImageFeatures(const image_t, cv::cuda::Stream&, log4cxx::LoggerPtr)> FeatureDetector;

ImageFeatures detectOrbFeatures(const image_t image, cv::cuda::Stream& stream, log4cxx::LoggerPtr logger);

class ImageFeatureDetectorModule : public SyncWrapperSystemModule {
   public:
    ImageFeatureDetectorModule(FeatureDetector detector) : SyncWrapperSystemModule("ImageFeatureDetector"), detector(detector) {
        this->providesData.push_back(CARTSLAM_KEY_FEATURES);
    };

    system_data_t runInternal(System& system, SystemRunData& data) override;

   private:
    const FeatureDetector detector;
};

class ImageFeatureVisualizationModule : public VisualizationModule {
   public:
    ImageFeatureVisualizationModule() : VisualizationModule("ImageFeatureVisualization") {
        this->requiresData.push_back(module_dependency_t(CARTSLAM_KEY_FEATURES));
    };

    bool updateImage(System& system, SystemRunData& data, cv::Mat& image) override;
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