#ifndef CARTSLAM_FEATURES_HPP
#define CARTSLAM_FEATURES_HPP

#include <log4cxx/logger.h>

#include "cartslam.hpp"
#include "datasource.hpp"
#include "opencv2/cudafeatures2d.hpp"

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

    MODULE_RETURN_VALUE runInternal(System& system, SystemRunData& data) override;

   private:
    const FeatureDetector detector;
};

class ImageFeatureDetectorVisitor : public DataElementVisitor<void*> {
   public:
    ImageFeatureDetectorVisitor(const FeatureDetector& detector, cv::cuda::Stream& stream, log4cxx::LoggerPtr logger) : detector(detector), stream(stream), logger(logger){};
    void* visitStereo(StereoDataElement* element) override;

   private:
    cv::cuda::Stream stream;
    const FeatureDetector detector;
    log4cxx::LoggerPtr logger;
};
}  // namespace cart

#endif