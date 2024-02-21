#ifndef CARTSLAM_FEATURES_HPP
#define CARTSLAM_FEATURES_HPP

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

typedef std::function<ImageFeatures(const CARTSLAM_IMAGE_TYPE, cv::cuda::Stream&)> FeatureDetector;

ImageFeatures detectOrbFeatures(const CARTSLAM_IMAGE_TYPE image, cv::cuda::Stream& stream);

class ImageFeatureDetectorModule : public SyncWrapperSystemModule {
   public:
    ImageFeatureDetectorModule(FeatureDetector detector) : detector(detector) {}
    MODULE_RETURN_VALUE runInternal(System& system, SystemRunData& data) override;

   private:
    FeatureDetector detector;
};

class ImageFeatureDetectorVisitor : public DataElementVisitor<void*> {
   public:
    ImageFeatureDetectorVisitor(FeatureDetector& detector, cv::cuda::Stream& stream) : detector(detector), stream(stream) {}
    void* visitStereo(StereoDataElement* element) override;

   private:
    cv::cuda::Stream stream;
    FeatureDetector detector;
};
}  // namespace cart

#endif