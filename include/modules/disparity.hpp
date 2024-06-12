#pragma once

#include <log4cxx/logger.h>

#include <opencv2/cudastereo.hpp>

#include "datasource.hpp"
#include "module.hpp"
#include "modules/disparity/interpolation.cuh"
#include "modules/visualization.hpp"
#include "utils/ui.hpp"

#define CARTSLAM_KEY_DISPARITY "disparity"
#define CARTSLAM_KEY_DISPARITY_DERIVATIVE "disparity_derivative"
#define CARTSLAM_KEY_DISPARITY_DERIVATIVE_HISTOGRAM "disparity_derivative_histogram"

#define CARTSLAM_DISPARITY_INVALID (-32768)

namespace cart {

typedef int16_t disparity_t;
typedef int16_t derivative_t;

class ImageDisparityModule : public SyncWrapperSystemModule {
   public:
    ImageDisparityModule(const cv::Size imageRes, int minDisparity = 0, int numDisparities = 256, int blockSize = 5, int smoothingRadius = -1, int smoothingIterations = 5)
        : SyncWrapperSystemModule("ImageDisparity"), smoothingRadius(smoothingRadius), smoothingIterations(smoothingIterations), 
        minDisparity((minDisparity + 4) * 16), maxDisparity(imageRes.width) {
        this->providesData.push_back(CARTSLAM_KEY_DISPARITY);

        this->stereoSGM = cv::cuda::createStereoSGM(minDisparity, numDisparities);
        this->stereoSGM->setUniquenessRatio(5);
        this->stereoSGM->setBlockSize(blockSize);
        this->stereoSGM->setSpeckleWindowSize(64);
        this->stereoSGM->setSpeckleRange(2);
    };

    system_data_t runInternal(System& system, SystemRunData& data) override;

   private:
    cv::Ptr<cv::cuda::StereoSGM> stereoSGM;
    int smoothingRadius;
    int smoothingIterations;
    const int minDisparity;
    const int maxDisparity;
};

#ifdef CARTSLAM_ZED
class ZEDImageDisparityModule : public SyncWrapperSystemModule {
   public:
    ZEDImageDisparityModule(int smoothingRadius = -1, int smoothingIterations = 5) : SyncWrapperSystemModule("ZEDImageDisparity"), smoothingRadius(smoothingRadius), smoothingIterations(smoothingIterations) {
        this->providesData.push_back(CARTSLAM_KEY_DISPARITY);
    };

    system_data_t runInternal(System& system, SystemRunData& data) override;

   private:
    int smoothingRadius;
    int smoothingIterations;
};
#endif  // CARTSLAM_ZED

class ImageDisparityVisualizationModule : public VisualizationModule {
   public:
    ImageDisparityVisualizationModule() : VisualizationModule("ImageDisparityVisualization") {
        this->requiresData.push_back(module_dependency_t(CARTSLAM_KEY_DISPARITY));
    };

    bool updateImage(System& system, SystemRunData& data, cv::Mat& image) override;
};

class ImageDisparityDerivativeModule : public SyncWrapperSystemModule {
   public:
    ImageDisparityDerivativeModule() : SyncWrapperSystemModule("ImageDisparityDerivative") {
        this->requiresData.push_back(module_dependency_t(CARTSLAM_KEY_DISPARITY));
        this->providesData.push_back(CARTSLAM_KEY_DISPARITY_DERIVATIVE);
        this->providesData.push_back(CARTSLAM_KEY_DISPARITY_DERIVATIVE_HISTOGRAM);
    };

    system_data_t runInternal(System& system, SystemRunData& data) override;
};

class ImageDisparityDerivativeVisualizationModule : public VisualizationModule {
   public:
    ImageDisparityDerivativeVisualizationModule() : VisualizationModule("ImageDisparityDerivativeVisualization") {
        this->requiresData.push_back(module_dependency_t(CARTSLAM_KEY_DISPARITY_DERIVATIVE));
    };

    bool updateImage(System& system, SystemRunData& data, cv::Mat& image) override;
};
}  // namespace cart