#pragma once

#include <log4cxx/logger.h>

#include <opencv2/cudastereo.hpp>

#include "cartslam.hpp"
#include "datasource.hpp"
#include "modules/disparity/interpolation.cuh"
#include "utils/ui.hpp"

#define CARTSLAM_KEY_DISPARITY "disparity"
#define CARTSLAM_KEY_DISPARITY_DERIVATIVE "disparity_derivative"
#define CARTSLAM_KEY_DISPARITY_DERIVATIVE_HISTOGRAM "disparity_derivative_histogram"

#define CARTSLAM_DISPARITY_INVALID -1

#define SGM_P1(blockSize) (2 * blockSize * blockSize)
#define SGM_P2(blockSize) (8 * blockSize * blockSize)

namespace cart {
    
typedef int16_t disparity_t;
typedef int16_t derivative_t;

class ImageDisparityModule : public SyncWrapperSystemModule {
   public:
    ImageDisparityModule(int minDisparity = 1, int numDisparities = 255, int blockSize = 3, int smoothingRadius = -1, int smoothingIterations = 5)
        : SyncWrapperSystemModule("ImageDisparity"), smoothingRadius(smoothingRadius), smoothingIterations(smoothingIterations) {
        this->stereoSGM = cv::cuda::createStereoSGM(minDisparity, numDisparities, SGM_P1(blockSize), SGM_P2(blockSize), 5);
        this->stereoSGM->setBlockSize(blockSize);
        this->stereoSGM->setSpeckleWindowSize(100);
        this->stereoSGM->setSpeckleRange(16);
    };

    system_data_t runInternal(System& system, SystemRunData& data) override;

   private:
    cv::Ptr<cv::cuda::StereoSGM> stereoSGM;
    int smoothingRadius;
    int smoothingIterations;
};

#ifdef CARTSLAM_ZED
class ZEDImageDisparityModule : public SyncWrapperSystemModule {
   public:
    ZEDImageDisparityModule(int smoothingRadius = -1, int smoothingIterations = 5) : SyncWrapperSystemModule("ZEDImageDisparity"), smoothingRadius(smoothingRadius), smoothingIterations(smoothingIterations){};

    system_data_t runInternal(System& system, SystemRunData& data) override;

   private:
    int smoothingRadius;
    int smoothingIterations;
};
#endif  // CARTSLAM_ZED

class ImageDisparityVisualizationModule : public SystemModule {
   public:
    ImageDisparityVisualizationModule() : SystemModule("ImageDisparityVisualization", {CARTSLAM_KEY_DISPARITY}) {
        this->imageThread = ImageProvider::create("Disparity");
    };

    boost::future<system_data_t> run(System& system, SystemRunData& data) override;

   private:
    boost::shared_ptr<ImageProvider> imageThread;
};

class ImageDisparityDerivativeModule : public SyncWrapperSystemModule {
   public:
    ImageDisparityDerivativeModule() : SyncWrapperSystemModule("ImageDisparityDerivative", {CARTSLAM_KEY_DISPARITY}){};

    system_data_t runInternal(System& system, SystemRunData& data) override;
};

class ImageDisparityDerivativeVisualizationModule : public SyncWrapperSystemModule {
   public:
    ImageDisparityDerivativeVisualizationModule() : SyncWrapperSystemModule("ImageDisparityDerivativeVisualization", {CARTSLAM_KEY_DISPARITY_DERIVATIVE}) {
        this->imageThread = ImageProvider::create("Disparity Derivative");
    };

    system_data_t runInternal(System& system, SystemRunData& data) override;

   private:
    boost::shared_ptr<ImageProvider> imageThread;
};
}  // namespace cart