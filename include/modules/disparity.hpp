#pragma once

#include <log4cxx/logger.h>

#include <opencv2/cudastereo.hpp>

#include "cartslam.hpp"
#include "datasource.hpp"
#include "utils/ui.hpp"

#define CARTSLAM_KEY_DISPARITY "disparity"

#define CARTSLAM_DISPARITY_INVALID -1

#define SGM_P1(blockSize) (2 * blockSize * blockSize)
#define SGM_P2(blockSize) (8 * blockSize * blockSize)

namespace cart {
typedef int16_t disparity_t;

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
}  // namespace cart