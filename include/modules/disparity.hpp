#ifndef CARTSLAM_DISPARITY_HPP
#define CARTSLAM_DISPARITY_HPP

#include <log4cxx/logger.h>

#include <opencv2/cudastereo.hpp>

#include "cartslam.hpp"
#include "datasource.hpp"
#include "utils/ui.hpp"

#define CARTSLAM_KEY_DISPARITY "disparity"

namespace cart {
class ImageDisparityModule : public SyncWrapperSystemModule {
   public:
    ImageDisparityModule(int minDisparity = 0, int numDisparities = 128) : SyncWrapperSystemModule("ImageDisparity") {
        this->stereoBM = cv::cuda::createStereoSGM(minDisparity, numDisparities);
    };

    MODULE_RETURN_VALUE runInternal(System& system, SystemRunData& data) override;

   private:
    cv::Ptr<cv::cuda::StereoSGM> stereoBM;
};

class ImageDisparityVisualizationModule : public SystemModule {
   public:
    ImageDisparityVisualizationModule() : SystemModule("ImageDisparityVisualization", {CARTSLAM_KEY_DISPARITY}) {
        this->imageThread = ImageProvider::create("Disparity");
    };

    boost::future<MODULE_RETURN_VALUE> run(System& system, SystemRunData& data) override;

   private:
    boost::shared_ptr<ImageProvider> imageThread;
};
}  // namespace cart

#endif