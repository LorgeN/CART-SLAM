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
    ImageDisparityModule(int numDisparities = 64,
                         int blockSize = 19) : SyncWrapperSystemModule("ImageDisparity") {
        this->stereoBM = cv::cuda::createStereoBM(numDisparities, blockSize);
    };

    MODULE_RETURN_VALUE runInternal(System& system, SystemRunData& data) override;

   private:
    cv::Ptr<cv::cuda::StereoBM> stereoBM;
};

class ImageDisparityVisualizationModule : public SystemModule {
   public:
    ImageDisparityVisualizationModule() : SystemModule("ImageDisparityVisualization", {CARTSLAM_KEY_DISPARITY}), imageThread("Disparity"){};

    boost::future<MODULE_RETURN_VALUE> run(System& system, SystemRunData& data) override;

   private:
    cart::ImageThread imageThread;
};
}  // namespace cart

#endif