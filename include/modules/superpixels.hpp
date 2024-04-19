#pragma once

#include <opencv2/opencv.hpp>

#include "cartslam.hpp"
#include "datasource.hpp"
#include "utils/ui.hpp"

#define CARTSLAM_KEY_SUPERPIXELS "superpixels"

namespace cart {
struct image_super_pixels_t {
};

class SuperPixelModule : public SyncWrapperSystemModule {
   public:
    SuperPixelModule() : SyncWrapperSystemModule("SuperPixelDetect"){};

    system_data_t runInternal(System &system, SystemRunData &data) override;
};

class SuperPixelVisualizationModule : public SystemModule {
   public:
    SuperPixelVisualizationModule() : SystemModule("SuperPixelVisualization") {
        this->imageThread = ImageProvider::create("Super Pixels");
    };

    boost::future<system_data_t> run(System &system, SystemRunData &data) override;

   private:
    boost::shared_ptr<ImageProvider> imageThread;
};
}  // namespace cart
