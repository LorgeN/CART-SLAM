#pragma once

#include <log4cxx/logger.h>

#include "cartslam.hpp"
#include "modules/disparity.hpp"
#include "utils/ui.hpp"

#define CARTSLAM_KEY_PLANES "planes"

namespace cart {
enum Plane {
    HORIZONTAL = 0,
    VERTICAL = 1
};

class DisparityPlaneSegmentationModule : public SyncWrapperSystemModule {
   public:
    DisparityPlaneSegmentationModule(const int xStep = 32, const int yStep = 32) : SyncWrapperSystemModule("PlaneSegmentation", {CARTSLAM_KEY_DISPARITY}), xStep(xStep), yStep(yStep){};

    MODULE_RETURN_VALUE runInternal(System& system, SystemRunData& data) override;

   private:
    const int xStep;
    const int yStep;
};

class DisparityPlaneSegmentationVisualizationModule : public SystemModule {
   public:
    DisparityPlaneSegmentationVisualizationModule() : SystemModule("PlaneSegmentationVisualization", {CARTSLAM_KEY_PLANES}) {
        this->imageThread = ImageProvider::create("Plane Segmentation");
        this->histThread = ImageProvider::create("Plane Segmentation Histogram");
    };
    
    boost::future<MODULE_RETURN_VALUE> run(System& system, SystemRunData& data) override;

   private:
    boost::shared_ptr<ImageProvider> imageThread;
    boost::shared_ptr<ImageProvider> histThread;
};
}  // namespace cart
