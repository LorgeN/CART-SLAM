#ifndef CARTSLAM_PLANESEG_HPP
#define CARTSLAM_PLANESEG_HPP

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
    DisparityPlaneSegmentationVisualizationModule() : SystemModule("PlaneSegmentationVisualization", {CARTSLAM_KEY_PLANES}), imageThread("Planes"){};

    boost::future<MODULE_RETURN_VALUE> run(System& system, SystemRunData& data) override;

   private:
    cart::ImageThread imageThread;
};
}  // namespace cart

#endif  // CARTSLAM_PLANESEG_HPP