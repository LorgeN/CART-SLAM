#pragma once

#include <opencv2/opencv.hpp>

#include "cartslam.hpp"
#include "datasource.hpp"
#include "modules/superpixels/contourrelaxation/contourrelaxation.hpp"
#include "utils/ui.hpp"

#define CARTSLAM_KEY_SUPERPIXELS "superpixels"

// Default value for compactness feature weight depends on the type of image.
// Since color images generate a total of 3 cost terms for the separate channels, we choose
// the compactness weight at 3 times the value for grayvalue images to preserve the relative weighting.
#ifdef CARTSLAM_IMAGE_MAKE_GRAYSCALE
#define CARTSLAM_DEFAULT_COMPACTNESS_WEIGHT 0.015
#else
#define CARTSLAM_DEFAULT_COMPACTNESS_WEIGHT 0.045
#endif

namespace cart {
class SuperPixelModule : public SyncWrapperSystemModule {
   public:
    SuperPixelModule(
        const unsigned int initialIterations = 20,
        const unsigned int blockWidth = 15,
        const unsigned int blockHeight = 15,
        const double directCliqueCost = 0.3,
        const double compactnessWeight = CARTSLAM_DEFAULT_COMPACTNESS_WEIGHT);

    system_data_t runInternal(System &system, SystemRunData &data) override;

   private:
    boost::shared_ptr<contour::ContourRelaxation> contourRelaxation;
    const unsigned int initialIterations;
    boost::mutex mutex;  // Mutex to protect the contour relaxation object
};

class SuperPixelVisualizationModule : public SystemModule {
   public:
    SuperPixelVisualizationModule() : SystemModule("SuperPixelVisualization", {CARTSLAM_KEY_SUPERPIXELS}) {
        this->imageThread = ImageProvider::create("Super Pixels");
    };

    boost::future<system_data_t> run(System &system, SystemRunData &data) override;

   private:
    boost::shared_ptr<ImageProvider> imageThread;
};
}  // namespace cart
