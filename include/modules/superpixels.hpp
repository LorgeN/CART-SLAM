#pragma once

#include <opencv2/opencv.hpp>

#include "datasource.hpp"
#include "module.hpp"
#include "modules/superpixels/contourrelaxation/contourrelaxation.hpp"
#include "utils/ui.hpp"

#define CARTSLAM_KEY_SUPERPIXELS "superpixels"
#define CARTSLAM_KEY_SUPERPIXELS_MAX_LABEL "superpixels_max_label"

// Default value for compactness feature weight depends on the type of image.
// Since color images generate a total of 3 cost terms for the separate channels, we choose
// the compactness weight at 3 times the value for grayvalue images to preserve the relative weighting.
#ifdef CARTSLAM_IMAGE_MAKE_GRAYSCALE
#define CARTSLAM_DEFAULT_COMPACTNESS_WEIGHT 0.1
#define CARTSLAM_DEFAULT_DISPARITY_WEIGHT 0.75
#else
#define CARTSLAM_DEFAULT_COMPACTNESS_WEIGHT 0.3
#define CARTSLAM_DEFAULT_DISPARITY_WEIGHT 2.25
#endif

namespace cart {
class SuperPixelModule : public SyncWrapperSystemModule {
   public:
    SuperPixelModule(
        const unsigned int initialIterations = 18,
        const unsigned int iterations = 6,
        const unsigned int blockWidth = 12,
        const unsigned int blockHeight = 12,
        const double directCliqueCost = 0.5,
        const double diagonalCliqueCost = 0.5 / sqrt(2),
        const double compactnessWeight = CARTSLAM_DEFAULT_COMPACTNESS_WEIGHT,
        const double imageWeight = 1.0,
        const double disparityWeight = CARTSLAM_DEFAULT_DISPARITY_WEIGHT);

    system_data_t runInternal(System &system, SystemRunData &data) override;

   private:
    boost::shared_ptr<contour::ContourRelaxation> contourRelaxation;
    boost::mutex mutex;  // Mutex to protect the contour relaxation object
    const unsigned int initialIterations;
    const unsigned int iterations;
    contour::label_t maxLabelId;
};

class SuperPixelVisualizationModule : public SystemModule {
   public:
    SuperPixelVisualizationModule() : SystemModule("SuperPixelVisualization") {
        this->requiresData.push_back(module_dependency_t(CARTSLAM_KEY_SUPERPIXELS));
        this->imageThread = ImageProvider::create("Super Pixels");
    };

    boost::future<system_data_t> run(System &system, SystemRunData &data) override;

   private:
    boost::shared_ptr<ImageProvider> imageThread;
};
}  // namespace cart
