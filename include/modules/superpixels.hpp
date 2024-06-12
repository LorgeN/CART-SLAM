#pragma once

#include <opencv2/opencv.hpp>

#include "datasource.hpp"
#include "module.hpp"
#include "modules/superpixels/contourrelaxation/contourrelaxation.hpp"
#include "utils/ui.hpp"
#include "visualization.hpp"

#define CARTSLAM_KEY_SUPERPIXELS "superpixels"
#define CARTSLAM_KEY_SUPERPIXELS_MAX_LABEL "superpixels_max_label"

namespace cart {
class SuperPixelModule : public SyncWrapperSystemModule {
   public:
    SuperPixelModule(
        const cv::Size imageRes,
        const unsigned int initialIterations = 18,
        const unsigned int iterations = 6,
        const unsigned int blockSize = 12,
        const unsigned int resetIterations = 64,
        const double directCliqueCost = 0.25,
        const double diagonalCliqueCost = 0.25 / sqrt(2),
        const double compactnessWeight = 0.05,
        const double progressiveCompactnessCost = 1.0, // Disabled by default
        const double imageWeight = 1.0,
        const double disparityWeight = 1.25);

    system_data_t runInternal(System &system, SystemRunData &data) override;

   private:
    boost::shared_ptr<contour::ContourRelaxation> contourRelaxation;
    boost::mutex mutex;  // Mutex to protect the contour relaxation object
    const unsigned int initialIterations;
    const unsigned int iterations;
    const unsigned int resetIterations;
    const unsigned int blockSize;
    const bool requiresDisparityDerivative;
    contour::label_t maxLabelId;
};

class SuperPixelVisualizationModule : public VisualizationModule {
   public:
    SuperPixelVisualizationModule() : VisualizationModule("SuperPixelVisualization") {
        this->requiresData.push_back(module_dependency_t(CARTSLAM_KEY_SUPERPIXELS));
    };

    bool updateImage(System &system, SystemRunData &data, cv::Mat &image) override;
};
}  // namespace cart
