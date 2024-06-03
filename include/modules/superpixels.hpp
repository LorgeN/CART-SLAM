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
        const unsigned int initialIterations = 18,
        const unsigned int iterations = 8,
        const unsigned int blockWidth = 12,
        const unsigned int blockHeight = 12,
        const double directCliqueCost = 0.2,
        const double diagonalCliqueCost = 0.2 / sqrt(2),
        const double compactnessWeight = 0.05,
        const double imageWeight = 1.0,
        const double disparityWeight = 1.5);

    system_data_t runInternal(System &system, SystemRunData &data) override;

   private:
    boost::shared_ptr<contour::ContourRelaxation> contourRelaxation;
    boost::mutex mutex;  // Mutex to protect the contour relaxation object
    const unsigned int initialIterations;
    const unsigned int iterations;
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
