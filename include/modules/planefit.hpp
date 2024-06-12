#pragma once

#include "depth.hpp"
#include "disparity.hpp"
#include "module.hpp"
#include "superpixels.hpp"
#include "visualization.hpp"

#define CARTSLAM_KEY_PLANES_EQ "planes_eq"

namespace cart {

class SuperPixelPlaneFitModule : public SyncWrapperSystemModule {
   public:
    SuperPixelPlaneFitModule() : SyncWrapperSystemModule("PlaneFit") {
        this->requiresData.push_back(module_dependency_t(CARTSLAM_KEY_DEPTH));
        this->requiresData.push_back(module_dependency_t(CARTSLAM_KEY_SUPERPIXELS));
        this->requiresData.push_back(module_dependency_t(CARTSLAM_KEY_SUPERPIXELS_MAX_LABEL));
        this->requiresData.push_back(module_dependency_t(CARTSLAM_KEY_DISPARITY_DERIVATIVE));

        this->providesData.push_back(CARTSLAM_KEY_PLANES_EQ);
    };

    system_data_t runInternal(System& system, SystemRunData& data) override;
};

class SuperPixelPlaneFitVisualizationModule : public VisualizationModule {
   public:
    SuperPixelPlaneFitVisualizationModule() : VisualizationModule("SuperPixelPlaneFitVisualization") {
        this->requiresData.push_back(module_dependency_t(CARTSLAM_KEY_PLANES_EQ));
        this->requiresData.push_back(module_dependency_t(CARTSLAM_KEY_SUPERPIXELS));
        this->requiresData.push_back(module_dependency_t(CARTSLAM_KEY_SUPERPIXELS_MAX_LABEL));
    };

    bool updateImage(System& system, SystemRunData& data, cv::Mat& image) override;
};
}  // namespace cart