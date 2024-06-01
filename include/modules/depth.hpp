#pragma once

#include "module.hpp"
#include "disparity.hpp"
#include "visualization.hpp"

#define CARTSLAM_KEY_DEPTH "depth"

namespace cart {

class DepthModule : public SyncWrapperSystemModule {
   public:
    DepthModule() : SyncWrapperSystemModule("Depth") {
        this->requiresData.push_back(module_dependency_t(CARTSLAM_KEY_DISPARITY));
        this->providesData.push_back(CARTSLAM_KEY_DEPTH);
    };

    system_data_t runInternal(System& system, SystemRunData& data) override;
};

class DepthVisualizationModule : public VisualizationModule {
   public:
    DepthVisualizationModule() : VisualizationModule("DepthVisualization") {
        this->requiresData.push_back(module_dependency_t(CARTSLAM_KEY_DEPTH));
    };

    bool updateImage(System& system, SystemRunData& data, cv::Mat &image) override;
};

}  // namespace cart