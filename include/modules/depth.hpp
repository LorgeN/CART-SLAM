#pragma once

#include "module.hpp"
#include "disparity.hpp"

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

class DepthVisualizationModule : public SystemModule {
   public:
    DepthVisualizationModule() : SystemModule("DepthVisualization") {
        this->requiresData.push_back(module_dependency_t(CARTSLAM_KEY_DEPTH));
        this->imageThread = ImageProvider::create("Depth");
    };

    boost::future<system_data_t> run(System& system, SystemRunData& data) override;

   private:
    boost::shared_ptr<ImageProvider> imageThread;
};

}  // namespace cart