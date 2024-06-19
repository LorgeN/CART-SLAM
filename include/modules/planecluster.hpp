#pragma once

#include "depth.hpp"
#include "disparity.hpp"
#include "module.hpp"
#include "superpixels.hpp"
#include "visualization.hpp"
#include "planefit.hpp"

namespace cart {

class SuperPixelPlaneClusterModule : public SyncWrapperSystemModule {
   public:
    SuperPixelPlaneClusterModule() : SyncWrapperSystemModule("SuperPixelPlaneCluster") {
        this->requiresData.push_back(module_dependency_t(CARTSLAM_KEY_DEPTH));
        this->requiresData.push_back(module_dependency_t(CARTSLAM_KEY_SUPERPIXELS));
        this->requiresData.push_back(module_dependency_t(CARTSLAM_KEY_SUPERPIXELS_MAX_LABEL));
    
        this->providesData.push_back(CARTSLAM_KEY_PLANES_EQ);
    };

    system_data_t runInternal(System& system, SystemRunData& data) override;
};
}  // namespace cart