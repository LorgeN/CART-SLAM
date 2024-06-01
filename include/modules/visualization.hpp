#pragma once

#include <log4cxx/logger.h>

#include "../utils/ui.hpp"
#include "module.hpp"

namespace cart {

class VisualizationModule : public SyncWrapperSystemModule {
   public:
    VisualizationModule(const std::string& name) : SyncWrapperSystemModule(name) {
        this->imageHandle = ImageProvider::create(name);
    }

    system_data_t runInternal(System& system, SystemRunData& data) override;

    virtual bool updateImage(System& system, SystemRunData& data, cv::Mat &image) = 0;
   private:
    boost::shared_ptr<ImageProvider> imageHandle;
};

}  // namespace cart
