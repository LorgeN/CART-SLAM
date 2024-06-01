#include "modules/visualization.hpp"

#include "cartslam.hpp"

namespace cart {
system_data_t VisualizationModule::runInternal(System& system, SystemRunData& data) {
    cv::Mat image;
    if (this->updateImage(system, data, image)) {
        this->imageHandle->setImageIfLater(image, data.id);
    }

    return MODULE_NO_RETURN_VALUE;
}
}  // namespace cart