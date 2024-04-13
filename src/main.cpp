#include <iostream>

#include "cartslam.hpp"
#include "datasource.hpp"
#include "logging.hpp"
#include "modules/disparity.hpp"
#include "modules/features.hpp"
#include "modules/optflow.hpp"
#include "modules/planeseg.hpp"
#include "opencv2/core/cuda.hpp"
#include "opencv2/cudaarithm.hpp"
#include "opencv2/opencv.hpp"
#include "timing.hpp"

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cout << "Please provide an image file to process." << std::endl;
        return 1;
    }

    cart::configureLogging("app.log");

    auto dataSource = boost::make_shared<cart::KITTIDataSource>(argv[1], 0);
    auto system = boost::make_shared<cart::System>(dataSource);

    system->addModule<cart::ImageDisparityModule>(1, 256, 5, 3, 5);
    system->addModule<cart::ImageDisparityVisualizationModule>();

    system->addModule<cart::DisparityPlaneSegmentationModule>();
    system->addModule<cart::DisparityPlaneSegmentationVisualizationModule>();

    // system.addModule(new cart::ImageFeatureDetectorModule(cart::detectOrbFeatures));
    // system.addModule(new cart::ImageFeatureVisualizationModule());

    // system.addModule(new cart::ImageOpticalFlowModule());
    // system.addModule(new cart::ImageOpticalFlowVisualizationModule());

    CARTSLAM_START_AVERAGE_TIMING(system);

    boost::future<void> last;

    for (int i = 0; i < 3000; i++) {
        CARTSLAM_START_TIMING(system);

        last = system->run();

        CARTSLAM_END_TIMING(system);
        CARTSLAM_INCREMENT_AVERAGE_TIMING(system);
    }

    last.wait();

    system->getThreadPool().join();

    CARTSLAM_END_AVERAGE_TIMING(system);
    return 0;
}