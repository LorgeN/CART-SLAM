#include <iostream>

#include "cartslam.hpp"
#include "logging.hpp"
#include "modules/disparity.hpp"
#include "modules/features.hpp"
#include "modules/optflow.hpp"
#include "modules/planeseg.hpp"
#include "opencv2/core/cuda.hpp"
#include "opencv2/cudaarithm.hpp"
#include "opencv2/opencv.hpp"
#include "sources/kitti.hpp"
#include "sources/zed.hpp"
#include "timing.hpp"

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cout << "Please provide an image file to process." << std::endl;
        return 1;
    }

    cart::configureLogging("app.log");

    auto dataSource = boost::make_shared<cart::sources::ZEDDataSource>(argv[1], true);
    // auto dataSource = boost::make_shared<cart::sources::KITTIDataSource>(argv[1], 0);
    auto system = boost::make_shared<cart::System>(dataSource);

    system->addModule<cart::ZEDImageDisparityModule>();
    // system->addModule<cart::ImageDisparityModule>(1, 256, 5, 3, 5);
    system->addModule<cart::ImageDisparityVisualizationModule>();

    auto provider = boost::make_shared<cart::StaticPlaneParameterProvider>(3, 0, std::make_pair(3, 9), std::make_pair(-3, 3));
    // auto provider = boost::make_shared<cart::HistogramPeakPlaneParameterProvider>();
    system->addModule<cart::DisparityPlaneSegmentationModule>(provider, 5, 1000);
    system->addModule<cart::DisparityPlaneSegmentationVisualizationModule>();

    // system.addModule(new cart::ImageFeatureDetectorModule(cart::detectOrbFeatures));
    // system.addModule(new cart::ImageFeatureVisualizationModule());

    // system.addModule(new cart::ImageOpticalFlowModule());
    // system.addModule(new cart::ImageOpticalFlowVisualizationModule());

    CARTSLAM_START_AVERAGE_TIMING(system);

    boost::future<void> last;

    while (dataSource->hasNext()) {
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