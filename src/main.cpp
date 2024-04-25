#include <iostream>

#include "cartslam.hpp"
#include "logging.hpp"
#include "modules/disparity.hpp"
#include "modules/features.hpp"
#include "modules/optflow.hpp"
#include "modules/planeseg.hpp"
#include "modules/superpixels.hpp"
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

    system->addModule<cart::SuperPixelModule>();
    system->addModule<cart::SuperPixelVisualizationModule>();

    // system->addModule<cart::ImageOpticalFlowModule>();
    // system->addModule<cart::ImageOpticalFlowVisualizationModule>();

    system->addModule<cart::ZEDImageDisparityModule>();
    // system->addModule<cart::ImageDisparityModule>(1, 256, 5, 3, 5);
    // system->addModule<cart::ImageDisparityVisualizationModule>();

    system->addModule<cart::ImageDisparityDerivativeModule>();
    // system->addModule<cart::ImageDisparityDerivativeVisualizationModule>();

    // auto provider = boost::make_shared<cart::StaticPlaneParameterProvider>(3, 0, std::make_pair(3, 9), std::make_pair(-3, 3));
    // auto provider = boost::make_shared<cart::HistogramPeakPlaneParameterProvider>();
    // system->addModule<cart::DisparityPlaneSegmentationModule>(provider, 30, 20, true);
    // system->addModule<cart::DisparityPlaneSegmentationVisualizationModule>(false, true);

    // system.addModule(new cart::ImageFeatureDetectorModule(cart::detectOrbFeatures));
    // system.addModule(new cart::ImageFeatureVisualizationModule());

    if (!dataSource->hasNext()) {
        LOG4CXX_WARN(cart::getLogger("main"), "The provided data source has no data. Exiting.");
        return 1;
    }

    CARTSLAM_START_AVERAGE_TIMING(system);

    auto logger = cart::getLogger("main");

    boost::future<void> last;

    while (dataSource->hasNext()) {
        // Not technically accurate timing because runs are async, but good enough for our purposes for now
        CARTSLAM_START_TIMING(system);

        last = system->run();

        CARTSLAM_END_TIMING(system);
        CARTSLAM_INCREMENT_AVERAGE_TIMING(system);

        // Skip a few frames
        int i = 0;
        cv::cuda::Stream stream;
        while (dataSource->hasNext() && i < 10) {
            LOG4CXX_DEBUG(logger, "Skipping frame " << i);
            dataSource->getNext(logger, stream);
            i++;
        }

        stream.waitForCompletion();
    }

    last.wait();

    system->getThreadPool().join();

    CARTSLAM_END_AVERAGE_TIMING(system);
    return 0;
}