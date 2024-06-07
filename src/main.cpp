#include <iostream>

#include "cartslam.hpp"
#include "logging.hpp"
#include "modules/depth.hpp"
#include "modules/disparity.hpp"
#include "modules/features.hpp"
#include "modules/optflow.hpp"
#include "modules/planefit.hpp"
#include "modules/planeseg.hpp"
#include "modules/superpixels.hpp"
#include "opencv2/core/cuda.hpp"
#include "opencv2/cudaarithm.hpp"
#include "opencv2/opencv.hpp"
#include "sources/kitti.hpp"
#include "sources/zed.hpp"

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cout << "Please provide an image file to process." << std::endl;
        return 1;
    }

    cart::configureLogging("app.log");

    int kittiSeq = 0;
    if (argc >= 3) {
        kittiSeq = std::stoi(argv[2]);
    }

    auto logger = cart::getLogger("main");

    //auto dataSource = boost::make_shared<cart::sources::ZEDDataSource>(argv[1], true);
    auto dataSource = boost::make_shared<cart::sources::KITTIDataSource>(argv[1], kittiSeq);
    auto system = boost::make_shared<cart::System>(dataSource);

    LOG4CXX_INFO(logger, "Resolution: " << dataSource->getImageSize().width << "x" << dataSource->getImageSize().height);

    system->addModule<cart::SuperPixelModule>(dataSource->getImageSize(), 32, 12, 10, 64, 0.3, 0.3 / sqrt(2));
    system->addModule<cart::SuperPixelVisualizationModule>();

    system->addModule<cart::ImageOpticalFlowModule>(dataSource->getImageSize());
    //  system->addModule<cart::ImageOpticalFlowVisualizationModule>(dataSource->getImageSize());

    // system->addModule<cart::ZEDImageDisparityModule>();
    system->addModule<cart::ImageDisparityModule>(1, 256, 3, 2, 1);
    // system->addModule<cart::ImageDisparityVisualizationModule>();

    system->addModule<cart::ImageDisparityDerivativeModule>();
    // system->addModule<cart::ImageDisparityDerivativeVisualizationModule>();

    system->addModule<cart::DepthModule>();
    // system->addModule<cart::DepthVisualizationModule>();

    // auto provider = boost::make_shared<cart::StaticPlaneParameterProvider>(3, 0, std::make_pair(3, 30), std::make_pair(-3, 3));
    auto provider = boost::make_shared<cart::HistogramPeakPlaneParameterProvider>();
    // system->addModule<cart::DisparityPlaneSegmentationModule>(provider, 30, 20, true);
    system->addModule<cart::SuperPixelDisparityPlaneSegmentationModule>(provider, 10, 30, true);
    system->addModule<cart::DisparityPlaneSegmentationVisualizationModule>(true, true);
    system->addModule<cart::PlaneSegmentationBEVVisualizationModule>();

    // system->addModule<cart::SuperPixelPlaneFitModule>();

    // system.addModule(new cart::ImageFeatureDetectorModule(cart::detectOrbFeatures));
    // system.addModule(new cart::ImageFeatureVisualizationModule());

    if (dataSource->isFinished()) {
        LOG4CXX_WARN(cart::getLogger("main"), "The provided data source has no data. Exiting.");
        return 1;
    }

    boost::future<void> last;

    while (!dataSource->isFinished()) {
        if (!dataSource->isNextReady()) {
            continue;
        }

        last = system->run().then([logger](boost::future<void> future) {
            try {
                future.get();
            } catch (const std::exception& e) {
                LOG4CXX_ERROR(logger, "Error in processing: " << e.what());
            }
        });
    }

    last.get();

    system->getThreadPool().join();
    return 0;
}