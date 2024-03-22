#include <iostream>

#include "cartslam.hpp"
#include "datasource.hpp"
#include "logging.hpp"
#include "modules/features.hpp"
#include "modules/optflow.hpp"
#include "opencv2/core/cuda.hpp"
#include "opencv2/cudaarithm.hpp"
#include "opencv2/opencv.hpp"
#include "timing.hpp"

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cout << "Please provide an image file to process." << std::endl;
        return 1;
    }

    // cv::cuda::Stream stream = cv::cuda::Stream();
    // auto opticalFlow = cart::createOpticalFlow(stream);

    cart::configureLogging("app.log");

    cart::System system(new cart::KITTIDataSource(argv[1], 0));
    system.addModule(new cart::ImageFeatureDetectorModule(cart::detectOrbFeatures));
    system.addModule(new cart::ImageFeatureVisualizationModule());
    // system.addModule(new cart::ImageOpticalFlowModule());
    // system.addModule(new cart::ImageOpticalFlowVisualizationModule());

    CARTSLAM_START_AVERAGE_TIMING(system);

    boost::future<void> last;

    for (int i = 0; i < 1000; i++) {
        CARTSLAM_START_TIMING(system);
        last = system.run();
        CARTSLAM_END_TIMING(system);
        CARTSLAM_INCREMENT_AVERAGE_TIMING(system);
    }

    last.wait();
    CARTSLAM_END_AVERAGE_TIMING(system);

    /*
    cart::StereoDataElement* element1;
    cart::StereoDataElement* element2 = static_cast<cart::StereoDataElement*>(dataSource.getNext(stream));

    CARTSLAM_START_AVERAGE_TIMING(optflow);

    for (int i = 0; i < 1000; i++) {
        CARTSLAM_START_TIMING(optflow);
        element1 = element2;
        element2 = static_cast<cart::StereoDataElement*>(dataSource.getNext(stream));

        cart::ImageOpticalFlow imageFlow = cart::detectOpticalFlow(element1->left, element2->left, opticalFlow);

        CARTSLAM_END_TIMING(optflow);
        CARTSLAM_INCREMENT_AVERAGE_TIMING(optflow);

        cv::Mat flowImage = cart::drawOpticalFlow(imageFlow, opticalFlow, stream);

        cv::Mat image1, image2;
        element1->left.download(image1, stream);
        element2->left.download(image2, stream);
        stream.waitForCompletion();

        cv::cvtColor(image1, image1, cv::COLOR_GRAY2BGR);
        cv::cvtColor(image2, image2, cv::COLOR_GRAY2BGR);

        cv::Mat concatRes;

        cv::vconcat(image1, image2, concatRes);
        cv::vconcat(concatRes, flowImage, concatRes);

        cv::imshow("Optical Flow", concatRes);

        if (cv::waitKey(100) == 27) {
            break;
        }
    }

    CARTSLAM_END_AVERAGE_TIMING(optflow);
    */
    /*
    cart::FeatureDetector detector = cart::detectOrbFeatures;

    CARTSLAM_START_AVERAGE_TIMING(keypoints);

    for (int i = 0; i < 1000; i++) {
        CARTSLAM_START_TIMING(keypoints);

        cart::StereoDataElement* element = static_cast<cart::StereoDataElement*>(dataSource.getNext(stream));
        detector(element->left, stream);

        CARTSLAM_END_TIMING(keypoints);
        CARTSLAM_INCREMENT_AVERAGE_TIMING(keypoints);
    }

    CARTSLAM_END_AVERAGE_TIMING(keypoints);
    */

    return 0;
}