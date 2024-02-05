#include "cartslam.hpp"

#include <iostream>
#include <opencv2/cudaarithm.hpp>

#include "datasource.hpp"
#include "features.hpp"
#include "opencv2/core/cuda.hpp"
#include "opencv2/opencv.hpp"
#include "timing.hpp"

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cout << "Please provide an image file to process." << std::endl;
        return 1;
    }

    cv::cuda::Stream stream = cv::cuda::Stream();

    cart::KITTIDataSource dataSource(argv[1], 0);
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

    return 0;
}