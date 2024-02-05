#include "cartslam.hpp"

#include <iostream>
#include <opencv2/cudaarithm.hpp>

#include "datasource.hpp"
#include "features.hpp"
#include "opencv2/core/cuda.hpp"
#include "opencv2/opencv.hpp"

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cout << "Please provide an image file to process." << std::endl;
        return 1;
    }

    cv::cuda::Stream stream = cv::cuda::Stream();

    cart::KITTIDataSource dataSource(argv[1], 0);
    cart::StereoDataElement* element = static_cast<cart::StereoDataElement*>(dataSource.getNextInternal(stream));

    cart::FeatureDetector detector = cart::detectOrbFeatures;

    cv::Mat keypointsImage;
    cv::Mat leftDownload, rightDownload;

    element->left.download(leftDownload);
    element->right.download(rightDownload);

    cart::ImageFeatures leftKeypoints = detector(element->left, stream);
    cv::drawKeypoints(leftDownload, leftKeypoints.keypoints, keypointsImage);
    cv::imshow("Left keypoints", keypointsImage);
    cv::waitKey();

    cart::ImageFeatures rightKeypoints = detector(element->right, stream);
    cv::drawKeypoints(rightDownload, rightKeypoints.keypoints, keypointsImage);
    cv::imshow("Right keypoints", keypointsImage);
    cv::waitKey();

    return 0;
}