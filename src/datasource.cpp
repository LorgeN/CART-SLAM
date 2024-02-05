#include "datasource.hpp"

#include "opencv2/cudawarping.hpp"

std::string addLeadingZeros(int number, std::size_t length) {
    std::string numberAsString = std::to_string(number);
    return std::string(length - std::min(length, numberAsString.length()), '0') + numberAsString;
}

namespace cart {
DataElement* DataSource::getNext(cv::cuda::Stream stream) {
    auto element = this->getNextInternal(stream);

    switch (element->getType()) {
        case STEREO: {
            auto stereoElement = static_cast<StereoDataElement*>(element);
            if (stereoElement->left.rows != CARTSLAM_IMAGE_RES_X || stereoElement->left.cols != CARTSLAM_IMAGE_RES_Y) {
                cv::cuda::resize(stereoElement->left, stereoElement->left, cv::Size(CARTSLAM_IMAGE_RES_X, CARTSLAM_IMAGE_RES_Y), 0, 0, cv::INTER_LINEAR, stream);
            }

            if (stereoElement->right.rows != CARTSLAM_IMAGE_RES_X || stereoElement->right.cols != CARTSLAM_IMAGE_RES_Y) {
                cv::cuda::resize(stereoElement->right, stereoElement->right, cv::Size(CARTSLAM_IMAGE_RES_X, CARTSLAM_IMAGE_RES_Y), 0, 0, cv::INTER_LINEAR, stream);
            }
        } break;
        default:
            break;
    }

    return element;
}

KITTIDataSource::KITTIDataSource(std::string basePath, int sequence) {
    this->path = basePath + "/sequences/" + addLeadingZeros(sequence, 2);
    this->currentFrame = 0;
}

KITTIDataSource::KITTIDataSource(std::string path) {
    this->path = path;
    this->currentFrame = 0;
}

DataElementType KITTIDataSource::getProvidedType() {
    return DataElementType::STEREO;
}

DataElement* KITTIDataSource::getNextInternal(cv::cuda::Stream stream) {
    cv::Mat left = cv::imread(this->path + "/image_2/" + addLeadingZeros(this->currentFrame, 6) + ".png");
    cv::Mat right = cv::imread(this->path + "/image_3/" + addLeadingZeros(this->currentFrame, 6) + ".png");

    this->currentFrame++;

    auto element = new StereoDataElement();

    element->left.upload(left, stream);
    element->right.upload(right, stream);

    return element;
}

}  // namespace cart