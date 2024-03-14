#include "datasource.hpp"

#include "opencv2/cudaimgproc.hpp"
#include "opencv2/cudawarping.hpp"

std::string addLeadingZeros(int number, std::size_t length) {
    std::string numberAsString = std::to_string(number);
    return std::string(length - std::min(length, numberAsString.length()), '0') + numberAsString;
}

void processImage(cv::cuda::GpuMat& image, cv::cuda::Stream& stream) {
    if (image.rows != CARTSLAM_IMAGE_RES_X || image.cols != CARTSLAM_IMAGE_RES_Y) {
        cv::cuda::resize(image, image, cv::Size(CARTSLAM_IMAGE_RES_X, CARTSLAM_IMAGE_RES_Y), 0, 0, cv::INTER_LINEAR, stream);
    }

    if (image.type() != CV_8UC1) {
        cv::cuda::cvtColor(image, image, cv::COLOR_BGR2GRAY, 0, stream);
    }
}

namespace cart {
DataElement* DataSource::getNext(cv::cuda::Stream& stream) {
    auto element = this->getNextInternal(stream);

    switch (element->type) {
        case STEREO: {
            auto stereoElement = static_cast<StereoDataElement*>(element);
            processImage(stereoElement->left, stream);
            processImage(stereoElement->right, stream);
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

DataElement* KITTIDataSource::getNextInternal(cv::cuda::Stream& stream) {
    cv::Mat left = cv::imread(this->path + "/image_2/" + addLeadingZeros(this->currentFrame, 6) + ".png");
    cv::Mat right = cv::imread(this->path + "/image_3/" + addLeadingZeros(this->currentFrame, 6) + ".png");

    this->currentFrame++;

    auto element = new StereoDataElement();

    element->left.upload(left, stream);
    element->right.upload(right, stream);

    return element;
}

}  // namespace cart