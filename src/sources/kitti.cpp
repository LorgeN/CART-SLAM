#include "sources/kitti.hpp"

#include <sys/stat.h>

#define PATH(ch, frame) (this->path + "/image_" + std::to_string(ch) + "/" + addLeadingZeros(frame, 6) + ".png")
#define LEFT_PATH(frame) PATH(2, frame)
#define RIGHT_PATH(frame) PATH(3, frame)

std::string addLeadingZeros(int number, std::size_t length) {
    std::string numberAsString = std::to_string(number);
    return std::string(length - std::min(length, numberAsString.length()), '0') + numberAsString;
}

namespace cart::sources {

KITTIDataSource::KITTIDataSource(std::string basePath, int sequence) {
    this->path = basePath + "/sequences/" + addLeadingZeros(sequence, 2);
    this->currentFrame = 0;
}

KITTIDataSource::KITTIDataSource(std::string path) {
    this->path = path;
    this->currentFrame = 0;
}

const CameraIntrinsics KITTIDataSource::getCameraIntrinsics() const {
    // TODO: Implement
    CameraIntrinsics intrinsics;
    return intrinsics;
};

DataElementType KITTIDataSource::getProvidedType() {
    return DataElementType::STEREO;
}

boost::shared_ptr<DataElement> KITTIDataSource::getNextInternal(log4cxx::LoggerPtr logger, cv::cuda::Stream& stream) {
    cv::Mat left = cv::imread(LEFT_PATH(this->currentFrame));
    cv::Mat right = cv::imread(RIGHT_PATH(this->currentFrame));

    this->currentFrame++;

    auto element = boost::make_shared<StereoDataElement>();

    element->left.upload(left, stream);
    element->right.upload(right, stream);

    return element;
}

bool KITTIDataSource::hasNext() {
    std::string leftPath = LEFT_PATH(this->currentFrame);

    // Check if the file exists
    struct stat buffer;
    return stat(leftPath.c_str(), &buffer) == 0;
}
}  // namespace cart::sources