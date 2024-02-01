#include "datasource.hpp"

std::string addLeadingZeros(int number, std::size_t length) {
    std::string numberAsString = std::to_string(number);
    return std::string(length - std::min(length, numberAsString.length()), '0') + numberAsString;
}

namespace cart {
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

DataElement* KITTIDataSource::getNext() {
    try {
        cv::Mat left = cv::imread(this->path + "/image_2/" + addLeadingZeros(this->currentFrame, 6) + ".png");
        cv::Mat right = cv::imread(this->path + "/image_3/" + addLeadingZeros(this->currentFrame, 6) + ".png");

        this->currentFrame++;

#ifdef CARTSLAM_USE_GPU
        cv::cuda::GpuMat leftGpu;
        cv::cuda::GpuMat rightGpu;

        leftGpu.upload(left);
        rightGpu.upload(right);

        return new StereoDataElement(leftGpu, rightGpu);
#else
        return new StereoDataElement(left, right);
#endif
    } catch (const cv::Exception& ex) {
        std::cout << "Error while reading image: " << ex.what() << std::endl;
        return nullptr;
    }
}

}  // namespace cart