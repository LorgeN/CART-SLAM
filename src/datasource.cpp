#include "datasource.hpp"

#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>

void processImage(cv::cuda::GpuMat& image, cv::cuda::Stream& stream) {
#ifdef CARTSLAM_IMAGE_MAKE_GRAYSCALE
    if (image.type() != CV_8UC1) {
        cv::cuda::cvtColor(image, image, cv::COLOR_BGR2GRAY, 0, stream);
    }
#else
    if (image.type() != CV_8UC3) {
        cv::cuda::cvtColor(image, image, cv::COLOR_GRAY2BGR, 0, stream);
    }
#endif
}

namespace cart {
image_t getReferenceImage(boost::shared_ptr<DataElement> element) {
    switch (element->type) {
        case STEREO: {
            auto stereoElement = boost::static_pointer_cast<StereoDataElement>(element);
            return stereoElement->left;
        } break;
        default:
            throw std::runtime_error("Unknown data element type");
    }
}

boost::shared_ptr<DataElement> DataSource::getNext(log4cxx::LoggerPtr logger, cv::cuda::Stream& stream) {
    if (!this->isNextReady()) {
        throw std::runtime_error("Next element is not ready!");
    }

    auto element = this->getNextInternal(logger, stream);

    switch (element->type) {
        case STEREO: {
            auto stereoElement = boost::static_pointer_cast<StereoDataElement>(element);
            processImage(stereoElement->left, stream);
            processImage(stereoElement->right, stream);
        } break;
        default:
            break;
    }

    return element;
}

const CameraIntrinsics DataSource::getCameraIntrinsics() const {
    return this->intrinsics;
}

const cv::Size DataSource::getImageSize() const {
    return this->imageSize;
}
}  // namespace cart