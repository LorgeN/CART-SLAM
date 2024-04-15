#include "datasource.hpp"

#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>

void processImage(cv::cuda::GpuMat& image, cv::cuda::Stream& stream) {
    if (image.rows != CARTSLAM_IMAGE_RES_X || image.cols != CARTSLAM_IMAGE_RES_Y) {
        cv::cuda::resize(image, image, cv::Size(CARTSLAM_IMAGE_RES_X, CARTSLAM_IMAGE_RES_Y), 0, 0, cv::INTER_LINEAR, stream);
    }

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
boost::shared_ptr<DataElement> DataSource::getNext(cv::cuda::Stream& stream) {
    if (!this->hasNext()) {
        throw std::runtime_error("No more elements to read");
    }

    auto element = this->getNextInternal(stream);

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
}  // namespace cart