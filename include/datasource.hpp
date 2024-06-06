#pragma once

#include <log4cxx/logger.h>

#include <boost/thread.hpp>
#include <opencv2/opencv.hpp>

namespace cart {
typedef cv::cuda::GpuMat image_t;

struct CameraIntrinsics {
    /**
     * @brief The Q matrix is used by OpenCV to reproject disparity images into 3D space.
     *
     * @see cv::cuda::reprojectImageTo3D
     */
    cv::Mat Q;
};

enum DataElementType {
    STEREO
};

class DataElement {
   public:
    DataElement(DataElementType type) : type(type){};

    virtual ~DataElement() = default;

    const DataElementType type;
};

class StereoDataElement : public DataElement {
   public:
    StereoDataElement() : DataElement(DataElementType::STEREO){};

    StereoDataElement(image_t left, image_t right) : DataElement(DataElementType::STEREO), left(left), right(right){};

    // TODO: Redefine type so that it is moved to CPU when the run finishes
    image_t left;
    image_t right;
};

image_t getReferenceImage(boost::shared_ptr<DataElement> element);

template <typename T>
class DataElementVisitor {
   public:
    virtual T visitStereo(boost::shared_ptr<StereoDataElement> element) {
        throw std::runtime_error("Not implemented");
    }

    T operator()(boost::shared_ptr<DataElement> element) {
        switch (element->type) {
            case STEREO: {
                return this->visitStereo(boost::static_pointer_cast<StereoDataElement>(element));
            } break;
            default:
                throw std::runtime_error("Unknown data element type");
        }
    }
};

class DataSource {
   public:
    DataSource(cv::Size imageSize) : imageSize(imageSize){};

    virtual ~DataSource() = default;
    boost::shared_ptr<DataElement> getNext(log4cxx::LoggerPtr logger, cv::cuda::Stream& stream);
    virtual bool isNextReady() = 0;
    virtual bool isFinished() = 0;
    virtual DataElementType getProvidedType() = 0;

    const CameraIntrinsics getCameraIntrinsics() const;
    const cv::Size getImageSize() const;

   protected:
    virtual boost::shared_ptr<DataElement> getNextInternal(log4cxx::LoggerPtr logger, cv::cuda::Stream& stream) = 0;

    CameraIntrinsics intrinsics;
    cv::Size imageSize;
};
}  // namespace cart
