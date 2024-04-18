#pragma once

#include <log4cxx/logger.h>

#include <boost/thread.hpp>
#include <opencv2/opencv.hpp>

#define CARTSLAM_IMAGE_TYPE cv::cuda::GpuMat
#define CARTSLAM_IMAGE_RES_X 1280
#define CARTSLAM_IMAGE_RES_Y 384

namespace cart {
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

    StereoDataElement(CARTSLAM_IMAGE_TYPE left, CARTSLAM_IMAGE_TYPE right) : DataElement(DataElementType::STEREO), left(left), right(right){};

    // TODO: Redefine type so that it is moved to CPU when the run finishes
    CARTSLAM_IMAGE_TYPE left;
    CARTSLAM_IMAGE_TYPE right;
};

CARTSLAM_IMAGE_TYPE getReferenceImage(boost::shared_ptr<DataElement> element);

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
    virtual ~DataSource() = default;
    boost::shared_ptr<DataElement> getNext(log4cxx::LoggerPtr logger, cv::cuda::Stream& stream);
    virtual bool hasNext() = 0;
    virtual DataElementType getProvidedType() = 0;

   protected:
    virtual boost::shared_ptr<DataElement> getNextInternal(log4cxx::LoggerPtr logger, cv::cuda::Stream& stream) = 0;
};
}  // namespace cart
