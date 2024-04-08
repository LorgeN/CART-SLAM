#ifndef CARTSLAM_DATASOURCE_HPP
#define CARTSLAM_DATASOURCE_HPP

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

    CARTSLAM_IMAGE_TYPE left;
    CARTSLAM_IMAGE_TYPE right;
};

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
    boost::shared_ptr<DataElement> getNext(cv::cuda::Stream& stream);
    virtual boost::shared_ptr<DataElement> getNextInternal(cv::cuda::Stream& stream) = 0;
    virtual DataElementType getProvidedType() = 0;
};

class KITTIDataSource : public DataSource {
   public:
    KITTIDataSource(std::string basePath, int sequence);
    KITTIDataSource(std::string path);
    boost::shared_ptr<DataElement> getNextInternal(cv::cuda::Stream& stream) override;
    DataElementType getProvidedType() override;

   private:
    std::string path;
    int currentFrame;
};

}  // namespace cart

#endif  // CARTSLAM_DATASOURCE_HPP