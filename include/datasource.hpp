#ifndef CARTSLAM_DATASOURCE_HPP
#define CARTSLAM_DATASOURCE_HPP

#include "opencv2/opencv.hpp"

#define CARTSLAM_IMAGE_TYPE cv::cuda::GpuMat
#define CARTSLAM_IMAGE_RES_X 1280
#define CARTSLAM_IMAGE_RES_Y 384

namespace cart {
enum DataElementType {
    STEREO
};

class DataElement {
   public:
    virtual ~DataElement() = default;
    virtual DataElementType getType() = 0;
};

class StereoDataElement : public DataElement {
   public:
    StereoDataElement() = default;

    StereoDataElement(CARTSLAM_IMAGE_TYPE left, CARTSLAM_IMAGE_TYPE right) {
        this->left = left;
        this->right = right;
    }

    DataElementType getType() override {
        return DataElementType::STEREO;
    }

    CARTSLAM_IMAGE_TYPE left;
    CARTSLAM_IMAGE_TYPE right;
};

class DataSource {
   public:
    virtual ~DataSource() = default;
    DataElement* getNext(cv::cuda::Stream &stream);
    virtual DataElement* getNextInternal(cv::cuda::Stream &stream) = 0;
    virtual DataElementType getProvidedType() = 0;
};

class KITTIDataSource : public DataSource {
   public:
    KITTIDataSource(std::string basePath, int sequence);
    KITTIDataSource(std::string path);
    DataElement* getNextInternal(cv::cuda::Stream &stream) override;
    DataElementType getProvidedType() override;

   private:
    std::string path;
    int currentFrame;
};
}  // namespace cart

#endif  // CARTSLAM_DATASOURCE_HPP