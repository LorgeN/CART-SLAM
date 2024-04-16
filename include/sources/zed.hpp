#pragma once

#ifdef CARTSLAM_ZED
#include <boost/thread.hpp>
#include <opencv2/opencv.hpp>
#include <sl/Camera.hpp>

#include "datasource.hpp"
#include "utils/ui.hpp"

namespace cart::sources {
class ZEDDataSource : public DataSource {
   public:
    ZEDDataSource(std::string path, bool extractDepthMeasure = false);
    ~ZEDDataSource();
    bool hasNext() override;
    DataElementType getProvidedType() override;

   protected:
    boost::shared_ptr<DataElement> getNextInternal(cv::cuda::Stream& stream) override;

   private:
    bool extractDisparityMeasure;
    std::string path;
    boost::shared_ptr<sl::Camera> camera;
    boost::shared_ptr<ImageProvider> imageThread;
};

class ZEDDataElement : public StereoDataElement {
   public:
    ZEDDataElement(){};

    ZEDDataElement(CARTSLAM_IMAGE_TYPE left, CARTSLAM_IMAGE_TYPE right) : StereoDataElement(left, right){};

    // Maintain references to the original ZED images
    sl::Mat slLeft;
    sl::Mat slRight;

    cv::cuda::GpuMat disparityMeasure;
};

cv::cuda::GpuMat slMat2cvGpuMat(sl::Mat& input);

cv::Mat slMat2cvMat(sl::Mat& input);
}  // namespace cart::sources
#endif
