#pragma once

#ifdef CARTSLAM_ZED
#include <boost/thread.hpp>
#include <opencv2/opencv.hpp>
#include <sl/Camera.hpp>

#include "datasource.hpp"
#include "utils/ui.hpp"

namespace cart::sources {

cv::cuda::GpuMat slMat2cvGpuMat(sl::Mat& input);

cv::Mat slMat2cvMat(sl::Mat& input);

class ZEDDataSource : public DataSource {
   public:
    ZEDDataSource(std::string path, bool extractDepthMeasure = false);
    ~ZEDDataSource();
    bool hasNext() override;
    DataElementType getProvidedType() override;
    const CameraIntrinsics getCameraIntrinsics() const override;

   protected:
    boost::shared_ptr<DataElement> getNextInternal(log4cxx::LoggerPtr logger, cv::cuda::Stream& stream) override;

   private:
    const bool extractDisparityMeasure;
    const std::string path;

    bool hasGrabbed = false;
    sl::ERROR_CODE grabResult;

    boost::shared_ptr<sl::Camera> camera;
};

class ZEDDataElement : public StereoDataElement {
   public:
    ZEDDataElement(){};

    ZEDDataElement(image_t left, image_t right) : StereoDataElement(left, right){};

    cv::cuda::GpuMat disparityMeasure;
};
}  // namespace cart::sources
#endif
