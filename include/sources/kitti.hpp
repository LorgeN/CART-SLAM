#pragma once

#include <boost/thread.hpp>
#include <opencv2/opencv.hpp>

#include "datasource.hpp"

namespace cart::sources {
class KITTIDataSource : public DataSource {
   public:
    KITTIDataSource(std::string basePath, int sequence);
    KITTIDataSource(std::string path);
    bool hasNext() override;
    DataElementType getProvidedType() override;
    const CameraIntrinsics getCameraIntrinsics() const override;

   protected:
    boost::shared_ptr<DataElement> getNextInternal(log4cxx::LoggerPtr logger, cv::cuda::Stream& stream) override;

   private:
    std::string path;
    int currentFrame;
};

}  // namespace cart::sources