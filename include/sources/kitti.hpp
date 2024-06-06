#pragma once

#include <boost/thread.hpp>
#include <opencv2/opencv.hpp>

#include "datasource.hpp"

namespace cart::sources {
class KITTIDataSource : public DataSource {
   public:
    KITTIDataSource(std::string basePath, int sequence, cv::Size imageSize = cv::Size(0, 0));
    KITTIDataSource(std::string path, cv::Size imageSize = cv::Size(0, 0));
    bool isNextReady() override;
    bool isFinished() override;
    DataElementType getProvidedType() override;

   protected:
    boost::shared_ptr<DataElement> getNextInternal(log4cxx::LoggerPtr logger, cv::cuda::Stream& stream) override;

   private:
    std::string path;
    int currentFrame;
};

}  // namespace cart::sources