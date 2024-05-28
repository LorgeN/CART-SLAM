#pragma once

#include <opencv2/opencv.hpp>

#include "../../../../logging.hpp"
#include "../constants.hpp"

namespace cart::contour {

enum DataType {
    Image,
    Disparity
};

class CUDAFeature;

class IFeature {
   protected:
    log4cxx::LoggerPtr logger;

   public:
    IFeature(const std::string featureName) {
        logger = getLogger(featureName);
    }

    virtual ~IFeature() = default;

    virtual void initializeCUDAFeature(CUDAFeature**& cudaFeature, const label_t maxLabelId, const cv::cuda::Stream& stream = cv::cuda::Stream::Null()) = 0;
};
}  // namespace cart::contour