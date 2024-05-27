#pragma once

#include <opencv2/opencv.hpp>

#include "../../../logging.hpp"
#include "../constants.hpp"

namespace cart::contour {

enum DataType {
    Image,
    Disparity
};

class CUDAFeature {
   public:
    virtual ~CUDAFeature() = default;

    __device__ virtual void initializeStatistics(const cv::cuda::PtrStepSz<label_t> labelImage, size_t xBatch, size_t yBatch) = 0;

    __device__ virtual double calculateCost(const cv::Point2i curPixelCoords,
                                            const label_t oldLabel, const label_t pretendLabel,
                                            const label_t* neighbourLabels, const size_t neighbourLabelSize) const = 0;

    __device__ virtual void updateStatistics(const cv::Point2i const& curPixelCoords, label_t const& oldLabel, label_t const& newLabel) = 0;

    __device__ virtual void setImageData(const cv::cuda::PtrStepSz<uint8_t> image) {}  // Provide a default implementation for features that do not need data

    __device__ virtual void setDisparityData(const cv::cuda::PtrStepSz<int16_t> disparity) {}  // Provide a default implementation for features that do not need data
};

class IFeature {
   protected:
    log4cxx::LoggerPtr logger;

   public:
    IFeature(const std::string featureName) {
        logger = getLogger(featureName);
    }

    virtual ~IFeature() = default;

    virtual void initializeCUDAFeature(CUDAFeature*& cudaFeature, const label_t maxLabelId, cudaStream_t stream) = 0;
};
}  // namespace cart::contour