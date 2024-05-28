#pragma once

#include <opencv2/opencv.hpp>

#include "../../../../logging.hpp"
#include "../constants.hpp"
#include "ifeature.hpp"

namespace cart::contour {

struct __align__(8) CRPoint {
    int x;
    int y;
};

class CUDAFeature {
   public:
    __device__ virtual ~CUDAFeature() {}

    __device__ virtual void initializeStatistics(const cv::cuda::PtrStepSz<label_t> labelImage, size_t xBatch, size_t yBatch) = 0;

    __device__ virtual double calculateCost(const CRPoint curPixelCoords,
                                            const label_t oldLabel, const label_t pretendLabel,
                                            const label_t* neighbourLabels, const size_t neighbourLabelSize) const = 0;

    __device__ virtual void updateStatistics(const CRPoint curPixelCoords, label_t const& oldLabel, label_t const& newLabel) = 0;

    __device__ virtual void setImageData(const cv::cuda::PtrStepSz<uint8_t> image) {}  // Provide a default implementation for features that do not need data

    __device__ virtual void setDisparityData(const cv::cuda::PtrStepSz<int16_t> disparity) {}  // Provide a default implementation for features that do not need data
};

struct __align__(16) CUDAFeatureContainer {
    CUDAFeature** feature;
    double weight;
};

struct __align__(16) CRSettings {
    double directCliqueCost;
    double diagonalCliqueCost;
    label_t maxLabelId;
    cv::cuda::PtrStepSz<label_t> labelImage;
    CUDAFeatureContainer* features;
    size_t numFeatures;
};

}  // namespace cart::contour