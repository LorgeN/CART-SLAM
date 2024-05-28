#pragma once

#include <opencv2/core/cuda_stream_accessor.hpp>

#include "../../../../utils/cuda.cuh"
#include "feature.cuh"

#define STAT_INDEX(x, y, width) ((y) * (width) + (x))
#define INDEX_ROW(x, chCount, ch) ((x) * (chCount) + (ch))
#define SQUARED(x) ((x) * (x))

namespace cart::contour {

/**
 * @struct LabelStatisticsGauss
 * @brief Struct containing the sufficient statistics needed to compute label likelihoods of a Gaussian-distributed feature channel.
 */
struct LabelStatisticsGauss {
    unsigned int pixelCount;  ///< the number of pixels assigned to the label
    double valueSum;          ///< the sum of values of all pixels assigned to the label
    double squareValueSum;    ///< the sum of squared values of all pixels assigned to the label
    double featureCost;

    /**
     * @brief Default constructor to ensure that all members are initialized to sensible values.
     */
    __device__ __host__ LabelStatisticsGauss() : pixelCount(0), valueSum(0), squareValueSum(0) {}
};

/**
 * @class CUDAGaussianFeature
 * @brief Abstract feature class providing the basic functionality needed for all features with Gaussian distribution.
 */
template <typename TData, size_t VChannels, DataType Type>
class CUDAGaussianFeature : public CUDAFeature {
   private:
    LabelStatisticsGauss* labelStatistics;
    label_t maxLabelId;

   protected:
    cv::cuda::PtrStepSz<TData> data;

    __device__ inline const LabelStatisticsGauss& getLabelStatistics(const label_t label, const size_t channel) const {
        return labelStatistics[STAT_INDEX(channel, label, VChannels)];
    }

    __device__ inline LabelStatisticsGauss& getLabelStatistics(const label_t label, const size_t channel) {
        return labelStatistics[STAT_INDEX(channel, label, VChannels)];
    }

    __device__ void updateGaussianStatistics(const CRPoint curPixelCoords, LabelStatisticsGauss* labelStatsOldLabel, LabelStatisticsGauss* labelStatsNewLabel) const;

   public:
    __device__ CUDAGaussianFeature(const label_t maxLabelId);

    __device__ ~CUDAGaussianFeature() override;

    __device__ void initializeStatistics(const cv::cuda::PtrStepSz<label_t> labelImage, size_t xBatch, size_t yBatch) override;

    __device__ double calculateCost(const CRPoint curPixelCoords,
                                    const label_t oldLabel, const label_t pretendLabel,
                                    const label_t* neighbourLabels, const size_t neighbourLabelSize) const override;

    __device__ void updateStatistics(const CRPoint curPixelCoords, label_t const& oldLabel, label_t const& newLabel) override;

    __device__ void setImageData(const cv::cuda::PtrStepSz<uint8_t> image) override;

    __device__ void setDisparityData(const cv::cuda::PtrStepSz<int16_t> disparity) override;
};

template <typename TData, size_t VChannels, DataType Type>
class AGaussianFeature : public IFeature {
   public:
    AGaussianFeature(const std::string featureName) : IFeature(featureName) {}

    void initializeCUDAFeature(CUDAFeature**& cudaFeature, const label_t maxLabelId, const cv::cuda::Stream& stream = cv::cuda::Stream::Null()) override;
};
}  // namespace cart::contour