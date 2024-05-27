#pragma once

#include "../../../utils/cuda.cuh"
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
    LabelStatisticsGauss() : pixelCount(0), valueSum(0), squareValueSum(0) {}
};

/**
 * @class CUDAGaussianFeature
 * @brief Abstract feature class providing the basic functionality needed for all features with Gaussian distribution.
 */
template <typename TData, size_t VChannels>
class CUDAGaussianFeature : public CUDAFeature {
   private:
    LabelStatisticsGauss* labelStatistics;
    label_t maxLabelId;

   protected:
    cv::cuda::PtrStepSz<TData> data;
    const DataType target;

    __device__ inline const LabelStatisticsGauss& getLabelStatistics(const label_t label, const size_t channel) const {
        return labelStatistics[STAT_INDEX(channel, label, VChannels)];
    }

    __device__ inline LabelStatisticsGauss& getLabelStatistics(const label_t label, const size_t channel) {
        return labelStatistics[STAT_INDEX(channel, label, VChannels)];
    }

   public:
    CUDAGaussianFeature(LabelStatisticsGauss* labelStatistics, const label_t maxLabelId, DataType target = Image) : target(target) {
        this->labelStatistics = labelStatistics;
        this->maxLabelId = maxLabelId;
    }

    ~CUDAGaussianFeature() = default;

    __device__ void initializeStatistics(const cv::cuda::PtrStepSz<label_t> labelImage, size_t xBatch, size_t yBatch) override;

    __device__ double calculateCost(const cv::Point2i curPixelCoords,
                                    const label_t oldLabel, const label_t pretendLabel,
                                    const label_t* neighbourLabels, const size_t neighbourLabelSize) const override;

    __device__ void updateStatistics(const cv::Point2i const& curPixelCoords, label_t const& oldLabel, label_t const& newLabel) override;

    __device__ void setImageData(const cv::cuda::PtrStepSz<uint8_t> image) override;

    __device__ void setDisparityData(const cv::cuda::PtrStepSz<int16_t> disparity) override;
};

template <typename TData, size_t VChannels>
class AGaussianFeature : public IFeature {
   private:
    __global__ void createFeatureInstance(CUDAGaussianFeature<TData, VChannels>** cudaFeature, LabelStatisticsGauss* labelStatistics, const label_t maxLabelId);

   public:
    AGaussianFeature(const std::string featureName) : IFeature(featureName) {}

    void initializeCUDAFeature(CUDAFeature*& cudaFeature, const label_t maxLabelId, cudaStream_t stream) override;
};

template <typename TData, size_t VChannels>
__global__ void AGaussianFeature<TData, VChannels>::createFeatureInstance(CUDAGaussianFeature<TData, VChannels>** cudaFeature, LabelStatisticsGauss* labelStatistics, const label_t maxLabelId) {
    // For virtual functions to work, we need to execute this on the device.
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *cudaFeature = new CUDAGaussianFeature<TData, VChannels>(labelStatistics, maxLabelId);
    }
}

template <typename TData, size_t VChannels>
void AGaussianFeature<TData, VChannels>::initializeCUDAFeature(CUDAFeature*& cudaFeature, const label_t maxLabelId, cudaStream_t stream) {
    CUDA_SAFE_CALL(cudaMallocAsync(&cudaFeature, sizeof(CUDAGaussianFeature<TData, VChannels>), stream), this->logger);

    LabelStatisticsGauss* labelStatistics;
    CUDA_SAFE_CALL(cudaMallocAsync(&labelStatistics, sizeof(LabelStatisticsGauss) * VChannels * (maxLabelId + 1), stream), this->logger);
    CUDA_SAFE_CALL(cudaMemsetAsync(labelStatistics, 0, sizeof(LabelStatisticsGauss) * VChannels * (maxLabelId + 1), stream), this->logger);

    this->createFeatureInstance<<<1, 1, 0, stream>>>(reinterpret_cast<CUDAGaussianFeature<TData, VChannels>**>(cudaFeature), labelStatistics, maxLabelId);
}

__device__ inline void deviceUpdateLabelFeatureCost(LabelStatisticsGauss& labelStats) {
    // Compute the variance of the Gaussian distribution of the current label.
    // Cast the numerator of both divisions to double so that we get double precision in the result,
    // because the statistics are most likely stored as integers.
    double variance = (labelStats.squareValueSum / static_cast<double>(labelStats.pixelCount)) - SQUARED(labelStats.valueSum / static_cast<double>(labelStats.pixelCount));

    // Ensure variance is bigger than zero, else we could get -infinity
    // cost which screws up everything. Could happen to labels with only
    // a few pixels which all have the exact same grayvalue (or a label
    // with just one pixel).
    variance = max(variance, featuresMinVariance);

    labelStats.featureCost = (static_cast<double>(labelStats.pixelCount) / 2 * log(2 * M_PI * variance)) + (static_cast<double>(labelStats.pixelCount) / 2);
}

template <typename TData, size_t VChannels>
__device__ void setImageData(const cv::cuda::PtrStepSz<uint8_t> image) {
    if (DataType::Image != this->target) {
        return;
    }

    this->data = data;
}

template <typename TData, size_t VChannels>
__device__ void setDisparityData(const cv::cuda::PtrStepSz<int16_t> disparity) {
    if (DataType::Disparity != this->target) {
        return;
    }

    this->data = data;
}

template <typename TData, size_t VChannels>
__device__ void CUDAGaussianFeature<TData, VChannels>::initializeStatistics(const cv::cuda::PtrStepSz<label_t> labelImage, size_t xBatch, size_t yBatch) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int pixelX = x * X_BATCH;
    int pixelY = y * Y_BATCH;

    for (int j = 0; j < yBatch; ++j) {
        if (pixelY + j >= labelImage.rows) {
            break;
        }

        const label_t* const labelImageRowPtr = labelImage.ptr<label_t>(pixelX + j);
        const TData* const dataRowPtr = this->data.ptr<TData>(pixelX + j);

        for (int i = 0; i < xBatch; ++i) {
            const label_t curLabel = labelImageRowPtr[pixelX + i];
            LabelStatisticsGauss* rowLabelStats = &getLabelStatistics(curLabel, 0);

#pragma unroll
            for (size_t ch = 0; ch < VChannels; ++ch) {
                const TData curData = dataRowPtr[INDEX_ROW(pixelX + i, VChannels, ch)];
                LabelStatisticsGauss* out_labelStatistics = rowLabelStats + ch;

                atomicAdd(&out_labelStatistics->pixelCount, 1);
                atomicAdd(&out_labelStatistics->valueSum, curData);
                atomicAdd(&out_labelStatistics->squareValueSum, SQUARED(curData));

                // Update the feature cost of the label. This will be calculated a few too many times, but should be OK
                deviceUpdateLabelFeatureCost(*out_labelStatistics);
            }
        }
    }
}

template <typename TData, size_t VChannels>
__device__ double CUDAGaussianFeature<TData, VChannels>::calculateCost(const cv::Point2i curPixelCoords,
                                                                       const label_t oldLabel, const label_t pretendLabel,
                                                                       const label_t* neighbourLabels, const size_t neighbourLabelSize) const {
    // Modify the label statistics if the pixel at curPixelCoords has another
    // label than pretendLabel. We only modify a local copy of the statistics,
    // however, since we do not actually want to change a label yet. We are
    // just calculating the cost of doing so.
    // We only copy the affected statistics, namely the ones of the old and the
    // new (pretend) label. We have to handle these two cases separately when accessing
    // the statistics, because we specifically need to access the local, modified versions.
    // But we save a lot of time since we do not need to create a copy of all statistics.
    LabelStatisticsGauss labelStatsOldLabel[VChannels];
    LabelStatisticsGauss labelStatsPretendLabel[VChannels];

#pragma unroll
    for (size_t ch = 0; ch < VChannels; ++ch) {
        labelStatsOldLabel[ch] = getLabelStatistics(oldLabel, ch);
        labelStatsPretendLabel[ch] = getLabelStatistics(pretendLabel, ch);
    }

    if (oldLabel != pretendLabel) {
        this->updateStatistics(curPixelCoords, labelStatsOldLabel, labelStatsPretendLabel);
    }

    double featureCost = 0;

    for (size_t i = 0; i < neighbourLabelSize; i++) {
        // Get a pointer to the label statistics of the current label.
        // This should be the associated entry in the vector of label statistics, except for the
        // special cases below.
        LabelStatisticsGauss const* rowLabelStats = &getLabelStatistics(neighbourLabels[i], 0);

        // If the current label is the old one at curPixelCoords, or the pretended new one,
        // then its statistics have changed. In this case, we need to read the statistics
        // from the modified local variables.
        if (neighbourLabels[i] == oldLabel) {
            rowLabelStats = labelStatsOldLabel;
        } else if (neighbourLabels[i] == pretendLabel) {
            rowLabelStats = labelStatsPretendLabel;
        }

#pragma unroll
        for (int ch = 0; ch < VChannels; ch++) {
            // Get a pointer to the label statistics of the current label.
            LabelStatisticsGauss const* curLabelStats = rowLabelStats + ch;

            // If a label completely vanished, disregard it (can happen to old label of pixel_index).
            if (curLabelStats->pixelCount == 0) {
                continue;
            }

            featureCost += curLabelStats->featureCost;
        }
    }

    return featureCost;
}

template <typename TData, size_t VChannels>
__device__ void CUDAGaussianFeature<TData, VChannels>::updateStatistics(const cv::Point2i const& curPixelCoords, label_t const& oldLabel, label_t const& newLabel) {
    auto rowPtr = data.ptr<TData>(curPixelCoords.y);

#pragma unroll
    for (int ch = 0; ch < VChannels; ch++) {
        // Update pixel count.
        atomicSub(&labelStatsOldLabel[ch].pixelCount, 1);
        atomicAdd(&labelStatsNewLabel[ch].pixelCount, 1);

        auto dataValue = rowPtr[INDEX_ROW(curPixelCoords.x, VChannels, ch)];

        // Update grayvalue sum.
        atomicSub(&labelStatsOldLabel[ch].valueSum, dataValue);
        atomicAdd(&labelStatsNewLabel[ch].valueSum, dataValue);

        auto dataValueSquared = dataValue * dataValue;

        // Update square grayvalue sum.
        atomicSub(&labelStatsOldLabel[ch].squareValueSum, dataValueSquared);
        atomicAdd(&labelStatsNewLabel[ch].squareValueSum, dataValueSquared);

        deviceUpdateLabelFeatureCost(labelStatsOldLabel[ch]);
        deviceUpdateLabelFeatureCost(labelStatsNewLabel[ch]);
    }
}
}  // namespace cart::contour