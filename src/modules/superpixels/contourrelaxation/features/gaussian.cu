#include "modules/superpixels/contourrelaxation/features/gaussian.cuh"

namespace cart::contour {

template <typename TData, size_t VChannels, DataType Type>
__device__ CUDAGaussianFeature<TData, VChannels, Type>::CUDAGaussianFeature(const label_t maxLabelId) : maxLabelId(maxLabelId) {
    this->labelStatistics = new LabelStatisticsGauss[VChannels * (maxLabelId + 1)]();
}

template <typename TData, size_t VChannels, DataType Type>
__device__ CUDAGaussianFeature<TData, VChannels, Type>::~CUDAGaussianFeature() {
    delete[] labelStatistics;
}

template <typename TData, size_t VChannels, DataType Type>
__global__ void createFeatureInstance(CUDAFeature** cudaFeature, const label_t maxLabelId) {
    // For virtual functions to work, we need to execute this on the device.
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *cudaFeature = new CUDAGaussianFeature<TData, VChannels, Type>(maxLabelId);
    }
}

template <typename TData, size_t VChannels, DataType Type>
void AGaussianFeature<TData, VChannels, Type>::initializeCUDAFeature(CUDAFeature**& cudaFeature, const label_t maxLabelId, const cv::cuda::Stream& cvStream) {
    cudaStream_t stream = cv::cuda::StreamAccessor::getStream(cvStream);
    createFeatureInstance<TData, VChannels, Type><<<1, 1, 0, stream>>>(cudaFeature, maxLabelId);
    CUDA_SAFE_CALL(this->logger, cudaPeekAtLastError());
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

template <typename TData, size_t VChannels, DataType Type>
__device__ void CUDAGaussianFeature<TData, VChannels, Type>::setImageData(const cv::cuda::PtrStepSz<uint8_t> image) {
    if constexpr (DataType::Image == Type) {
        data = image;
    }
}

template <typename TData, size_t VChannels, DataType Type>
__device__ void CUDAGaussianFeature<TData, VChannels, Type>::setDisparityData(const cv::cuda::PtrStepSz<int16_t> disparity) {
    if constexpr (DataType::Disparity == Type) {
        data = disparity;
    }
}

template <typename TData, size_t VChannels, DataType Type>
__device__ void CUDAGaussianFeature<TData, VChannels, Type>::initializeStatistics(const cv::cuda::PtrStepSz<label_t> labelImage, size_t xBatch, size_t yBatch) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int pixelX = x * xBatch;
    int pixelY = y * yBatch;

    for (int j = 0; j < yBatch; ++j) {
        if (pixelY + j >= labelImage.rows) {
            break;
        }

        const label_t* const labelImageRowPtr = labelImage.ptr(pixelY + j);
        const TData* const dataRowPtr = data.ptr(pixelY + j);

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

template <typename TData, size_t VChannels, DataType Type>
__device__ double CUDAGaussianFeature<TData, VChannels, Type>::calculateCost(const CRPoint curPixelCoords,
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
        auto rowPtr = data.ptr(curPixelCoords.y);

#pragma unroll
        for (int ch = 0; ch < VChannels; ch++) {
            // Update pixel count.
            labelStatsOldLabel[ch].pixelCount--;
            labelStatsPretendLabel[ch].pixelCount++;

            double dataValue = rowPtr[INDEX_ROW(curPixelCoords.x, VChannels, ch)];

            // Update grayvalue sum.
            labelStatsOldLabel[ch].valueSum -= dataValue;
            labelStatsPretendLabel[ch].valueSum += dataValue;

            double dataValueSquared = SQUARED(dataValue);

            // Update square grayvalue sum.
            labelStatsOldLabel[ch].squareValueSum -= dataValueSquared;
            labelStatsPretendLabel[ch].squareValueSum += dataValueSquared;

            deviceUpdateLabelFeatureCost(labelStatsOldLabel[ch]);
            deviceUpdateLabelFeatureCost(labelStatsPretendLabel[ch]);
        }
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

    return featureCost / static_cast<double>(VChannels);
}

template <typename TData, size_t VChannels, DataType Type>
__device__ void CUDAGaussianFeature<TData, VChannels, Type>::updateStatistics(const CRPoint curPixelCoords, label_t const& oldLabel, label_t const& newLabel) {
    this->updateGaussianStatistics(curPixelCoords, &getLabelStatistics(oldLabel, 0), &getLabelStatistics(newLabel, 0));
}

template <typename TData, size_t VChannels, DataType Type>
__device__ void CUDAGaussianFeature<TData, VChannels, Type>::updateGaussianStatistics(const CRPoint curPixelCoords, LabelStatisticsGauss* labelStatsOldLabel, LabelStatisticsGauss* labelStatsNewLabel) const {
    auto rowPtr = data.ptr(curPixelCoords.y);

#pragma unroll
    for (int ch = 0; ch < VChannels; ch++) {
        // Update pixel count.
        atomicSub(&labelStatsOldLabel[ch].pixelCount, 1);
        atomicAdd(&labelStatsNewLabel[ch].pixelCount, 1);

        double dataValue = rowPtr[INDEX_ROW(curPixelCoords.x, VChannels, ch)];

        // Update grayvalue sum.
        atomicSub(&labelStatsOldLabel[ch].valueSum, dataValue);
        atomicAdd(&labelStatsNewLabel[ch].valueSum, dataValue);

        double dataValueSquared = dataValue * dataValue;

        // Update square grayvalue sum.
        atomicSub(&labelStatsOldLabel[ch].squareValueSum, dataValueSquared);
        atomicAdd(&labelStatsNewLabel[ch].squareValueSum, dataValueSquared);

        deviceUpdateLabelFeatureCost(labelStatsOldLabel[ch]);
        deviceUpdateLabelFeatureCost(labelStatsNewLabel[ch]);
    }
}

// Explicit template instantiations, so that we can keep source code out of the header file.
template class AGaussianFeature<uint8_t, 1, DataType::Image>;
template class AGaussianFeature<uint8_t, 3, DataType::Image>;
template class AGaussianFeature<int16_t, 2, DataType::Disparity>;
}  // namespace cart::contour