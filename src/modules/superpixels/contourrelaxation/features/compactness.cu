#include "modules/superpixels/contourrelaxation/features/compactness.cuh"

namespace cart::contour {

__device__ CUDACompactnessFeature::CUDACompactnessFeature(label_t maxLabelId, double progressiveCost) : maxLabelId(maxLabelId), progressiveCost(progressiveCost) {
    labelStatisticsPosX = new LabelStatisticsGauss[maxLabelId + 1]();
    labelStatisticsPosY = new LabelStatisticsGauss[maxLabelId + 1]();
}

__device__ CUDACompactnessFeature::~CUDACompactnessFeature() {
    delete[] labelStatisticsPosX;
    delete[] labelStatisticsPosY;
}

__global__ void createFeatureInstance(CUDAFeature** cudaFeature, const label_t maxLabelId, const double progressiveCost) {
    // For virtual functions to work, we need to execute this on the device.
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        (*cudaFeature) = new CUDACompactnessFeature(maxLabelId, progressiveCost);
    }
}

void CompactnessFeature::initializeCUDAFeature(CUDAFeature**& cudaFeature, const label_t maxLabelId, const cv::cuda::Stream& cvStream) {
    cudaStream_t stream = cv::cuda::StreamAccessor::getStream(cvStream);
    createFeatureInstance<<<1, 1, 0, stream>>>(cudaFeature, maxLabelId, this->progressiveCost);
    CUDA_SAFE_CALL(this->logger, cudaPeekAtLastError());
}

__device__ inline void updateCompactnessCost(LabelStatisticsGauss& labelStats) {
    if (labelStats.pixelCount == 0) {
        labelStats.featureCost = 0;
        return;
    }

    labelStats.featureCost = labelStats.squareValueSum - (SQUARED(labelStats.valueSum) / static_cast<double>(labelStats.pixelCount));
}

__device__ void CUDACompactnessFeature::updateStatistics(CRPoint const curPixelCoords,
                                                         LabelStatisticsGauss& labelStatsOldLabelPosX, LabelStatisticsGauss& labelStatsNewLabelPosX,
                                                         LabelStatisticsGauss& labelStatsOldLabelPosY, LabelStatisticsGauss& labelStatsNewLabelPosY) const {
    // Update pixel count.
    atomicSub(&labelStatsOldLabelPosX.pixelCount, 1);
    atomicAdd(&labelStatsNewLabelPosX.pixelCount, 1);

    // Update x-position sum.
    atomicSub(&labelStatsOldLabelPosX.valueSum, static_cast<double>(curPixelCoords.x));
    atomicAdd(&labelStatsNewLabelPosX.valueSum, static_cast<double>(curPixelCoords.x));

    // Update square x-position sum.
    atomicSub(&labelStatsOldLabelPosX.squareValueSum, SQUARED(curPixelCoords.x));
    atomicAdd(&labelStatsNewLabelPosX.squareValueSum, SQUARED(curPixelCoords.x));

    // Update pixel count.
    atomicSub(&labelStatsOldLabelPosY.pixelCount, 1);
    atomicAdd(&labelStatsNewLabelPosY.pixelCount, 1);

    // Update x-position sum.
    atomicSub(&labelStatsOldLabelPosY.valueSum, static_cast<double>(curPixelCoords.y));
    atomicAdd(&labelStatsNewLabelPosY.valueSum, static_cast<double>(curPixelCoords.y));

    // Update square x-position sum.
    atomicSub(&labelStatsOldLabelPosY.squareValueSum, SQUARED(curPixelCoords.y));
    atomicAdd(&labelStatsNewLabelPosY.squareValueSum, SQUARED(curPixelCoords.y));

    updateCompactnessCost(labelStatsOldLabelPosX);
    updateCompactnessCost(labelStatsNewLabelPosX);
    updateCompactnessCost(labelStatsOldLabelPosY);
    updateCompactnessCost(labelStatsNewLabelPosY);
}

__device__ void CUDACompactnessFeature::initializeStatistics(const cv::cuda::PtrStepSz<label_t> labelImage, size_t xBatch, size_t yBatch) {
    // Set the image height in a single thread
    if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0 && threadIdx.y == 0) {
        height = labelImage.rows;
    }

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int pixelX = x * xBatch;
    int pixelY = y * yBatch;

    for (int j = 0; j < yBatch; ++j) {
        if (pixelY + j >= labelImage.rows) {
            break;
        }

        const label_t* const labelImageRowPtr = labelImage.ptr(pixelY + j);

        for (int i = 0; i < xBatch; ++i) {
            const label_t curLabel = labelImageRowPtr[pixelX + i];

            atomicAdd(&labelStatisticsPosX[curLabel].pixelCount, 1);
            atomicAdd(&labelStatisticsPosX[curLabel].valueSum, pixelX + i);
            atomicAdd(&labelStatisticsPosX[curLabel].squareValueSum, SQUARED(pixelX + i));
            updateCompactnessCost(labelStatisticsPosX[curLabel]);

            atomicAdd(&labelStatisticsPosY[curLabel].pixelCount, 1);
            atomicAdd(&labelStatisticsPosY[curLabel].valueSum, pixelY + j);
            atomicAdd(&labelStatisticsPosY[curLabel].squareValueSum, SQUARED(pixelY + j));
            updateCompactnessCost(labelStatisticsPosY[curLabel]);
        }
    }
}

__device__ double CUDACompactnessFeature::calculateCost(const CRPoint curPixelCoords,
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
    LabelStatisticsGauss labelStatsPosXOldLabel(labelStatisticsPosX[oldLabel]);
    LabelStatisticsGauss labelStatsPosXPretendLabel(labelStatisticsPosX[pretendLabel]);

    LabelStatisticsGauss labelStatsPosYOldLabel(labelStatisticsPosY[oldLabel]);
    LabelStatisticsGauss labelStatsPosYPretendLabel(labelStatisticsPosY[pretendLabel]);

    if (oldLabel != pretendLabel) {
        // Update pixel count.
        labelStatsPosXOldLabel.pixelCount--;
        labelStatsPosXPretendLabel.pixelCount++;

        // Update x-position sum.
        labelStatsPosXOldLabel.valueSum -= curPixelCoords.x;
        labelStatsPosXPretendLabel.valueSum += curPixelCoords.x;

        // Update square x-position sum.
        labelStatsPosXOldLabel.squareValueSum -= SQUARED(curPixelCoords.x);
        labelStatsPosXPretendLabel.squareValueSum += SQUARED(curPixelCoords.x);

        // Update pixel count.
        labelStatsPosYOldLabel.pixelCount--;
        labelStatsPosYPretendLabel.pixelCount++;

        // Update x-position sum.
        labelStatsPosYOldLabel.valueSum -= curPixelCoords.y;
        labelStatsPosYPretendLabel.valueSum += curPixelCoords.y;

        // Update square x-position sum.
        labelStatsPosYOldLabel.squareValueSum -= SQUARED(curPixelCoords.y);
        labelStatsPosYPretendLabel.squareValueSum += SQUARED(curPixelCoords.y);

        updateCompactnessCost(labelStatsPosXOldLabel);
        updateCompactnessCost(labelStatsPosXPretendLabel);
        updateCompactnessCost(labelStatsPosYOldLabel);
        updateCompactnessCost(labelStatsPosYPretendLabel);
    }

    double featureCost = 0;

    // For each neighbouring label, add its cost.
    for (size_t i = 0; i < neighbourLabelSize; i++) {
        // Get a pointer to the label statistics of the current label.
        // This should be the associated entry in the vector of label statistics, except for the
        // special cases below.
        LabelStatisticsGauss const* curLabelStatsPosX = &labelStatisticsPosX[neighbourLabels[i]];
        LabelStatisticsGauss const* curLabelStatsPosY = &labelStatisticsPosY[neighbourLabels[i]];

        // If the current label is the old one at curPixelCoords, or the pretended new one,
        // then its statistics have changed. In this case, we need to read the statistics
        // from the modified local variables.
        if (neighbourLabels[i] == oldLabel) {
            curLabelStatsPosX = &labelStatsPosXOldLabel;
            curLabelStatsPosY = &labelStatsPosYOldLabel;
        } else if (neighbourLabels[i] == pretendLabel) {
            curLabelStatsPosX = &labelStatsPosXPretendLabel;
            curLabelStatsPosY = &labelStatsPosYPretendLabel;
        }

        // If a label completely vanished, disregard it (can happen to old label of pixel_index).
        if (curLabelStatsPosX->pixelCount == 0) {
            continue;
        }

        // Add the cost of the current region.
        featureCost += curLabelStatsPosX->featureCost + curLabelStatsPosY->featureCost;
    }

    // Increase the cost for the pixels that are at the top of the image, as these should
    // naturally be more compact.
    if (this->progressiveCost > 0.0) {
        // Note that y is the row and starts at 0 at the top of the image.
        featureCost *= 1.0 + this->progressiveCost * (this->height - static_cast<double>(curPixelCoords.y)) / static_cast<double>(height);
    }

    return featureCost;
}

__device__ void CUDACompactnessFeature::updateStatistics(const CRPoint curPixelCoords, label_t const& oldLabel,
                                                         label_t const& newLabel) {
    updateStatistics(curPixelCoords, labelStatisticsPosX[oldLabel], labelStatisticsPosX[newLabel],
                     labelStatisticsPosY[oldLabel], labelStatisticsPosY[newLabel]);
}
}  // namespace cart::contour