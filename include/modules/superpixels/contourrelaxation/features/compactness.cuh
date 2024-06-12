// Copyright 2013 Visual Sensorics and Information Processing Lab, Goethe University, Frankfurt
//
// This file is part of Contour-relaxed Superpixels.
//
// Contour-relaxed Superpixels is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// Contour-relaxed Superpixels is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with Contour-relaxed Superpixels.  If not, see <http://www.gnu.org/licenses/>.

#pragma once

#include "feature.cuh"
#include "gaussian.cuh"

namespace cart::contour {
/**
 * @class CompactnessFeature
 * @brief Feature class for computing a cost based on the spatial distribution of a label, to enforce compactness.
 */
class CUDACompactnessFeature : public CUDAFeature {
   private:
    LabelStatisticsGauss* labelStatisticsPosX;
    LabelStatisticsGauss* labelStatisticsPosY;
    label_t maxLabelId;
    double progressiveCost;

    /**
     * @brief Update label statistics to reflect a label change of the given pixel.
     * @param curPixelCoords coordinates of the pixel changing its label
     * @param oldLabel old label of the regarded pixel
     * @param newLabel new label of the regarded pixel
     */
    __device__ void updateStatistics(CRPoint const curPixelCoords,
                                     LabelStatisticsGauss& labelStatsOldLabelPosX, LabelStatisticsGauss& labelStatsNewLabelPosX,
                                     LabelStatisticsGauss& labelStatsOldLabelPosY, LabelStatisticsGauss& labelStatsNewLabelPosY) const;

   public:
    __device__ CUDACompactnessFeature(label_t maxLabelId, double progressiveCost);

    __device__ ~CUDACompactnessFeature() override;

    __device__ void initializeStatistics(const cv::cuda::PtrStepSz<label_t> labelImage, size_t xBatch, size_t yBatch) override;

    __device__ double calculateCost(const CRPoint curPixelCoords,
                                    const label_t oldLabel, const label_t pretendLabel,
                                    const label_t* neighbourLabels, const size_t neighbourLabelSize) const override;

    __device__ void updateStatistics(const CRPoint curPixelCoords, label_t const& oldLabel, label_t const& newLabel) override;
};

class CompactnessFeature : public IFeature {
   private:
    const double progressiveCost;

   public:
    CompactnessFeature(const double progressiveCost = 1.0) : IFeature("Compactness"), progressiveCost(progressiveCost) {}

    void initializeCUDAFeature(CUDAFeature**& cudaFeature, const label_t maxLabelId, const cv::cuda::Stream& stream = cv::cuda::Stream::Null()) override;
};
}  // namespace cart::contour
