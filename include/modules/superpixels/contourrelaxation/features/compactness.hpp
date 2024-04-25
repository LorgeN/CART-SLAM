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

#include <opencv2/opencv.hpp>
#include <vector>

#include "gaussian.hpp"
#include "ifeature.hpp"

namespace cart::contour {
/**
 * @class CompactnessFeature
 * @brief Feature class for computing a cost based on the spatial distribution of a label, to enforce compactness.
 */
class CompactnessFeature : public IFeature {
   private:
    std::vector<LabelStatisticsGauss> labelStatisticsPosX;  ///< the statistics of x-positions of all labels
    std::vector<LabelStatisticsGauss> labelStatisticsPosY;  ///< the statistics of y-positions of all labels

    /**
     * @brief Update label statistics to reflect a label change of the given pixel.
     * @param curPixelCoords coordinates of the pixel changing its label
     * @param oldLabel old label of the regarded pixel
     * @param newLabel new label of the regarded pixel
     */
    void updateStatistics(cv::Point2i const& curPixelCoords,
                          LabelStatisticsGauss& labelStatsOldLabelPosX, LabelStatisticsGauss& labelStatsNewLabelPosX,
                          LabelStatisticsGauss& labelStatsOldLabelPosY, LabelStatisticsGauss& labelStatsNewLabelPosY) const;

   public:
    /**
     * @brief Estimate the statistics of the spatial distribution for each label.
     * @param labelImage contains the label identifier to which each pixel is assigned
     */
    void initializeStatistics(const cv::Mat& labelImage, const label_t maxLabelId) override;

    /**
     * @brief Calculate the total cost of all labels in the 8-neighbourhood of a pixel, assuming the pixel would change its label.
     * @param curPixelCoords coordinates of the regarded pixel
     * @param oldLabel old label of the regarded pixel
     * @param pretendLabel assumed new label of the regarded pixel
     * @param neighbourLabels all labels found in the 8-neighbourhood of the regarded pixel, including the old label of the pixel itself
     * @return weighted total cost of all labels in the 8-neighbourhood
     *
     * The cost is in this case not defined by a probabilistic distribution, but as the
     * sum over the squared distance of each pixel of all regarded labels from the spatial center of the pixel's label.
     * Because of this definition, we need a weight to adjust how much this cost should influence the total cost.
     * The usual assumption that we have a likelihood which is independent of all other features and can just be
     * added (since we use log-likelihoods) does not hold here.
     */
    double calculateCost(cv::Point2i const& curPixelCoords,
                         label_t const& oldLabel, label_t const& pretendLabel, std::vector<label_t> const& neighbourLabels) const override;

    /**
     * @brief Update label statistics to reflect a label change of the given pixel.
     * @param curPixelCoords coordinates of the pixel changing its label
     * @param labelStatsOldLabelPosX label statistics for x-positions of the old label (will be updated)
     * @param labelStatsNewLabelPosX label statistics for x-positions of the new label (will be updated)
     * @param labelStatsOldLabelPosY label statistics for y-positions of the old label (will be updated)
     * @param labelStatsNewLabelPosY label statistics for y-positions of the new label (will be updated)
     */
    void updateStatistics(cv::Point2i const& curPixelCoords, label_t const& oldLabel, label_t const& newLabel) override;
};
}  // namespace cart::contour
