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

#include <boost/cstdint.hpp>
#include <opencv2/opencv.hpp>
#include <vector>

#include "../constants.hpp"
#include "ifeature.hpp"

namespace cart::contour {

/**
 * @struct LabelStatisticsGauss
 * @brief Struct containing the sufficient statistics needed to compute label likelihoods of a Gaussian-distributed feature channel.
 */
struct LabelStatisticsGauss {
    boost::uint_fast32_t pixelCount;  ///< the number of pixels assigned to the label
    double valueSum;                  ///< the sum of values of all pixels assigned to the label
    double squareValueSum;            ///< the sum of squared values of all pixels assigned to the label

    /**
     * @brief Default constructor to ensure that all members are initialized to sensible values.
     */
    LabelStatisticsGauss() : pixelCount(0), valueSum(0), squareValueSum(0) {}
};

/**
 * @class AGaussianFeature
 * @brief Abstract feature class providing the basic functionality needed for all features with Gaussian distribution.
 */
class AGaussianFeature : public IFeature {
   protected:
    /**
     * @brief Update label statistics of a Gaussian distribution to reflect a label change of the given pixel.
     * @param curPixelCoords coordinates of the pixel whose label changes
     * @param labelStatsOldLabel statistics of the old label of the changing pixel (will be updated)
     * @param labelStatsNewLabel statistics of the new label of the changing pixel (will be updated)
     * @param data observed data of the distribution whose statistics are being modelled
     */
    template <typename TData>
    void updateGaussianStatistics(cv::Point2i const& curPixelCoords, LabelStatisticsGauss& labelStatsOldLabel,
                                  LabelStatisticsGauss& labelStatsNewLabel, cv::Mat const& data) const;

    /**
     * @brief Estimate the statistics of a Gaussian distribution for each label on the given observed data.
     * @param labelImage contains the label identifier to which each pixel is assigned
     * @param data observed data of the Gaussian distributions
     * @param out_labelStatistics will be created and contain the label statistics of all labels in labelImage
     */
    template <typename TData>
    void initializeGaussianStatistics(const cv::Mat& labelImage, const label_t maxLabelId, const cv::Mat& data,
                                      std::vector<LabelStatisticsGauss>& out_labelStatistics) const;
    /**
     * @brief Calculate the total cost of all labels in the 8-neighbourhood of a pixel, assuming the pixel would change its label.
     * @param curPixelCoords coordinates of the regarded pixel
     * @param oldLabel old label of the regarded pixel
     * @param pretendLabel assumed new label of the regarded pixel
     * @param neighbourLabels all labels found in the 8-neighbourhood of the regarded pixel, including the old label of the pixel itself
     * @param labelStatistics label statistics of all labels in the image
     * @param data observed data of the modelled Gaussian distributions
     * @return total negative log-likelihood (or cost) of all labels in the neighbourhood, assuming the label change
     */
    template <typename TData>
    double calculateGaussianCost(cv::Point2i const& curPixelCoords,
                                 label_t const& oldLabel, label_t const& pretendLabel, std::vector<label_t> const& neighbourLabels,
                                 std::vector<LabelStatisticsGauss> const& labelStatistics, cv::Mat const& data) const;

   public:
    /**
     * @brief Provide a virtual destructor so instances of derived classes can be safely destroyed through a pointer or reference of type AGaussianFeature.
     */
    virtual ~AGaussianFeature() {}
};

template <typename TData>
void AGaussianFeature::updateGaussianStatistics(cv::Point2i const& curPixelCoords,
                                                LabelStatisticsGauss& labelStatsOldLabel, LabelStatisticsGauss& labelStatsNewLabel, cv::Mat const& data) const {
    assert(curPixelCoords.inside(cv::Rect(0, 0, data.cols, data.rows)));
    assert(data.type() == cv::DataType<TData>::type);

    // Update pixel count.
    labelStatsOldLabel.pixelCount--;
    labelStatsNewLabel.pixelCount++;

    // Update grayvalue sum.
    labelStatsOldLabel.valueSum -= data.at<TData>(curPixelCoords);
    labelStatsNewLabel.valueSum += data.at<TData>(curPixelCoords);

    // Update square grayvalue sum.
    labelStatsOldLabel.squareValueSum -= pow(static_cast<double>(data.at<TData>(curPixelCoords)), 2.0);
    labelStatsNewLabel.squareValueSum += pow(static_cast<double>(data.at<TData>(curPixelCoords)), 2.0);
}

template <typename TData>
void AGaussianFeature::initializeGaussianStatistics(const cv::Mat& labelImage, const label_t maxLabelId, const cv::Mat& data,
                                                    std::vector<LabelStatisticsGauss>& out_labelStatistics) const {
    assert(labelImage.size() == data.size());
    assert(labelImage.type() == cv::DataType<label_t>::type);
    assert(data.type() == cv::DataType<TData>::type);

    // Allocate the vector of label statistics, with the maximum index being the maximum label identifier.
    // This might waste a small amount of memory, but we can use the label identifier as index for this vector.
    out_labelStatistics = std::vector<LabelStatisticsGauss>(maxLabelId + 1, LabelStatisticsGauss());

    for (int row = 0; row < labelImage.rows; ++row) {
        const label_t* const labelImageRowPtr = labelImage.ptr<label_t>(row);
        const TData* const dataRowPtr = data.ptr<TData>(row);

        for (int col = 0; col < labelImage.cols; ++col) {
            const label_t curLabel = labelImageRowPtr[col];
            const TData curData = dataRowPtr[col];

            out_labelStatistics[curLabel].pixelCount++;
            out_labelStatistics[curLabel].valueSum += curData;
            out_labelStatistics[curLabel].squareValueSum += curData * curData;
        }
    }
}

template <typename TData>
double AGaussianFeature::calculateGaussianCost(cv::Point2i const& curPixelCoords,
                                               label_t const& oldLabel, label_t const& pretendLabel, std::vector<label_t> const& neighbourLabels,
                                               std::vector<LabelStatisticsGauss> const& labelStatistics, cv::Mat const& data) const {
    assert(curPixelCoords.inside(cv::Rect(0, 0, data.cols, data.rows)));
    assert(data.type() == cv::DataType<TData>::type);

    // Modify the label statistics if the pixel at curPixelCoords has another
    // label than pretendLabel. We only modify a local copy of the statistics,
    // however, since we do not actually want to change a label yet. We are
    // just calculating the cost of doing so.
    // We only copy the affected statistics, namely the ones of the old and the
    // new (pretend) label. We have to handle these two cases separately when accessing
    // the statistics, because we specifically need to access the local, modified versions.
    // But we save a lot of time since we do not need to create a copy of all statistics.
    LabelStatisticsGauss labelStatsOldLabel(labelStatistics[oldLabel]);
    LabelStatisticsGauss labelStatsPretendLabel(labelStatistics[pretendLabel]);

    if (oldLabel != pretendLabel) {
        updateGaussianStatistics<TData>(curPixelCoords, labelStatsOldLabel, labelStatsPretendLabel, data);
    }

    double featureCost = 0;

    // For each neighbouring label, add its cost.
    for (typename std::vector<label_t>::const_iterator it_neighbourLabel = neighbourLabels.begin();
         it_neighbourLabel != neighbourLabels.end(); ++it_neighbourLabel) {
        // Get a pointer to the label statistics of the current label.
        // This should be the associated entry in the vector of label statistics, except for the
        // special cases below.
        LabelStatisticsGauss const* curLabelStats = &labelStatistics[*it_neighbourLabel];

        // If the current label is the old one at curPixelCoords, or the pretended new one,
        // then its statistics have changed. In this case, we need to read the statistics
        // from the modified local variables.
        if (*it_neighbourLabel == oldLabel) {
            curLabelStats = &labelStatsOldLabel;
        } else if (*it_neighbourLabel == pretendLabel) {
            curLabelStats = &labelStatsPretendLabel;
        }

        // If a label completely vanished, disregard it (can happen to old label of pixel_index).
        if (curLabelStats->pixelCount == 0) {
            continue;
        }

        // Compute the variance of the Gaussian distribution of the current label.
        // Cast the numerator of both divisions to double so that we get double precision in the result,
        // because the statistics are most likely stored as integers.
        double variance = (curLabelStats->squareValueSum / curLabelStats->pixelCount) - pow(curLabelStats->valueSum / curLabelStats->pixelCount, 2.0);

        // Ensure variance is bigger than zero, else we could get -infinity
        // cost which screws up everything. Could happen to labels with only
        // a few pixels which all have the exact same grayvalue (or a label
        // with just one pixel).
        variance = std::max(variance, featuresMinVariance);

        // Add the cost of the current region.
        featureCost += (static_cast<double>(curLabelStats->pixelCount) / 2 * log(2 * M_PI * variance)) + (static_cast<double>(curLabelStats->pixelCount) / 2);
    }

    return featureCost;
}
}  // namespace cart::contour