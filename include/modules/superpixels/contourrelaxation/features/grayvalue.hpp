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

#include "../constants.hpp"
#include "gaussian.hpp"
#include "ifeature.hpp"

namespace cart::contour {
/**
 * @class GrayvalueFeature
 * @brief Feature of one Gaussian-distributed channel, intended for use with grayvalue images.
 */
class GrayvalueFeature : public AGaussianFeature {
   private:
    typedef uchar TGrayvalueData;  ///< the type of the used grayvalue images

    std::vector<LabelStatisticsGauss> labelStatistics;  ///< Gaussian label statistics of the grayvalue image
    cv::Mat grayvalImage;                               ///< the observed grayvalue data

   public:
    /**
     * @brief Assign new data to the feature object.
     * @param grayvalueImage observed grayvalue data
     */
    void setData(cv::Mat const& grayvalueImage);

    /**
     * @brief Estimate the label statistics of all labels in the given label image, using the observed data saved in the feature object.
     * @param labelImage label identifiers of all pixels
     */
    void initializeStatistics(cv::Mat const& labelImage);

    /**
     * @brief Calculate the total cost of all labels in the 8-neighbourhood of a pixel, assuming the pixel would change its label.
     * @param curPixelCoords coordinates of the regarded pixel
     * @param oldLabel old label of the regarded pixel
     * @param pretendLabel assumed new label of the regarded pixel
     * @param neighbourLabels all labels found in the 8-neighbourhood of the regarded pixel, including the old label of the pixel itself
     * @return total negative log-likelihood (or cost) of all labels in the neighbourhood, assuming the label change
     */
    double calculateCost(cv::Point2i const& curPixelCoords,
                         label_t const& oldLabel, label_t const& pretendLabel, std::vector<label_t> const& neighbourLabels) const;

    /**
     * @brief Update the saved label statistics to reflect a label change of the given pixel.
     * @param curPixelCoords coordinates of the pixel whose label changes
     * @param oldLabel old label of the changing pixel
     * @param newLabel new label of the changing pixel
     */
    void updateStatistics(cv::Point2i const& curPixelCoords, label_t const& oldLabel, label_t const& newLabel);

    /**
     * @brief Create a representation of the current image visualizing each pixel with the mean grayvalue of its assigned label.
     * @param labelImage label identifier of each pixel
     * @param out_regionMeanImage will be (re)allocated if necessary and filled with the described visualization
     */
    void generateRegionMeanImage(cv::Mat const& labelImage, cv::Mat& out_regionMeanImage) const;
};

}  // namespace cart::contour