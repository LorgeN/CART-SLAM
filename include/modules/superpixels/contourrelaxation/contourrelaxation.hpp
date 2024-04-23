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

#include <assert.h>
#include <math.h>

#include <algorithm>
#include <boost/cstdint.hpp>
#include <boost/shared_ptr.hpp>
#include <opencv2/opencv.hpp>
#include <vector>

#include "constants.hpp"
#include "features/color.hpp"
#include "features/compactness.hpp"
#include "features/grayvalue.hpp"
#include "features/type.hpp"
#include "initialization.hpp"
#include "traversion.hpp"

namespace cart::contour {
/**
 * @class ContourRelaxation
 * @brief Main class for applying Contour Relaxation to a label image, using an arbitrary set of features.
 */
class ContourRelaxation {
   private:
    boost::shared_ptr<GrayvalueFeature> grayvalueFeature;  ///< Pointer to the grayvalue feature object, if enabled.
    bool grayvalueFeatureEnabled;                          ///< True if grayvalue feature is enabled.

    boost::shared_ptr<ColorFeature> colorFeature;  ///< Pointer to the color feature object, if enabled.
    bool colorFeatureEnabled;                      ///< True if color feature is enabled.

    boost::shared_ptr<CompactnessFeature> compactnessFeature;  ///< Pointer to the compactness feature object, if enabled.
    bool compactnessFeatureEnabled;                            ///< True if compactness feature is enabled.

    std::vector<boost::shared_ptr<IFeature>> allFeatures;                                       ///< Vector of pointers to all enabled feature objects.
    typedef typename std::vector<boost::shared_ptr<IFeature>>::const_iterator FeatureIterator;  ///< Shorthand for const_iterator over vector of feature pointers.

    std::vector<label_t> getNeighbourLabels(cv::Mat const& labelImage, cv::Point2i const& curPixelCoords) const;

    double calculateCost(cv::Mat const& labelImage, cv::Point2i const& curPixelCoords,
                         label_t const& pretendLabel, std::vector<label_t> const& neighbourLabels) const;

    double calculateCliqueCost(cv::Mat const& labelImage, cv::Point2i const& curPixelCoords, label_t const& pretendLabel) const;

    void computeBoundaryMap(cv::Mat const& labelImage, cv::Mat& out_boundaryMap) const;

    void computeBoundaryMapSmall(cv::Mat const& labelImage, cv::Mat& out_boundaryMap) const;

    void updateBoundaryMap(cv::Mat const& labelImage, cv::Point2i const& curPixelCoords, cv::Mat& boundaryMap) const;

    cv::Mat labelImage;
    cv::Mat boundaryMap;
    const double directCliqueCost;
    const double diagonalCliqueCost;
    label_t maxLabelId;

   public:
    ContourRelaxation(std::vector<FeatureType> features, const cv::Mat initialLabelImage, const double directCliqueCost,
                      const double diagonalCliqueCost);

    void relax(unsigned int const numIterations, cv::OutputArray out_labelImage);

    void setData(const cv::Mat& image);

    void setCompactnessData(double const& compactnessWeight);
};
}  // namespace cart::contour