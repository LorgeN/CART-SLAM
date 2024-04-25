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
#include <boost/make_shared.hpp>
#include <boost/shared_ptr.hpp>
#include <opencv2/opencv.hpp>
#include <vector>

#include "constants.hpp"
#include "features/color.hpp"
#include "features/compactness.hpp"
#include "features/disparity.hpp"
#include "features/grayvalue.hpp"
#include "initialization.hpp"
#include "traversion.hpp"

namespace cart::contour {
/**
 * @brief Struct for storing a feature and its weight.
 */
struct FeatureContainer {
    boost::shared_ptr<IFeature> feature;
    double weight;
};

typedef typename std::vector<FeatureContainer>::const_iterator FeatureIterator;  ///< Shorthand for const_iterator over vector of features

/**
 * @class ContourRelaxation
 * @brief Main class for applying Contour Relaxation to a label image, using an arbitrary set of features.
 */
class ContourRelaxation {
   private:
    std::vector<FeatureContainer> features;

    /**
     * @brief Get all labels in the 8-neighbourhood of a pixel, including the label of the center pixel itself.
     * @param labelImage the current label image, contains one label identifier per pixel
     * @param curPixelCoords the coordinates of the regarded pixel
     * @return a vector containing all labels in the neighbourhood, each only once, sorted in ascending order
     */
    std::vector<label_t> getNeighbourLabels(cv::Mat const& labelImage, cv::Point2i const& curPixelCoords) const;

    /**
     * @brief Calculate the total cost of all labels in the 8-neighbourhood of a pixel, assuming the pixel would change its label.
     * @param labelImage the current label image, contains one label identifier per pixel
     * @param curPixelCoords coordinates of the regarded pixel
     * @param pretendLabel assumed new label of the regarded pixel
     * @param neighbourLabels all labels in the neighbourhood of the regarded pixel, including the label of the pixel itself
     * @return the total cost, summed over all labels in the neighbourhood and all enabled features, plus the Markov clique costs
     */
    double calculateCost(cv::Mat const& labelImage, cv::Point2i const& curPixelCoords,
                         label_t const& pretendLabel, std::vector<label_t> const& neighbourLabels) const;

    /**
     * @brief Calculate the Markov clique cost of a pixel, assuming it would change its label.
     * @param labelImage the current label image, contains one label identifier per pixel
     * @param curPixelCoords coordinates of the regarded pixel
     * @param pretendLabel assumed new label of the regarded pixel
     * @return the total Markov clique cost for the given label at the given pixel coordinates
     */
    double calculateCliqueCost(cv::Mat const& labelImage, cv::Point2i const& curPixelCoords, label_t const& pretendLabel) const;

    /**
     * @brief Create a binary map highlighting pixels on the boundary of their respective labels (1 for boundary pixels, 0 otherwise).
     * @param labelImage the current label image, contains one label identifier per pixel
     * @param out_boundaryMap the resulting boundary map, will be (re)allocated if necessary, binary by nature but stored as unsigned char
     */
    void computeBoundaryMap(cv::Mat const& labelImage, cv::Mat& out_boundaryMap) const;

    void computeBoundaryMapSmall(cv::Mat const& labelImage, cv::Mat& out_boundaryMap) const;

    /**
     * @brief Update a boundary map to reflect a label change of a single pixel.
     * @param labelImage the current label image (after the label change), contains one label identifier per pixel
     * @param curPixelCoords the coordinates of the changed pixel
     * @param boundaryMap the boundary map before the label change, will be updated if necessary to be consistent with the change
     */
    void updateBoundaryMap(cv::Mat const& labelImage, cv::Point2i const& curPixelCoords, cv::Mat& boundaryMap) const;

    cv::Mat labelImage;
    cv::Mat boundaryMap;
    const double directCliqueCost;
    const double diagonalCliqueCost;
    label_t maxLabelId;

   public:
    /**
     * @brief Constructor. Create a ContourRelaxation object with the specified features enabled.
     * @param initialLabelImage the initial label image, containing one label identifier per pixel
     * @param directCliqueCost the cost of a direct clique
     * @param diagonalCliqueCost the cost of a diagonal clique
     */
    ContourRelaxation(const cv::Mat initialLabelImage, const double directCliqueCost,
                      const double diagonalCliqueCost);

    template <typename T, typename... Args>
    void addFeature(const double weight, Args... args) {
        this->addFeature(boost::make_shared<T>(args...), weight);
    }

    void addFeature(boost::shared_ptr<IFeature> feature, const double weight);

    /**
     * @brief Apply Contour Relaxation to the given label image, with the features enabled in this ContourRelaxation object.
     * @param labelImage the input label image, containing one label identifier per pixel
     * @param numIterations number of iterations of Contour Relaxation to be performed (one iteration can include multiple passes)
     * @param out_labelImage the resulting label image after Contour Relaxation, will be (re)allocated if necessary
     *
     * One iteration of Contour Relaxation may pass over the image multiple times, in changing directions, in order to
     * mitigate the dependency of the result on the chosen order in which pixels are processed. This dependency comes from
     * the greedy nature of the performed optimization.
     */
    void relax(unsigned int const numIterations, cv::OutputArray out_labelImage);

    void setData(const DataType type, const cv::Mat& image);
};
}  // namespace cart::contour