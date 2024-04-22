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

/**
 * @file InitializationFunctions.h
 * @brief Header file providing some initialization functions for label images. These can be used for generating a starting point for superpixel generation with Contour Relaxation.
 */

#include <algorithm>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <vector>

#include "constants.hpp"

namespace cart::contour {

/**
 * @brief Create a label image initialization of rectangular blocks of the given size.
 * @param imageSize the size of the label image which will be created
 * @param blockWidth the width of one rectangular block
 * @param blockHeight the height of one rectangular block
 * @return a label image of the given size, constructed by blocks of the given size, with each block being a single, unique label
 */
cv::Mat createBlockInitialization(cv::Size const& imageSize, int const& blockWidth, int const& blockHeight);

/**
 * @brief Create a label image initialization of diamonds / rotated rectangular blocks of the given size.
 * @param imageSize the size of the label image which will be created
 * @param sideLength the width and height of one rectangular block which will be rotated by 45 degrees to form a diamond
 * @return a label image of the given size, constructed by diamonds of the given size, with each diamond being a single, unique label
 */
cv::Mat createDiamondInitialization(cv::Size const& imageSize, int const& sideLength);
}  // namespace cart::contour