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
typedef uint8_t TGrayvalueData;  ///< the type of the used grayvalue images

/**
 * @class GrayvalueFeature
 * @brief Feature of one Gaussian-distributed channel, intended for use with grayvalue images.
 */
class GrayvalueFeature : public AGaussianFeature<TGrayvalueData, 1> {};

}  // namespace cart::contour
