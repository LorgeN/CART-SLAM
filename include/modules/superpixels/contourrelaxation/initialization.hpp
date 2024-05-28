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

void createBlockInitialization(cv::Size const& imageSize, int const& blockWidth, int const& blockHeight, cv::cuda::GpuMat& labelImage, cart::contour::label_t& maxLabelId, cv::cuda::Stream& stream = cv::cuda::Stream::Null());

}  // namespace cart::contour