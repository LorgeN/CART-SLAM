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
 * @file visualization.cuh
 * @brief Header file providing some functions for visualization purposes.
 */

#include <opencv2/opencv.hpp>
#include <vector>

#include "contourrelaxation/constants.hpp"

namespace cart::contour {

/**
 * @brief Create a boundary overlay, that is: mark the label boundaries in the given color image by making them red.
 * @param bgrImage the color image to serve as basis for the boundary overlay image, in BGR format
 * @param labelImage the label image, contains one label identifier per pixel
 * @param out_boundaryOverlay the resulting boundary overlay image, will be (re)allocated if necessary
 *
 * This function uses a rather crude way to find boundary pixels: it only compares with the pixel to the right
 * and the one below. This works most of the time, however a few boundary pixels may not be found, especially
 * in the last row and column. Boundary overlays are only useful for visualization, so this is not much of an
 * issue. A perfectly accurate version of this function could be implemented by using the functionality from
 * ::computeBoundaryImage to identify boundary pixels, and just marking those in red.
 */
void computeBoundaryOverlay(cv::cuda::GpuMat bgrImage, cv::cuda::GpuMat labelImage, cv::cuda::GpuMat &out_boundaryOverlay);

/**
 * @brief Compute a binary boundary image, containing 1 for pixels on a label boundary, 0 otherwise.
 * @param labelImage the label image, contains one label identifier per pixel
 * @param out_boundaryImage the resulting boundary image, binary by nature, but stored as unsigned chars, will be (re)allocated if necessary
 *
 * This function is basically taken from ContourRelaxation::computeBoundaryMap and provides exact boundary maps
 * (with 2 pixel-wide boundaries) which can be used e.g. for benchmarks.
 */
void computeBoundaryImage(cv::cuda::GpuMat labelImage, cv::cuda::GpuMat &out_boundaryImage);

}  // namespace cart::contour