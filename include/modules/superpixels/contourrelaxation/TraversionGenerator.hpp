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

#include <opencv2/opencv.hpp>

/**
 * @brief Class for looping over images in specified traversion orders.
 *
 * This class provides the functionality to get image coordinates one by one in specified traversion orders.
 * Use this to loop over an image multiple times using such orders and you only need
 * to fetch the current coordinates in each loop iteration with one simple function call.
 */
class TraversionGenerator {
   private:
    int imWidth;   ///< the current image width, must be set with TraversionGenerator::begin
    int imHeight;  ///< the current image height, must be set with TraversionGenerator::begin
    int curIndex;  ///< the current index (current position of the traversion), will be reset by each call of TraversionGenerator::begin

    /**
     * @brief The different traversion orders (plus the special state Finished) are defined here.
     *
     * The order in which these traversion orders will be used is defined in the functions TraversionGenerator::begin (initialization)
     * and TraversionGenerator::nextPixel (transitions), not here!
     */
    enum TraversionOrder {
        LeftRight,
        RightLeft,
        TopDown,
        BottomUp,
        Finished
    };

    TraversionOrder curOrder;  ///< the current traversion order being used

   public:
    cv::Point2i begin(cv::Size imageSize);

    cv::Point2i nextPixel();

    cv::Point2i end() const;
};
