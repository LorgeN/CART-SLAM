#pragma once

#include <opencv2/opencv.hpp>
#include <vector>

#include "../constants.hpp"
#include "gaussian.cuh"
#include "feature.cuh"

namespace cart::contour {
typedef uint8_t TGrayvalueData;  ///< the type of the used grayvalue images

/**
 * @class GrayvalueFeature
 * @brief Feature of one Gaussian-distributed channel, intended for use with grayvalue images.
 */
class GrayvalueFeature : public AGaussianFeature<TGrayvalueData, 1> {};

}  // namespace cart::contour
