#pragma once

#include <opencv2/opencv.hpp>
#include <vector>

#include "../constants.hpp"
#include "gaussian.hpp"
#include "ifeature.hpp"

namespace cart::contour {

typedef int16_t TDisparityData;  ///< the type of the used disparity channels

/**
 * @class DisparityFeature
 * @brief Feature of two independently Gaussian-distributed channels, intended for use with vertical/horizontal disparity derivative images.
 */
class DisparityFeature : public AGaussianFeature<TDisparityData, 2> {};
}  // namespace cart::contour