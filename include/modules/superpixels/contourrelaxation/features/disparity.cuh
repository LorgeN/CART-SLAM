#pragma once

#include "../constants.hpp"
#include "feature.cuh"
#include "gaussian.cuh"

namespace cart::contour {

typedef int16_t TDisparityData;  ///< the type of the used disparity channels

/**
 * @class DisparityFeature
 * @brief Feature of two independently Gaussian-distributed channels, intended for use with vertical/horizontal disparity derivative images.
 */
class DisparityFeature : public AGaussianFeature<TDisparityData, 2, DataType::Disparity> {
   public:
    DisparityFeature() : AGaussianFeature("Disparity") {}
};
}  // namespace cart::contour