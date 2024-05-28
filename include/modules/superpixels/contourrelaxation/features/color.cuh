#pragma once

#include "../constants.hpp"
#include "feature.cuh"
#include "gaussian.cuh"

namespace cart::contour {

typedef uint8_t TColorData;  ///< the type of the used image channels

/**
 * @class ColorFeature
 * @brief Feature of three independently Gaussian-distributed channels, intended for use with color images in YUV space.
 */
class ColorFeature : public AGaussianFeature<TColorData, 3, DataType::Image> {
   public:
    ColorFeature() : AGaussianFeature("Color") {}
};
}  // namespace cart::contour