#pragma once

#include <cmath>
#include <cstdint>

#include "utils/colors.hpp"

namespace cart {
struct DeviceColor {
    uint8_t color[3];
};

__constant__ DeviceColor COLOR_WHEEL_DEVICE[NCOLS];

void copyColorWheelToDevice(cudaStream_t &stream) {
    const util::ColorWheel colorWheel;
    cudaMemcpyToSymbolAsync(COLOR_WHEEL_DEVICE, colorWheel.colorWheel, NCOLS * sizeof(DeviceColor), 0, cudaMemcpyHostToDevice, stream);
}

__device__ void assignColor(float fx, float fy, uint8_t *pix) {
    const float rad = sqrt(fx * fx + fy * fy);
    const float a = atan2(-fy, -fx) / (float)M_PI;

    const float fk = (a + 1.0f) / 2.0f * (NCOLS - 1);
    const int k0 = static_cast<int>(fk);
    const int k1 = (k0 + 1) % NCOLS;
    const float f = fk - k0;

    for (int b = 0; b < 3; b++) {
        const float col0 = COLOR_WHEEL_DEVICE[k0].color[b] / 255.0f;
        const float col1 = COLOR_WHEEL_DEVICE[k1].color[b] / 255.0f;

        float col = (1 - f) * col0 + f * col1;

        if (rad <= 1)
            col = 1 - rad * (1 - col);  // increase saturation with radius
        else
            col *= .75;  // out of range

        pix[2 - b] = static_cast<uint8_t>(255.0 * col);
    }
}
}  // namespace cart