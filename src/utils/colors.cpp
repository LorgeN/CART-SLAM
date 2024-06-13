#include "utils/colors.hpp"

namespace cart::util {
void fillColorWheel(cv::Vec3i* colorWheel) {
    int k = 0;

    for (int i = 0; i < RY; ++i, ++k)
        colorWheel[k] = cv::Vec3i(255, 255 * i / RY, 0);

    for (int i = 0; i < YG; ++i, ++k)
        colorWheel[k] = cv::Vec3i(255 - 255 * i / YG, 255, 0);

    for (int i = 0; i < GC; ++i, ++k)
        colorWheel[k] = cv::Vec3i(0, 255, 255 * i / GC);

    for (int i = 0; i < CB; ++i, ++k)
        colorWheel[k] = cv::Vec3i(0, 255 - 255 * i / CB, 255);

    for (int i = 0; i < BM; ++i, ++k)
        colorWheel[k] = cv::Vec3i(255 * i / BM, 0, 255);

    for (int i = 0; i < MR; ++i, ++k)
        colorWheel[k] = cv::Vec3i(255, 0, 255 - 255 * i / MR);
}

const ColorWheel COLOR_WHEEL;

cv::Vec3b computeColor(float color) {
    const float fx = color * (NCOLS - 1);
    const int ix = static_cast<int>(fx);

    const cv::Vec3f c0 = COLOR_WHEEL[ix];

    return cv::Vec3b(static_cast<uint8_t>(c0[0]), static_cast<uint8_t>(c0[1]), static_cast<uint8_t>(c0[1]));
}

// From https://github.com/opencv/opencv_contrib/blob/4.x/modules/cudaoptflow/samples/nvidia_optical_flow.cpp
cv::Vec3b computeColor(float fx, float fy) {
    const float rad = sqrt(fx * fx + fy * fy);
    const float a = atan2(-fy, -fx) / (float)CV_PI;

    const float fk = (a + 1.0f) / 2.0f * (NCOLS - 1);
    const int k0 = static_cast<int>(fk);
    const int k1 = (k0 + 1) % NCOLS;
    const float f = fk - k0;

    cv::Vec3b pix;

    for (int b = 0; b < 3; b++) {
        const float col0 = COLOR_WHEEL[k0][b] / 255.0f;
        const float col1 = COLOR_WHEEL[k1][b] / 255.0f;

        float col = (1 - f) * col0 + f * col1;

        if (rad <= 1)
            col = 1 - rad * (1 - col);  // increase saturation with radius
        else
            col *= .75;  // out of range

        pix[2 - b] = static_cast<uchar>(255.0 * col);
    }

    return pix;
}
}  // namespace cart::util