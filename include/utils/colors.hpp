#pragma once

#include "opencv2/opencv.hpp"

#define RY (15)
#define YG (6)
#define GC (4)
#define CB (11)
#define BM (13)
#define MR (6)
#define NCOLS (RY + YG + GC + CB + BM + MR)

namespace cart::util {

void fillColorWheel(cv::Vec3i* colorWheel);

class ColorWheel {
   public:
    ColorWheel() {
        fillColorWheel(this->colorWheel);
    }

    cv::Vec3i colorWheel[NCOLS];

    const cv::Vec3i& operator[](size_t index) const {
        return this->colorWheel[index];
    }
};

cv::Vec3b computeColor(float fx, float fy);

// CUDA-variant in utils/cuda.cuh

}  // namespace cart::util
