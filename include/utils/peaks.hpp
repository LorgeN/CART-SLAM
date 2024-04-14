#pragma once

#include <opencv2/opencv.hpp>
#include <vector>

// Implementation based on https://www.sthu.org/blog/13-perstopology-peakdetection/index.html

namespace cart::util {
class Peak {
   public:
    Peak(int index) : born(index), left(index), right(index), died(-1){};

    int getPersistence(cv::Mat& mat);

    int born;
    int left;
    int right;
    int died;
};

std::vector<Peak> findPeaks(cv::Mat data);
}  // namespace cart::util