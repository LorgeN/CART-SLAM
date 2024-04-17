#include "utils/peaks.hpp"

namespace cart::util {
int Peak::getPersistence(cv::Mat& mat) {
    if (this->died == -1) {
        return INT_MAX;
    }

    return mat.at<int>(this->born) - mat.at<int>(this->died);
}

std::vector<Peak> findPeaks(cv::Mat data) {
    if (data.rows != 1) {
        throw std::invalid_argument("Data must be a row vector");
    }

    std::vector<Peak> peaks;

    size_t n = data.cols;

    int idxtopeak[n];
    memset(idxtopeak, -1, sizeof(idxtopeak));

    int indices[n];
    for (int i = 0; i < n; i++) {
        indices[i] = i;
    }

    // Sort indices by descending data value
    std::sort(indices, indices + n, [&data](int a, int b) { return data.at<int>(a) > data.at<int>(b); });

    for (auto idx : indices) {
        bool lftdone = idx > 0 && idxtopeak[idx - 1] != -1;
        bool rgtdone = idx < n - 1 && idxtopeak[idx + 1] != -1;

        int il = lftdone ? idxtopeak[idx - 1] : -1;
        int ir = rgtdone ? idxtopeak[idx + 1] : -1;

        // New peak born
        if (!lftdone && !rgtdone) {
            peaks.push_back(Peak(idx));
            idxtopeak[idx] = peaks.size() - 1;

            // Directly merge to next peak left
        } else if (lftdone && !rgtdone) {
            peaks[il].right += 1;
            idxtopeak[idx] = il;

            // Directly merge to next peak right
        } else if (!lftdone && rgtdone) {
            peaks[ir].left -= 1;
            idxtopeak[idx] = ir;

            // Merge left and right peaks
        } else {
            // Left was born earlier: merge right to left
            if (data.at<int>(peaks[il].born) > data.at<int>(peaks[ir].born)) {
                peaks[ir].died = idx;
                peaks[il].right = peaks[ir].right;
                idxtopeak[peaks[il].right] = idxtopeak[idx] = il;
            } else {
                peaks[il].died = idx;
                peaks[ir].left = peaks[il].left;
                idxtopeak[peaks[ir].left] = idxtopeak[idx] = ir;
            }
        }
    }

    // Sort peaks by persistence
    std::sort(peaks.begin(), peaks.end(), [&data](Peak a, Peak b) { return a.getPersistence(data) > b.getPersistence(data); });
    return peaks;
}
}  // namespace cart::util