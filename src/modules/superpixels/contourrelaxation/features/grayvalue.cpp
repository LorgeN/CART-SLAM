#include "modules/superpixels/contourrelaxation/features/grayvalue.hpp"

namespace cart::contour {
void GrayvalueFeature::setData(cv::Mat const& grayvalueImage) {
    assert(grayvalueImage.type() == cv::DataType<TGrayvalueData>::type);

    grayvalueImage.copyTo(grayvalImage);
}

void GrayvalueFeature::initializeStatistics(cv::Mat const& labelImage) {
    // Use the provided initialization method for gaussian statistics from AGaussianFeature.
    this->template initializeGaussianStatistics<TGrayvalueData>(labelImage, grayvalImage, labelStatistics);
}

double GrayvalueFeature::calculateCost(cv::Point2i const& curPixelCoords,
                                       label_t const& oldLabel, label_t const& pretendLabel, std::vector<label_t> const& neighbourLabels) const {
    // Use the provided cost calculation method for gaussian statistics from AGaussianFeature.
    return this->template calculateGaussianCost<TGrayvalueData>(curPixelCoords, oldLabel, pretendLabel,
                                                                neighbourLabels, labelStatistics, grayvalImage);
}

void GrayvalueFeature::updateStatistics(cv::Point2i const& curPixelCoords, label_t const& oldLabel,
                                        label_t const& newLabel) {
    // Use the provided update method for gaussian statistics from AGaussianFeature.
    this->template updateGaussianStatistics<TGrayvalueData>(curPixelCoords, labelStatistics[oldLabel],
                                                            labelStatistics[newLabel], grayvalImage);
}

void GrayvalueFeature::generateRegionMeanImage(cv::Mat const& labelImage, cv::Mat& out_regionMeanImage) const {
    assert(labelImage.size() == grayvalImage.size());
    assert(labelImage.type() == cv::DataType<label_t>::type);

    // Generate an image which represents all pixels by the mean grayvalue of their label.
    // cv::Mat::create only reallocates memory if necessary, so this is no slowdown.
    // We don't care about initialization of out_regionMeanImage since we will set all pixels anyway.
    out_regionMeanImage.create(grayvalImage.size(), cv::DataType<TGrayvalueData>::type);

    for (int row = 0; row < out_regionMeanImage.rows; ++row) {
        TGrayvalueData* const out_rmiRowPtr = out_regionMeanImage.ptr<TGrayvalueData>(row);
        label_t const* const labelImRowPtr = labelImage.ptr<label_t>(row);

        for (int col = 0; col < out_regionMeanImage.cols; ++col) {
            out_rmiRowPtr[col] = labelStatistics[labelImRowPtr[col]].valueSum / labelStatistics[labelImRowPtr[col]].pixelCount;
        }
    }
}
}  // namespace cart::contour