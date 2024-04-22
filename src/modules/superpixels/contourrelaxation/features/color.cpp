#include "modules/superpixels/contourrelaxation/features/color.hpp"

namespace cart::contour {

void ColorFeature::setData(cv::Mat const& colorChannel1, cv::Mat const& colorChannel2, cv::Mat const& colorChannel3) {
    assert(colorChannel2.size() == colorChannel1.size());
    assert(colorChannel3.size() == colorChannel1.size());
    assert(colorChannel1.type() == cv::DataType<TColorData>::type);
    assert(colorChannel2.type() == cv::DataType<TColorData>::type);
    assert(colorChannel3.type() == cv::DataType<TColorData>::type);

    colorChannel1.copyTo(channel1);
    colorChannel2.copyTo(channel2);
    colorChannel3.copyTo(channel3);
}

void ColorFeature::initializeStatistics(cv::Mat const& labelImage) {
    // Use the provided initialization method for gaussian statistics from AGaussianFeature.
    this->initializeGaussianStatistics<TColorData>(labelImage, channel1, labelStatisticsChan1);
    this->initializeGaussianStatistics<TColorData>(labelImage, channel2, labelStatisticsChan2);
    this->initializeGaussianStatistics<TColorData>(labelImage, channel3, labelStatisticsChan3);
}

double ColorFeature::calculateCost(cv::Point2i const& curPixelCoords,
                                   label_t const& oldLabel, label_t const& pretendLabel, std::vector<label_t> const& neighbourLabels) const {
    // Use the provided cost calculation method for gaussian statistics from AGaussianFeature.
    double cost = this->calculateGaussianCost<TColorData>(curPixelCoords, oldLabel, pretendLabel,
                                                          neighbourLabels, labelStatisticsChan1, channel1) +
                  this->calculateGaussianCost<TColorData>(curPixelCoords, oldLabel, pretendLabel,
                                                          neighbourLabels, labelStatisticsChan2, channel2) +
                  this->calculateGaussianCost<TColorData>(curPixelCoords, oldLabel, pretendLabel,
                                                          neighbourLabels, labelStatisticsChan3, channel3);

    return cost;
}

void ColorFeature::updateStatistics(cv::Point2i const& curPixelCoords, label_t const& oldLabel,
                                    label_t const& newLabel) {
    // Use the provided update method for gaussian statistics from AGaussianFeature.
    this->updateGaussianStatistics<TColorData>(curPixelCoords, labelStatisticsChan1[oldLabel],
                                               labelStatisticsChan1[newLabel], channel1);
    this->updateGaussianStatistics<TColorData>(curPixelCoords, labelStatisticsChan2[oldLabel],
                                               labelStatisticsChan2[newLabel], channel2);
    this->updateGaussianStatistics<TColorData>(curPixelCoords, labelStatisticsChan3[oldLabel],
                                               labelStatisticsChan3[newLabel], channel3);
}

void ColorFeature::generateRegionMeanImage(cv::Mat const& labelImage, cv::Mat& out_regionMeanImage) const {
    assert(labelImage.size() == channel1.size());
    assert(labelImage.type() == cv::DataType<label_t>::type);

    // Generate an image which represents all pixels by the mean color value of their label.
    // Channels will be filled separately and later on merged, since accessing multi-channel
    // matrices is not quite straightforward.
    std::vector<cv::Mat> out_channels(3);
    out_channels[0].create(channel1.size(), cv::DataType<TColorData>::type);
    out_channels[1].create(channel1.size(), cv::DataType<TColorData>::type);
    out_channels[2].create(channel1.size(), cv::DataType<TColorData>::type);

    for (int row = 0; row < out_channels[0].rows; ++row) {
        TColorData* const out_chan1RowPtr = out_channels[0].ptr<TColorData>(row);
        TColorData* const out_chan2RowPtr = out_channels[1].ptr<TColorData>(row);
        TColorData* const out_chan3RowPtr = out_channels[2].ptr<TColorData>(row);
        label_t const* const labelImRowPtr = labelImage.ptr<label_t>(row);

        for (int col = 0; col < out_channels[0].cols; ++col) {
            out_chan1RowPtr[col] = labelStatisticsChan1[labelImRowPtr[col]].valueSum / labelStatisticsChan1[labelImRowPtr[col]].pixelCount;
            out_chan2RowPtr[col] = labelStatisticsChan2[labelImRowPtr[col]].valueSum / labelStatisticsChan2[labelImRowPtr[col]].pixelCount;
            out_chan3RowPtr[col] = labelStatisticsChan3[labelImRowPtr[col]].valueSum / labelStatisticsChan3[labelImRowPtr[col]].pixelCount;
        }
    }

    // Merge the channels into one 3-dim output image.
    // out_regionMeanImage will be (re)allocated automatically by cv::merge if necessary.
    cv::merge(out_channels, out_regionMeanImage);
}
}  // namespace cart::contour