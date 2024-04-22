#include "modules/superpixels/contourrelaxation/features/compactness.hpp"

#include "modules/superpixels/contourrelaxation/constants.hpp"

namespace cart::contour {

void CompactnessFeature::updateStatistics(cv::Point2i const& curPixelCoords,
                                          LabelStatisticsGauss& labelStatsOldLabelPosX, LabelStatisticsGauss& labelStatsNewLabelPosX,
                                          LabelStatisticsGauss& labelStatsOldLabelPosY, LabelStatisticsGauss& labelStatsNewLabelPosY) const {
    // Update pixel count.
    labelStatsOldLabelPosX.pixelCount--;
    labelStatsNewLabelPosX.pixelCount++;

    // Update x-position sum.
    labelStatsOldLabelPosX.valueSum -= curPixelCoords.x;
    labelStatsNewLabelPosX.valueSum += curPixelCoords.x;

    // Update square x-position sum.
    labelStatsOldLabelPosX.squareValueSum -= pow(curPixelCoords.x, 2.0);
    labelStatsNewLabelPosX.squareValueSum += pow(curPixelCoords.x, 2.0);

    // Update pixel count.
    labelStatsOldLabelPosY.pixelCount--;
    labelStatsNewLabelPosY.pixelCount++;

    // Update x-position sum.
    labelStatsOldLabelPosY.valueSum -= curPixelCoords.y;
    labelStatsNewLabelPosY.valueSum += curPixelCoords.y;

    // Update square x-position sum.
    labelStatsOldLabelPosY.squareValueSum -= pow(curPixelCoords.y, 2.0);
    labelStatsNewLabelPosY.squareValueSum += pow(curPixelCoords.y, 2.0);
}

void CompactnessFeature::setData(double const& compactnessWeight) {
    assert(compactnessWeight >= 0);

    featureWeight = compactnessWeight;
}

void CompactnessFeature::initializeStatistics(cv::Mat const& labelImage) {
    assert(labelImage.type() == cv::DataType<label_t>::type);

    // Find maximum label identifier in label image.
    label_t maxLabelId = 0;

    for (int row = 0; row < labelImage.rows; ++row) {
        label_t const* const labelImageRowPtr = labelImage.ptr<label_t>(row);

        for (int col = 0; col < labelImage.cols; ++col) {
            maxLabelId = std::max(maxLabelId, labelImageRowPtr[col]);
        }
    }

    // Allocate the vectors of label statistics, with the maximum index being the maximum label identifier.
    // This might waste a small amount of memory, but we can use the label identifier as index for this vector.
    labelStatisticsPosX = std::vector<LabelStatisticsGauss>(maxLabelId + 1, LabelStatisticsGauss());
    labelStatisticsPosY = std::vector<LabelStatisticsGauss>(maxLabelId + 1, LabelStatisticsGauss());

    for (int row = 0; row < labelImage.rows; ++row) {
        label_t const* const labelImageRowPtr = labelImage.ptr<label_t>(row);

        for (int col = 0; col < labelImage.cols; ++col) {
            label_t const curLabel = labelImageRowPtr[col];

            labelStatisticsPosX[curLabel].pixelCount++;
            labelStatisticsPosX[curLabel].valueSum += col;
            labelStatisticsPosX[curLabel].squareValueSum += pow(col, 2.0);

            labelStatisticsPosY[curLabel].pixelCount++;
            labelStatisticsPosY[curLabel].valueSum += row;
            labelStatisticsPosY[curLabel].squareValueSum += pow(row, 2.0);
        }
    }
}

double CompactnessFeature::calculateCost(cv::Point2i const& curPixelCoords,
                                         label_t const& oldLabel, label_t const& pretendLabel, std::vector<label_t> const& neighbourLabels) const {
    // Modify the label statistics if the pixel at curPixelCoords has another
    // label than pretendLabel. We only modify a local copy of the statistics,
    // however, since we do not actually want to change a label yet. We are
    // just calculating the cost of doing so.
    // We only copy the affected statistics, namely the ones of the old and the
    // new (pretend) label. We have to handle these two cases separately when accessing
    // the statistics, because we specifically need to access the local, modified versions.
    // But we save a lot of time since we do not need to create a copy of all statistics.
    LabelStatisticsGauss labelStatsPosXOldLabel(labelStatisticsPosX[oldLabel]);
    LabelStatisticsGauss labelStatsPosXPretendLabel(labelStatisticsPosX[pretendLabel]);

    LabelStatisticsGauss labelStatsPosYOldLabel(labelStatisticsPosY[oldLabel]);
    LabelStatisticsGauss labelStatsPosYPretendLabel(labelStatisticsPosY[pretendLabel]);

    if (oldLabel != pretendLabel) {
        updateStatistics(curPixelCoords, labelStatsPosXOldLabel, labelStatsPosXPretendLabel,
                         labelStatsPosYOldLabel, labelStatsPosYPretendLabel);
    }

    double featureCost = 0;

    // For each neighbouring label, add its cost.
    for (typename std::vector<label_t>::const_iterator it_neighbourLabel = neighbourLabels.begin();
         it_neighbourLabel != neighbourLabels.end(); ++it_neighbourLabel) {
        // Get a pointer to the label statistics of the current label.
        // This should be the associated entry in the vector of label statistics, except for the
        // special cases below.
        LabelStatisticsGauss const* curLabelStatsPosX = &labelStatisticsPosX[*it_neighbourLabel];
        LabelStatisticsGauss const* curLabelStatsPosY = &labelStatisticsPosY[*it_neighbourLabel];

        // If the current label is the old one at curPixelCoords, or the pretended new one,
        // then its statistics have changed. In this case, we need to read the statistics
        // from the modified local variables.
        if (*it_neighbourLabel == oldLabel) {
            curLabelStatsPosX = &labelStatsPosXOldLabel;
            curLabelStatsPosY = &labelStatsPosYOldLabel;
        } else if (*it_neighbourLabel == pretendLabel) {
            curLabelStatsPosX = &labelStatsPosXPretendLabel;
            curLabelStatsPosY = &labelStatsPosYPretendLabel;
        }

        // If a label completely vanished, disregard it (can happen to old label of pixel_index).
        if (curLabelStatsPosX->pixelCount == 0) {
            continue;
        }

        // Add the cost of the current region.
        featureCost += curLabelStatsPosX->squareValueSum - (pow(curLabelStatsPosX->valueSum, 2) / curLabelStatsPosX->pixelCount);

        featureCost += curLabelStatsPosY->squareValueSum - (pow(curLabelStatsPosY->valueSum, 2) / curLabelStatsPosY->pixelCount);
    }

    return featureWeight * featureCost;
}

void CompactnessFeature::updateStatistics(cv::Point2i const& curPixelCoords, label_t const& oldLabel,
                                          label_t const& newLabel) {
    updateStatistics(curPixelCoords, labelStatisticsPosX[oldLabel], labelStatisticsPosX[newLabel],
                     labelStatisticsPosY[oldLabel], labelStatisticsPosY[newLabel]);
}
}  // namespace cart::contour