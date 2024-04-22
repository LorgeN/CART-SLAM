#include "modules/superpixels/contourrelaxation/initialization.hpp"

namespace cart::contour {

cv::Mat createBlockInitialization(cv::Size const& imageSize, int const& blockWidth, int const& blockHeight) {
    assert(imageSize.width > 0 && imageSize.height > 0);
    assert(blockWidth > 0 && blockHeight > 0);
    assert(imageSize.width >= blockWidth && imageSize.height >= blockHeight);

    cv::Mat out_labelImage(imageSize, cv::DataType<label_t>::type);

    // Find out how many blocks there will be in each direction. If image size is not a multiple of block size,
    // we need to round upwards because there will be one additional (smaller) block.
    int const numBlocksX = ceil(static_cast<double>(imageSize.width) / blockWidth);
    int const numBlocksY = ceil(static_cast<double>(imageSize.height) / blockHeight);

    // Counter for the label of the current block.
    label_t curLabel = 0;

    for (int blockYi = 0; blockYi < numBlocksY; ++blockYi) {
        // Get the y-limits of all blocks in the current row of blocks.
        // End is tricky because of a possibly smaller last block.
        int const blockYStart = blockYi * blockHeight;
        int const blockYEnd = std::min((blockYi + 1) * blockHeight, imageSize.height);

        for (int blockXi = 0; blockXi < numBlocksX; ++blockXi) {
            // Get the x-limits of the current block.
            // End is tricky because of a possibly smaller last block.
            int const blockXStart = blockXi * blockWidth;
            int const blockXEnd = std::min((blockXi + 1) * blockWidth, imageSize.width);

            // Get a rectangle of the current blocksize and at the current block position.
            cv::Rect const curBlock(blockXStart, blockYStart, blockXEnd - blockXStart, blockYEnd - blockYStart);
            // Get a matrix header to the current block in the label image.
            cv::Mat curBlockInLabelImage = out_labelImage(curBlock);

            // Assign current label to block, increase label counter.
            curBlockInLabelImage = curLabel;

            ++curLabel;
        }
    }

    return out_labelImage;
}

cv::Mat createDiamondInitialization(cv::Size const& imageSize, int const& sideLength) {
    assert(imageSize.width > 0 && imageSize.height > 0);
    assert(sideLength > 0);
    assert(imageSize.width >= sideLength && imageSize.height >= sideLength);

    // For creating the bigger labels, we use the bigger dimension for both so we won't get into trouble after rotation.
    int const maxDimSize = std::max(imageSize.width, imageSize.height);

    // Create a block initialization of twice the size, which ensures we can pick our labels from the center
    // after rotation without getting border effects.
    cv::Mat const bigBlockLabels = createBlockInitialization(cv::Size(2 * maxDimSize, 2 * maxDimSize),
                                                             sideLength, sideLength);

    // Create a rotation matrix which rotates around the center of bigBlockLabels, for 45 degrees ccw, with scale 1.
    cv::Mat const rotationMatrix = cv::getRotationMatrix2D(cv::Point2f(maxDimSize, maxDimSize), 45, 1);
    cv::Mat rotatedLabels;

    // Rotate the big label image. Destination image shall have the same size and interpolation shall be NN
    // so that no new labels are created.
    // Type of rotatedLabels will be the same as of bigBlockLabels.
    cv::warpAffine(bigBlockLabels, rotatedLabels, rotationMatrix, bigBlockLabels.size(), cv::INTER_NEAREST);

    // Pick the upper left corner of the region we want to use as result.
    // This ensures we pick our result from the center of the rotated labels.
    int const xStart = maxDimSize - (imageSize.width / 2);
    int const yStart = maxDimSize - (imageSize.height / 2);

    // Copy the labels into a new matrix and return it.
    cv::Rect const roi(xStart, yStart, imageSize.width, imageSize.height);
    cv::Mat roiLabels = rotatedLabels(roi);

    cv::Mat out_labelImage;
    roiLabels.copyTo(out_labelImage);

    return out_labelImage;
}
}  // namespace cart::contour