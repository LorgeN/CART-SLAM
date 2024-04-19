#include "modules/superpixels/contourrelaxation/TraversionGenerator.hpp"

/**
 * @brief Set the image size and reset all counters to start values, get the first coordinates.
 * @param imageSize the size of the image to be traversed
 * @return the first coordinates
 *
 * This function must be called each time a new set of traversions (not a switch to the next traversion order) of an image shall start.
 */
cv::Point2i TraversionGenerator::begin(cv::Size imageSize) {
    assert(imageSize.width > 0 && imageSize.height > 0);

    // Save the image size for this traversion.
    imWidth = imageSize.width;
    imHeight = imageSize.height;

    // Set the first order and the start index as current states.
    // Do not just change the initialization of curOrder here!
    // You also need to make sure that from this initialization, through the transitions
    // specified in nextPixel(), all orders are visited and then we end with Finished.
    curOrder = LeftRight;
    curIndex = 0;

    // Return the first coordinate.
    return cv::Point2i(0, 0);
}

/**
 * @brief Increase internal counters and get the next coordinates.
 * @return the next coordinates
 */
cv::Point2i TraversionGenerator::nextPixel() {
    // Initialize the return variable with the value for "we are done".
    cv::Point2i curPoint = end();

    // If we are still inside a valid traversion, go ahead.
    if (curOrder != Finished) {
        // Increment the index counter.
        ++curIndex;

        // If the index counter goes beyond the count of image pixels (or rather, reaches it, since it is zero-based),
        // reset to 0 and go to the next traversion order.
        // This is the place to define the order of traversion orders.
        // Just make sure that, starting with the initialization from begin(), all orders are used and we end with Finished.
        if (curIndex >= imWidth * imHeight) {
            curIndex = 0;

            switch (curOrder) {
                case LeftRight:
                    curOrder = RightLeft;
                    break;

                case RightLeft:
                    curOrder = TopDown;
                    break;

                case TopDown:
                    curOrder = BottomUp;
                    break;

                case BottomUp:
                    curOrder = Finished;
                    break;

                // Just to remove the compiler warning for non-handled cases.
                // It really is handled, see if above.
                case Finished:
                    break;
            }
        }

        // Since the traversion order may have changed, we need to check again if there are still points left.
        if (curOrder != Finished) {
            // Only now will we evaluate the current coordinates and write them to the return variable.
            // This can be done with a modulo and an integer division in each case.
            // For the "reverse cases" (RightLeft, BottomUp), we simply simulate the counter going backwards.
            switch (curOrder) {
                case LeftRight:
                    curPoint.x = curIndex % imWidth;
                    curPoint.y = curIndex / imWidth;
                    break;

                case RightLeft:
                    curPoint.x = (imWidth * imHeight - 1 - curIndex) % imWidth;
                    curPoint.y = (imWidth * imHeight - 1 - curIndex) / imWidth;
                    break;

                case TopDown:
                    curPoint.x = curIndex / imHeight;
                    curPoint.y = curIndex % imHeight;
                    break;

                case BottomUp:
                    curPoint.x = (imWidth * imHeight - 1 - curIndex) / imHeight;
                    curPoint.y = (imWidth * imHeight - 1 - curIndex) % imHeight;
                    break;

                // Just to remove the compiler warning for non-handled cases.
                // It really is handled, see ifs above.
                case Finished:
                    break;
            }
        }
    }

    // Return the current coordinates.
    return curPoint;
}

/**
 * @brief Get the special coordinates which will be used to signal the end of traversions (for comparison).
 * @return the special coordinates signaling the end of traversions
 */
cv::Point2i TraversionGenerator::end() const {
    // Return a point which can never occur as valid image coordinates
    // and which will be used to signal the end of the traversion to the calling function.
    return cv::Point2i(-1, -1);
}