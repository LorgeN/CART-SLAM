#include <boost/make_shared.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>

#include "modules/superpixels/contourrelaxation/contourrelaxation.hpp"
#include "timing.hpp"
#include "utils/cuda.cuh"

#define THREADS_PER_BLOCK_X 32
#define THREADS_PER_BLOCK_Y 32
#define X_BATCH 4
#define Y_BATCH 4

#define SHARED_SIZE (X_BATCH * (2 + THREADS_PER_BLOCK_X)) * (Y_BATCH * (2 + THREADS_PER_BLOCK_Y))
#define LOCAL_INDEX(x, y) SHARED_INDEX(sharedPixelX + x, sharedPixelY + y, 1, 1, sharedRowStep)

__global__ void computeBoundaries(cv::cuda::PtrStepSz<cart::contour::label_t> labels, cv::cuda::PtrStepSz<uint8_t> out) {
    __shared__ cart::contour::label_t sharedLabels[SHARED_SIZE];

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int sharedPixelX = threadIdx.x * X_BATCH;
    int sharedPixelY = threadIdx.y * Y_BATCH;

    int pixelX = x * X_BATCH;
    int pixelY = y * Y_BATCH;

    size_t sharedRowStep = X_BATCH * blockDim.x;
    size_t outStep = out.step / sizeof(uint8_t);

    cart::copyToShared<cart::contour::label_t, X_BATCH, Y_BATCH, false>(sharedLabels, labels, 1, 1);

    __syncthreads();

    for (int i = 0; i < Y_BATCH; i++) {
        for (int j = 0; j < X_BATCH; j++) {
            if (pixelX + j >= labels.cols || pixelY + i >= labels.rows) {
                continue;
            }

            cart::contour::label_t label = sharedLabels[LOCAL_INDEX(j, i)];

            uint8_t border = false;

            for (int k = -1; k <= 1; k++) {
                for (int l = -1; l <= 1; l++) {
                    if (k == 0 && l == 0) {
                        continue;
                    }

                    cart::contour::label_t neighbor = sharedLabels[LOCAL_INDEX(j + k, i + l)];

                    if (label != neighbor) {
                        border = true;
                        break;
                    }
                }
            }

            out[INDEX(pixelX + j, pixelY + i, outStep)] = border;
        }
    }
}

namespace cart::contour {
ContourRelaxation::ContourRelaxation(const cv::Mat initialLabelImage, const double directCliqueCost,
                                     const double diagonalCliqueCost) : directCliqueCost(directCliqueCost), diagonalCliqueCost(diagonalCliqueCost) {
    this->labelImage = initialLabelImage;
}

void ContourRelaxation::addFeature(boost::shared_ptr<IFeature> feature, const double weight) {
    FeatureContainer container = {
        .feature = feature,
        .weight = weight,
    };

    this->features.push_back(container);
}

void ContourRelaxation::relax(const unsigned int numIterations, cv::OutputArray out_labelImage) {
    assert(labelImage.type() == cv::DataType<label_t>::type);

    double maxLabelDbl = 0;
    cv::minMaxIdx(this->labelImage, nullptr, &maxLabelDbl, nullptr, nullptr, cv::noArray());
    this->maxLabelId = static_cast<label_t>(maxLabelDbl);

// Compute the initial statistics of all labels given in the label image, for all features.
#pragma omp parallel for
    for (FeatureIterator it_curFeature = this->features.begin(); it_curFeature != this->features.end(); ++it_curFeature) {
        (*it_curFeature).feature->initializeStatistics(this->labelImage, this->maxLabelId);
    }

    // Create the initial boundary map.
    computeBoundaryMap(this->labelImage, this->boundaryMap);

    // Create a traversion generator object, which will give us all the pixel coordinates in the current image
    // in all traversion orders specified inside that class. We will just need to loop over the coordinates
    // we receive by this object.

    CARTSLAM_START_AVERAGE_TIMING(iteration);

    std::vector<cv::Point2i> pixels;
    getPointsAsVector(labelImage.size(), pixels);

    // Loop over specified number of iterations.
    for (unsigned int curIteration = 0; curIteration < numIterations; ++curIteration) {
        // Loop over all coordinates received by the traversion generator.
        // It is important to start with begin() here, which does not only set the correct image size,
        // but also resets all internal counters.
        CARTSLAM_START_TIMING(iteration);

        // TODO: Parallelize this loop further
#pragma omp parallel for
        for (auto curPixelCoords : pixels) {
            if (BOOST_LIKELY(boundaryMap.at<unsigned char>(curPixelCoords) == 0)) {
                // We are not at a boundary pixel, no further processing necessary.
                continue;
            }

            // Get all neighbouring labels. This vector also contains the label of the current pixel itself.
            std::vector<label_t> const neighbourLabels = getNeighbourLabels(this->labelImage, curPixelCoords);

            // If we have more than one label in the neighbourhood, the current pixel is a boundary pixel
            // and optimization will be carried out. Else, the neighbourhood only contains the label of the
            // pixel itself (since this label will definitely be there, and there is only one), so we don't
            // have a boundary pixel.
            if (BOOST_LIKELY(neighbourLabels.size() > 1)) {
                double minCost = std::numeric_limits<double>::max();
                double bestLabel = this->labelImage.at<label_t>(curPixelCoords);

                for (typename std::vector<label_t>::const_iterator it_neighbourLabel = neighbourLabels.begin();
                     it_neighbourLabel != neighbourLabels.end(); ++it_neighbourLabel) {
                    double cost = calculateCost(this->labelImage, curPixelCoords, *it_neighbourLabel, neighbourLabels);

                    if (cost < minCost) {
                        minCost = cost;
                        bestLabel = *it_neighbourLabel;
                    }
                }

                // If we have found a better label for the pixel, update the statistics for all features
                // and change the label of the pixel.
                if (bestLabel != this->labelImage.at<label_t>(curPixelCoords)) {
#pragma omp critical
                    {
                        for (FeatureIterator it_curFeature = this->features.begin(); it_curFeature != this->features.end(); ++it_curFeature) {
                            (*it_curFeature).feature->updateStatistics(curPixelCoords, this->labelImage.at<label_t>(curPixelCoords), bestLabel);
                        }

                        this->labelImage.at<label_t>(curPixelCoords) = bestLabel;

                        // We also need to update the boundary map around the current pixel.
                        updateBoundaryMap(this->labelImage, curPixelCoords, boundaryMap);
                    }
                }
            }
        }

        CARTSLAM_END_TIMING(iteration);
        CARTSLAM_INCREMENT_AVERAGE_TIMING(iteration);
    }

    CARTSLAM_END_AVERAGE_TIMING(iteration);

    // Return the resulting label image.
    if (out_labelImage.needed() && out_labelImage.isMat()) {
        out_labelImage.create(labelImage.size(), labelImage.type());
        labelImage.copyTo(out_labelImage.getMat());
    }
}

std::vector<label_t> ContourRelaxation::getNeighbourLabels(cv::Mat const& labelImage, cv::Point2i const& curPixelCoords)
    const {
    assert(labelImage.type() == cv::DataType<label_t>::type);
    assert(curPixelCoords.inside(cv::Rect(0, 0, labelImage.cols, labelImage.rows)));

    // To get all pixels in the 8-neighbourhood (or 9, since the rectangle includes the central pixel itself)
    // we form the intersection between the theoretical full neighbourhood and the bounding area.
    cv::Rect const fullNeighbourhoodRect(curPixelCoords.x - 1, curPixelCoords.y - 1, 3, 3);
    cv::Rect const boundaryRect(0, 0, labelImage.cols, labelImage.rows);
    cv::Rect const croppedNeighbourhoodRect = fullNeighbourhoodRect & boundaryRect;

    // Get a new matrix header to the relevant neighbourhood in the label image.
    cv::Mat const neighbourhoodLabelImage = labelImage(croppedNeighbourhoodRect);

    // Push all labels in the neighbourhood into a vector.
    // Reserve enough space for the maximum of 9 labels in the neighbourhood.
    // Making this one big allocation is extremely faster than making multiple small allocations when pushing elements.
    std::vector<label_t> neighbourLabels;
    neighbourLabels.reserve(9);

    for (int row = 0; row < neighbourhoodLabelImage.rows; ++row) {
        label_t const* const neighbLabelsRowPtr = neighbourhoodLabelImage.ptr<label_t>(row);

        for (int col = 0; col < neighbourhoodLabelImage.cols; ++col) {
            neighbourLabels.push_back(neighbLabelsRowPtr[col]);
        }
    }

    // Remove duplicates from the vector of neighbour labels.
    // First sort the vector, then remove consecutive duplicates, then resize.
    std::sort(neighbourLabels.begin(), neighbourLabels.end());
    typename std::vector<label_t>::iterator newVecEnd = std::unique(neighbourLabels.begin(), neighbourLabels.end());
    neighbourLabels.resize(newVecEnd - neighbourLabels.begin());

    return neighbourLabels;
}

double ContourRelaxation::calculateCost(cv::Mat const& labelImage, cv::Point2i const& curPixelCoords,
                                        label_t const& pretendLabel, std::vector<label_t> const& neighbourLabels) const {
    assert(labelImage.type() == cv::DataType<label_t>::type);
    assert(curPixelCoords.inside(cv::Rect(0, 0, labelImage.cols, labelImage.rows)));

    // Calculate clique cost.
    double cost = calculateCliqueCost(labelImage, curPixelCoords, pretendLabel);

    // Calculate and add up the costs of all features.
    label_t const oldLabel = labelImage.at<label_t>(curPixelCoords);

    for (FeatureIterator it_curFeature = this->features.begin(); it_curFeature != this->features.end(); ++it_curFeature) {
        cost += (*it_curFeature).weight * (*it_curFeature).feature->calculateCost(curPixelCoords, oldLabel, pretendLabel, neighbourLabels);
    }

    return cost;
}

double ContourRelaxation::calculateCliqueCost(cv::Mat const& labelImage, cv::Point2i const& curPixelCoords,
                                              label_t const& pretendLabel) const {
    assert(labelImage.type() == cv::DataType<label_t>::type);
    assert(curPixelCoords.inside(cv::Rect(0, 0, labelImage.cols, labelImage.rows)));

    // Find number of (direct / diagonal) cliques around pixelIndex, pretending the pixel at
    // curPixelCoords belongs to pretendLabel. Then calculate and return the associated combined cost.

    // Create a rectangle spanning the image area. This will be used to check if points are inside the image area.
    cv::Rect boundaryRect(0, 0, labelImage.cols, labelImage.rows);

    // Direct cliques.

    // Store the differences in coordinates of all direct cliques in reference to the central pixel.
    // Fill this static vector on the first function call, the elements will never change.
    static const std::vector<cv::Point2i> directCoordDiffs = {cv::Point2i(-1, 0), cv::Point2i(1, 0), cv::Point2i(0, -1), cv::Point2i(0, 1)};

    int numDirectCliques = 0;

    // Loop over all direct clique coordinate differences.
    // Translate the central pixel by the current difference.
    // If the resulting coordinates are inside the image area, and the label there differs from the pretended label of the
    // central pixel, increase the number of direct cliques.
    for (std::vector<cv::Point2i>::const_iterator it_coordDiff = directCoordDiffs.begin(); it_coordDiff != directCoordDiffs.end(); ++it_coordDiff) {
        cv::Point2i comparisonCoords = curPixelCoords + *it_coordDiff;

        if (comparisonCoords.inside(boundaryRect) &&
            labelImage.at<label_t>(comparisonCoords) != pretendLabel) {
            ++numDirectCliques;
        }
    }

    // Diagonal cliques.

    // Store the differences in coordinates of all diagonal cliques in reference to the central pixel.
    // Fill this static vector on the first function call, the elements will never change.
    static const std::vector<cv::Point2i> diagonalCoordDiffs = {cv::Point2i(-1, -1), cv::Point2i(-1, 1), cv::Point2i(1, -1), cv::Point2i(1, 1)};

    int numDiagonalCliques = 0;

    // Loop over all diagonal clique coordinate differences.
    // Translate the central pixel by the current difference.
    // If the resulting coordinates are inside the image area, and the label there differs from the pretended label of the
    // central pixel, increase the number of diagonal cliques.
    for (std::vector<cv::Point2i>::const_iterator it_coordDiff = diagonalCoordDiffs.begin(); it_coordDiff != diagonalCoordDiffs.end(); ++it_coordDiff) {
        cv::Point2i comparisonCoords = curPixelCoords + *it_coordDiff;

        if (comparisonCoords.inside(boundaryRect) &&
            labelImage.at<label_t>(comparisonCoords) != pretendLabel) {
            ++numDiagonalCliques;
        }
    }

    // Calculate and return the combined clique cost.
    return numDirectCliques * directCliqueCost + numDiagonalCliques * diagonalCliqueCost;
}

void ContourRelaxation::computeBoundaryMap(cv::Mat const& labelImage, cv::Mat& out_boundaryMap) const {
    assert(labelImage.type() == cv::DataType<label_t>::type);

    cv::cuda::Stream stream;

    cv::cuda::GpuMat labelImageGpu;
    labelImageGpu.upload(labelImage, stream);

    cv::cuda::GpuMat out_boundaryMapGpu;
    out_boundaryMapGpu.create(labelImage.size(), CV_8UC1);

    dim3 threadsPerBlock(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y);
    dim3 numBlocks((labelImage.cols + threadsPerBlock.x * X_BATCH - 1) / (threadsPerBlock.x * X_BATCH), (labelImage.rows + threadsPerBlock.y * Y_BATCH - 1) / (threadsPerBlock.y * Y_BATCH));

    cudaStream_t cudaStream = cv::cuda::StreamAccessor::getStream(stream);

    computeBoundaries<<<numBlocks, threadsPerBlock, 0, cudaStream>>>(labelImageGpu, out_boundaryMapGpu);

    CUDA_SAFE_CALL(logger, cudaGetLastError());

    out_boundaryMapGpu.download(out_boundaryMap, stream);

    stream.waitForCompletion();
}

void ContourRelaxation::computeBoundaryMapSmall(cv::Mat const& labelImage, cv::Mat& out_boundaryMap) const {
    assert(labelImage.type() == cv::DataType<label_t>::type);

    // Initialize (or reset) boundary map with zeros.
    out_boundaryMap = cv::Mat::zeros(labelImage.size(), cv::DataType<unsigned char>::type);

    // For each pixel, compare with neighbors. If different label, set both to 1 (= boundary pixel).
    // Compare only half of the neighbors, the other half will be compared when they themselves are the current pixel.
    for (int row = 0; row < labelImage.rows; ++row) {
        label_t const* const labelImageUpperRowPtr = labelImage.ptr<label_t>(row);
        unsigned char* const boundaryImageUpperRowPtr = out_boundaryMap.ptr<unsigned char>(row);

        label_t const* labelImageLowerRowPtr = 0;
        unsigned char* boundaryImageLowerRowPtr = 0;

        // Check whether we have one more row downwards.
        // We can only get the row pointers to that row if it exists, obviously.
        bool canLookDown = false;
        if (row < labelImage.rows - 1) {
            labelImageLowerRowPtr = labelImage.ptr<label_t>(row + 1);
            boundaryImageLowerRowPtr = out_boundaryMap.ptr<unsigned char>(row + 1);
            canLookDown = true;
        }

        for (int col = 0; col < labelImage.cols; ++col) {
            // Check whether we have one more column to the right.
            bool canLookRight = false;
            if (col < labelImage.cols - 1) {
                canLookRight = true;
            }

            // Neighbor to the right.
            if (canLookRight) {
                if (labelImageUpperRowPtr[col] != labelImageUpperRowPtr[col + 1]) {
                    boundaryImageUpperRowPtr[col] = 1;
                    boundaryImageUpperRowPtr[col + 1] = 1;
                }
            }

            // Neighbor to the bottom.
            if (canLookDown) {
                if (labelImageUpperRowPtr[col] != labelImageLowerRowPtr[col]) {
                    boundaryImageUpperRowPtr[col] = 1;
                    boundaryImageLowerRowPtr[col] = 1;
                }
            }

            // Neighbor to the bottom right.
            if (canLookDown && canLookRight) {
                if (labelImageUpperRowPtr[col] != labelImageLowerRowPtr[col + 1]) {
                    boundaryImageUpperRowPtr[col] = 1;
                    boundaryImageLowerRowPtr[col + 1] = 1;
                }
            }

            // Neighbor to the bottom left.
            if (canLookDown && col > 0) {
                if (labelImageUpperRowPtr[col] != labelImageLowerRowPtr[col - 1]) {
                    boundaryImageUpperRowPtr[col] = 1;
                    boundaryImageLowerRowPtr[col - 1] = 1;
                }
            }
        }
    }
}

void ContourRelaxation::updateBoundaryMap(cv::Mat const& labelImage, cv::Point2i const& curPixelCoords,
                                          cv::Mat& boundaryMap) const {
    // Update the boundary map in the 8-neighbourhood around curPixelCoords.
    // This needs to be done each time a pixel's label was changed.

    assert(labelImage.type() == cv::DataType<label_t>::type);
    assert(boundaryMap.type() == cv::DataType<unsigned char>::type);
    assert(boundaryMap.size() == labelImage.size());
    assert(curPixelCoords.inside(cv::Rect(0, 0, labelImage.cols, labelImage.rows)));

    // The current pixel can influence all pixels in the 8-neighborhood (and itself).
    // But for the neighbors we also need to look at all their neighbors, so we need to
    // scan a 5x5 window centered around the current pixel, but only update the central 3x3 window
    // in the boundary map.

    // Find out how far we can look in all four directions around the current pixel, i.e. handle border pixels.
    // We are only interested in a maximum distance of 2 in all directions, resulting in a 5x5 window.
    unsigned char const win5SizeLeft = std::min(2, curPixelCoords.x);
    unsigned char const win5SizeRight = std::min(2, labelImage.cols - 1 - curPixelCoords.x);
    unsigned char const win5SizeTop = std::min(2, curPixelCoords.y);
    unsigned char const win5SizeBottom = std::min(2, labelImage.rows - 1 - curPixelCoords.y);

    cv::Rect const window5by5(curPixelCoords.x - win5SizeLeft, curPixelCoords.y - win5SizeTop,
                              win5SizeLeft + 1 + win5SizeRight, win5SizeTop + 1 + win5SizeBottom);

    // Compute a boundary map for the (maximum) 5x5 window around the current pixel.
    cv::Mat const labelArray5Window = labelImage(window5by5);
    cv::Mat boundaryMap5by5;
    computeBoundaryMapSmall(labelArray5Window, boundaryMap5by5);

    // Find out which parts of the 8-neighborhood are available around the current pixel
    // and get a window for this potentially cropped 8-neighborhood.
    unsigned char const win3SizeLeft = std::min(1, curPixelCoords.x);
    unsigned char const win3SizeRight = std::min(1, labelImage.cols - 1 - curPixelCoords.x);
    unsigned char const win3SizeTop = std::min(1, curPixelCoords.y);
    unsigned char const win3SizeBottom = std::min(1, labelImage.rows - 1 - curPixelCoords.y);

    cv::Rect const window3by3(curPixelCoords.x - win3SizeLeft, curPixelCoords.y - win3SizeTop,
                              win3SizeLeft + 1 + win3SizeRight, win3SizeTop + 1 + win3SizeBottom);

    // Get the coordinates of the top-left corner of the (cropped) 8-neighborhood in the
    // temporary 5x5 boundary map. The width and height of the 8-neighborhood is the same
    // as above.
    unsigned char const tempBoundaryMapWin3x = std::max(win5SizeLeft - 1, 0);
    unsigned char const tempBoundaryMapWin3y = std::max(win5SizeTop - 1, 0);

    cv::Rect const tempBoundaryMapWin3by3(tempBoundaryMapWin3x, tempBoundaryMapWin3y,
                                          window3by3.width, window3by3.height);

    // Copy the central (cropped) 3x3 window of the up-to-date (cropped) 5x5 boundary map
    // to the (cropped) 3x3 window in the full boundary map.
    cv::Mat boundaryMap3Window = boundaryMap(window3by3);
    boundaryMap5by5(tempBoundaryMapWin3by3).copyTo(boundaryMap3Window);
}

void ContourRelaxation::setData(const DataType type, const cv::Mat& data) {
    for (FeatureIterator it_curFeature = this->features.begin(); it_curFeature != this->features.end(); ++it_curFeature) {
        (*it_curFeature).feature->setData(type, data);
    }
}
}  // namespace cart::contour