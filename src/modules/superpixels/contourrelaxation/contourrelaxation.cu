#include <inttypes.h>

#include <boost/make_shared.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>

#include "modules/superpixels/contourrelaxation/contourrelaxation.hpp"
#include "modules/superpixels/contourrelaxation/features/feature.cuh"
#include "utils/cuda.cuh"

#define THREADS_PER_BLOCK_X 16
#define THREADS_PER_BLOCK_Y 16
#define THREADS_PER_BLOCK_RELAX 64
#define X_BATCH 4
#define Y_BATCH 4
#define POINT_BATCH 8

#define SHARED_SIZE(x, y) ((x) * (2 + THREADS_PER_BLOCK_X)) * ((y) * (2 + THREADS_PER_BLOCK_Y))
#define LOCAL_INDEX(x, y) SHARED_INDEX(sharedPixelX + x, sharedPixelY + y, 1, 1, sharedRowStep)

#define NEIGHBOUR_INDEX(x, y) ((x + 1) + (y + 1) * 3)
#define OUT_OF_BOUNDS (1 << 14)  // A value that is guaranteed to be out of bounds

__global__ void computeBoundaries(cv::cuda::PtrStepSz<cart::contour::label_t> labels, cv::cuda::PtrStepSz<uint8_t> out) {
    __shared__ cart::contour::label_t sharedLabels[SHARED_SIZE(X_BATCH, Y_BATCH)];

    cart::copyToShared<cart::contour::label_t, X_BATCH, Y_BATCH, false>(sharedLabels, labels, 1, 1);

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int sharedPixelX = threadIdx.x * X_BATCH;
    int sharedPixelY = threadIdx.y * Y_BATCH;

    int pixelX = x * X_BATCH;
    int pixelY = y * Y_BATCH;

    size_t sharedRowStep = X_BATCH * blockDim.x;
    size_t outStep = out.step / sizeof(uint8_t);

    __syncthreads();

    for (int i = 0; i < Y_BATCH; i++) {
        for (int j = 0; j < X_BATCH; j++) {
            if (pixelX + j >= out.cols || pixelY + i >= out.rows) {
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

__device__ void getNeighbourLabels(cart::contour::label_t* const neighbourhood, cart::contour::label_t* neighbourLabels, size_t& neighbourLabelCount) {
    // Finds the unique neighbouring labels of the current pixel and stores them in the neighbourLabels array using a method similar to insertion sort

    for (int i = -1; i <= 1; i++) {
        for (int j = -1; j <= 1; j++) {
            cart::contour::label_t label = neighbourhood[NEIGHBOUR_INDEX(i, j)];
            if (label == OUT_OF_BOUNDS) {
                continue;
            }

            // Due to the small size of the neighbourhood, we can use a simple linear search to find the label. Using
            // a bitset was attempted, but there are a lot of complications with that approach, most notably it requiring
            // a lot of memory, and a dynamic size that is not known at compile time since it depends on max label.
            // Using shared memory also makes it somewhat tricky to use a bitset, since there is no good way to reset it.
            // An idea that could potentially be expanded on is the fact that neighbouring labels should be quite close together,
            // meaning it may be possible to use a bitset with a smaller fixed size.
            bool found = false;
            for (size_t k = 0; k < neighbourLabelCount; k++) {
                if (neighbourLabels[k] == label) {
                    found = true;
                    break;
                }
            }

            if (!found) {
                neighbourLabels[neighbourLabelCount++] = label;
            }
        }
    }
}

__device__ inline int checkClique(cart::contour::label_t* const neighbourhood, const cart::contour::label_t label, const int x, const int y) {
    cart::contour::label_t neighbour = neighbourhood[NEIGHBOUR_INDEX(x, y)];
    return neighbour != OUT_OF_BOUNDS && neighbour != label;
}

__device__ double calculateCliqueCost(const cart::contour::CRSettings settings, cart::contour::label_t* const neighbourhood, const cart::contour::label_t pretendLabel) {
    // Find number of (direct / diagonal) cliques around pixelIndex, pretending the pixel at
    // curPixelCoords belongs to pretendLabel. Then calculate and return the associated combined cost.

    // Direct cliques.
    int numDirectCliques = 0;
    numDirectCliques += checkClique(neighbourhood, pretendLabel, -1, 0);
    numDirectCliques += checkClique(neighbourhood, pretendLabel, 1, 0);
    numDirectCliques += checkClique(neighbourhood, pretendLabel, 0, -1);
    numDirectCliques += checkClique(neighbourhood, pretendLabel, 0, 1);

    // Diagonal cliques.
    int numDiagonalCliques = 0;
    numDiagonalCliques += checkClique(neighbourhood, pretendLabel, -1, -1);
    numDiagonalCliques += checkClique(neighbourhood, pretendLabel, -1, 1);
    numDiagonalCliques += checkClique(neighbourhood, pretendLabel, 1, -1);
    numDiagonalCliques += checkClique(neighbourhood, pretendLabel, 1, 1);

    // Calculate and return the combined clique cost.
    return numDirectCliques * settings.directCliqueCost + numDiagonalCliques * settings.diagonalCliqueCost;
}

__device__ double calculateCost(const cart::contour::CRSettings settings, cart::contour::label_t* const neighbourhood, const cart::contour::CRPoint curPixelCoords,
                                const cart::contour::label_t pretendLabel, const cart::contour::label_t* neighbourLabels, size_t neighbourLabelCount) {
    // Calculate clique cost.
    double cost = calculateCliqueCost(settings, neighbourhood, pretendLabel);

    // Calculate and add up the costs of all features.
    const cart::contour::label_t oldLabel = neighbourhood[NEIGHBOUR_INDEX(0, 0)];

    for (size_t i = 0; i < settings.numFeatures; i++) {
        auto feature = settings.features[i];
        cost += feature.weight * (*feature.feature)->calculateCost(curPixelCoords, oldLabel, pretendLabel, neighbourLabels, neighbourLabelCount);
    }

    return cost;
}

__global__ void findBorderPixels(cv::cuda::PtrStepSz<cart::contour::label_t> labels,
                                 cart::contour::CRPoint* borderPixels,
                                 unsigned int* borderCount) {
    __shared__ cart::contour::label_t sharedLabels[SHARED_SIZE(X_BATCH, Y_BATCH)];
    // Note that this is a shared memory allocation that could in theory cover every pixel. There was some artifacting
    // when allocating half of the block size, so it seems that in some cases the assumption that half of the pixels are
    // border pixels does not hold.
    __shared__ cart::contour::CRPoint borderPixelsShared[THREADS_PER_BLOCK_X * THREADS_PER_BLOCK_Y * X_BATCH * Y_BATCH];
    __shared__ unsigned int borderCountShared;
    __shared__ unsigned int startingIndex;
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        borderCountShared = 0;
    }

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int sharedPixelX = threadIdx.x * X_BATCH;
    int sharedPixelY = threadIdx.y * Y_BATCH;

    int pixelX = x * X_BATCH;
    int pixelY = y * Y_BATCH;

    size_t sharedRowStep = X_BATCH * blockDim.x;

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

            if (border) {
                borderPixelsShared[atomicAdd(&borderCountShared, 1)] = {
                    .x = pixelX + j,
                    .y = pixelY + i};
            }
        }
    }

    __syncthreads();

    if (threadIdx.x == 0 && threadIdx.y == 0) {
        startingIndex = atomicAdd(borderCount, borderCountShared);
    }

    __syncthreads();

    for (int i = threadIdx.x + (blockDim.x * threadIdx.y); i < borderCountShared; i += blockDim.x * blockDim.y) {
        borderPixels[startingIndex + i] = borderPixelsShared[i];
    }
}

__global__ void performRelaxation(cart::contour::CRSettings settings, cart::contour::CRPoint* borderPixels, cart::contour::label_t* newLabels, unsigned int* borderCount) {
    __shared__ cart::contour::label_t neighboursShared[THREADS_PER_BLOCK_RELAX * 9];
    __shared__ cart::contour::label_t neighbourhoods[THREADS_PER_BLOCK_RELAX * 9];

    const size_t baseIndex = blockIdx.x * blockDim.x * POINT_BATCH + threadIdx.x;
    const size_t labelStep = settings.labelImage.step / sizeof(cart::contour::label_t);

    cart::contour::label_t* const neighbours = neighboursShared + threadIdx.x * 9;
    // These are used to calculate neighbours and to calculate clique cost, so we copy to shared to avoid reading twice
    cart::contour::label_t* const neighbourhood = neighbourhoods + threadIdx.x * 9;

    for (size_t i = 0; i < POINT_BATCH; i++) {
        size_t index = baseIndex + i * blockDim.x;
        if (index >= *borderCount) {
            return;
        }

        cart::contour::CRPoint curPixelCoords = borderPixels[index];
        cart::contour::label_t currLabel = settings.labelImage[INDEX(curPixelCoords.x, curPixelCoords.y, labelStep)];

        // Copy 8-neighbourhood to shared memory
        for (int x = -1; x <= 1; x++) {
            for (int y = -1; y <= 1; y++) {
                int xCurr = curPixelCoords.x + x;
                int yCurr = curPixelCoords.y + y;

                if (xCurr < 0 || yCurr < 0 || xCurr >= settings.labelImage.cols || yCurr >= settings.labelImage.rows) {
                    neighbourhood[NEIGHBOUR_INDEX(x, y)] = OUT_OF_BOUNDS;  // Set to a value that is guaranteed to not be a label
                    continue;
                }

                neighbourhood[NEIGHBOUR_INDEX(x, y)] = settings.labelImage[INDEX(xCurr, yCurr, labelStep)];
            }
        }

        size_t neighbourLabelCount = 0;

        getNeighbourLabels(neighbourhood, neighbours, neighbourLabelCount);

        double minCost = DBL_MAX;
        cart::contour::label_t bestLabel = currLabel;

        // This is not great, and may lead to a lot of divergence. However, it is the best we can do for now.
        // An alternative is to do another step where we further process each pixel to a given neighbour value per thread or something similar
        for (size_t i = 0; i < neighbourLabelCount; i++) {
            double cost = calculateCost(settings, neighbourhood, curPixelCoords, neighbours[i], neighbours, neighbourLabelCount);

            if (cost < minCost) {
                minCost = cost;
                bestLabel = neighbours[i];
            }
        }

        newLabels[index] = bestLabel;
    }
}

__global__ void updateLabels(cart::contour::CRSettings settings, cart::contour::CRPoint* borderPixels, cart::contour::label_t* newLabels, unsigned int* borderCount) {
    size_t baseIndex = (blockIdx.x * blockDim.x + threadIdx.x) * POINT_BATCH;

    for (size_t i = 0; i < POINT_BATCH; i++) {
        size_t index = baseIndex + i;
        if (index >= *borderCount) {
            return;
        }

        cart::contour::CRPoint curPixelCoords = borderPixels[index];
        cart::contour::label_t currLabel = settings.labelImage[INDEX(curPixelCoords.x, curPixelCoords.y, settings.labelImage.step / sizeof(cart::contour::label_t))];
        cart::contour::label_t newLabel = newLabels[index];

        if (currLabel == newLabel) {
            continue;
        }

        for (size_t i = 0; i < settings.numFeatures; i++) {
            (*settings.features[i].feature)->updateStatistics(curPixelCoords, currLabel, newLabel);
        }

        settings.labelImage[INDEX(curPixelCoords.x, curPixelCoords.y, settings.labelImage.step / sizeof(cart::contour::label_t))] = newLabel;
    }
}

__global__ void setImageDataKernel(cart::contour::CUDAFeatureContainer* features, const size_t numFeatures, const cv::cuda::PtrStepSz<uint8_t> image) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        for (size_t i = 0; i < numFeatures; i++) {
            (*features[i].feature)->setImageData(image);
        }
    }
}

__global__ void setDisparityDataKernel(cart::contour::CUDAFeatureContainer* features, const size_t numFeatures, const cv::cuda::PtrStepSz<int16_t> disparity) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        for (size_t i = 0; i < numFeatures; i++) {
            (*features[i].feature)->setDisparityData(disparity);
        }
    }
}

__global__ void initializeStatisticsKernel(cart::contour::CUDAFeatureContainer* features, const size_t numFeatures, const cv::cuda::PtrStepSz<cart::contour::label_t> labelImage) {
    (*features[threadIdx.z].feature)->initializeStatistics(labelImage, X_BATCH, Y_BATCH);
}

__global__ void deleteFeatures(cart::contour::CUDAFeatureContainer* features, const size_t numFeatures) {
    for (size_t i = 0; i < numFeatures; i++) {
        delete *features[i].feature;
    }
}

namespace cart::contour {
ContourRelaxation::ContourRelaxation(const cv::cuda::GpuMat initialLabelImage, const label_t maxLabelId, const double directCliqueCost,
                                     const double diagonalCliqueCost) : directCliqueCost(directCliqueCost), diagonalCliqueCost(diagonalCliqueCost), labelImage(initialLabelImage), maxLabelId(maxLabelId) {
    this->logger = getLogger("ContourRelaxation");
}

void ContourRelaxation::setLabelImage(const cv::cuda::GpuMat& labelImage, const label_t maxLabelId) {
    this->labelImage = labelImage;
    this->maxLabelId = maxLabelId;
}

void ContourRelaxation::addFeature(boost::shared_ptr<IFeature> feature, const double weight) {
    feature_container_t container = {
        .feature = feature,
        .weight = weight,
    };

    this->features.push_back(container);
}

void ContourRelaxation::relax(unsigned int const numIterations, const cv::cuda::GpuMat& image, const cv::cuda::GpuMat& disparity, cv::OutputArray out_labelImage) {
    assert(this->labelImage.type() == cv::DataType<label_t>::type);
    assert(image.size() == this->labelImage.size());

    cudaStream_t stream;
    CUDA_SAFE_CALL(this->logger, cudaStreamCreate(&stream));

    cv::cuda::Stream cvStream = cv::cuda::StreamAccessor::wrapStream(stream);

    // Create the CUDA feature wrappers
    CUDAFeatureContainer* cudaFeaturesHost = new CUDAFeatureContainer[this->features.size()];
    for (size_t i = 0; i < this->features.size(); i++) {
        CUDA_SAFE_CALL(this->logger, cudaMallocAsync(&cudaFeaturesHost[i].feature, sizeof(CUDAFeature**), stream));
        this->features[i].feature->initializeCUDAFeature(cudaFeaturesHost[i].feature, this->maxLabelId, cvStream);
        cudaFeaturesHost[i].weight = this->features[i].weight;
    }

    // Create the feature list on device
    CUDAFeatureContainer* cudaFeatures;
    CUDA_SAFE_CALL(this->logger, cudaMallocAsync(&cudaFeatures, sizeof(CUDAFeatureContainer) * this->features.size(), stream));
    CUDA_SAFE_CALL(this->logger, cudaMemcpyAsync(cudaFeatures, cudaFeaturesHost, sizeof(CUDAFeatureContainer) * this->features.size(), cudaMemcpyHostToDevice, stream));

    // Set the data values
    setImageDataKernel<<<1, 1, 0, stream>>>(cudaFeatures, this->features.size(), image);

    if (!disparity.empty()) {
        setDisparityDataKernel<<<1, 1, 0, stream>>>(cudaFeatures, this->features.size(), disparity);
    }

    // Initialize the statistics
    dim3 threadsPerBlock(8, 8, this->features.size());
    dim3 numBlocks(ceil(this->labelImage.cols / (threadsPerBlock.x * X_BATCH)), ceil(this->labelImage.rows / (threadsPerBlock.y * Y_BATCH)));
    initializeStatisticsKernel<<<numBlocks, threadsPerBlock, 0, stream>>>(cudaFeatures, this->features.size(), this->labelImage);

    CRPoint* borderPixels;
    unsigned int* borderCount;
    label_t* newLabels;

    CUDA_SAFE_CALL(this->logger, cudaMallocAsync(&newLabels, sizeof(label_t) * (this->labelImage.cols * this->labelImage.rows), stream));
    CUDA_SAFE_CALL(this->logger, cudaMallocAsync(&borderPixels, sizeof(CRPoint) * (this->labelImage.cols * this->labelImage.rows), stream));
    CUDA_SAFE_CALL(this->logger, cudaMallocAsync(&borderCount, sizeof(unsigned int), stream));

    CRSettings settings = {
        .directCliqueCost = this->directCliqueCost,
        .diagonalCliqueCost = this->diagonalCliqueCost,
        .maxLabelId = this->maxLabelId,
        .labelImage = this->labelImage,
        .features = cudaFeatures,
        .numFeatures = this->features.size(),
    };

    unsigned int hostBorderCount;
    dim3 threadsPerBlockBorder(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y);
    dim3 numBlocksBorder(ceil(static_cast<double>(labelImage.cols) / (THREADS_PER_BLOCK_X * X_BATCH)), ceil(static_cast<double>(labelImage.rows) / (THREADS_PER_BLOCK_Y * Y_BATCH)));

    for (size_t i = 0; i < numIterations; i++) {
        // Reset the border count
        CUDA_SAFE_CALL(this->logger, cudaMemsetAsync(borderCount, 0, sizeof(unsigned int), stream));

        // Find the border pixels
        findBorderPixels<<<numBlocksBorder, threadsPerBlockBorder, 0, stream>>>(labelImage, borderPixels, borderCount);

        CUDA_SAFE_CALL(this->logger, cudaMemcpyAsync(&hostBorderCount, borderCount, sizeof(unsigned int), cudaMemcpyDeviceToHost, stream));

        // Synchronize stream and transfer the border count to the host
        CUDA_SAFE_CALL(this->logger, cudaStreamSynchronize(stream));
        CUDA_SAFE_CALL(this->logger, cudaGetLastError());

        // Perform the relaxation
        size_t gridSize = ceil(hostBorderCount / (static_cast<double>(THREADS_PER_BLOCK_RELAX) * POINT_BATCH));
        performRelaxation<<<gridSize, THREADS_PER_BLOCK_RELAX, 0, stream>>>(settings, borderPixels, newLabels, borderCount);
        updateLabels<<<gridSize, THREADS_PER_BLOCK_RELAX, 0, stream>>>(settings, borderPixels, newLabels, borderCount);
    }

    // Return the resulting label image.
    if (out_labelImage.needed() && out_labelImage.isGpuMat()) {
        out_labelImage.create(labelImage.size(), labelImage.type());
        labelImage.copyTo(out_labelImage.getGpuMat(), cvStream);
    }

    // Free the memory
    deleteFeatures<<<1, 1, 0, stream>>>(cudaFeatures, this->features.size());

    CUDA_SAFE_CALL(this->logger, cudaFreeAsync(borderPixels, stream));
    CUDA_SAFE_CALL(this->logger, cudaFreeAsync(borderCount, stream));

    // Free the pointers to the CUDA features
    for (size_t i = 0; i < this->features.size(); i++) {
        CUDA_SAFE_CALL(this->logger, cudaFreeAsync(cudaFeaturesHost[i].feature, stream));
    }

    CUDA_SAFE_CALL(this->logger, cudaFreeAsync(cudaFeatures, stream));

    delete[] cudaFeaturesHost;

    CUDA_SAFE_CALL(this->logger, cudaStreamSynchronize(stream));
    CUDA_SAFE_CALL(this->logger, cudaGetLastError());
    CUDA_SAFE_CALL(this->logger, cudaStreamDestroy(stream));
}
}  // namespace cart::contour