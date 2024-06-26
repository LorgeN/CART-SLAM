#include <opencv2/cudaimgproc.hpp>

#include "cartslam.hpp"
#include "modules/disparity.hpp"
#include "modules/superpixels.hpp"

#ifdef CARTSLAM_IMAGE_MAKE_GRAYSCALE
#include "modules/superpixels/contourrelaxation/features/grayvalue.cuh"
#else
#include "modules/superpixels/contourrelaxation/features/color.cuh"
#endif

#include "modules/superpixels/contourrelaxation/features/compactness.cuh"
#include "modules/superpixels/contourrelaxation/features/disparity.cuh"
#include "modules/superpixels/visualization.cuh"
#include "utils/modules.hpp"

namespace cart {
SuperPixelModule::SuperPixelModule(
    const cv::Size imageRes,
    const unsigned int initialIterations,
    const unsigned int iterations,
    const unsigned int blockSize,
    const unsigned int resetIterations,
    const double directCliqueCost,
    const double diagonalCliqueCost,
    const double compactnessWeight,
    const double progressiveCompactnessCost,
    const double imageWeight,
    const double disparityWeight)
    : SyncWrapperSystemModule("SuperPixelDetect"),
      initialIterations(initialIterations),
      iterations(iterations),
      resetIterations(resetIterations),
      blockSize(blockSize),
      requiresDisparityDerivative(disparityWeight > 0) {
    if (blockSize < 1) {
        throw std::invalid_argument("blockSize must be more than 1");
    }

    if (directCliqueCost < 0) {
        throw std::invalid_argument("directCliqueCost must be non-negative");
    }

    if (compactnessWeight < 0 || imageWeight < 0 || disparityWeight < 0) {
        throw std::invalid_argument("weight must be non-negative");
    }

    if (disparityWeight > 0) {
        this->requiresData.push_back(module_dependency_t(CARTSLAM_KEY_DISPARITY_DERIVATIVE));
    }

    this->providesData.push_back(CARTSLAM_KEY_SUPERPIXELS);
    this->providesData.push_back(CARTSLAM_KEY_SUPERPIXELS_MAX_LABEL);

    cv::cuda::GpuMat initialLabelImage;
    contour::createBlockInitialization(imageRes, blockSize, blockSize, initialLabelImage, this->maxLabelId);

    this->contourRelaxation = boost::make_shared<contour::ContourRelaxation>(initialLabelImage, this->maxLabelId, directCliqueCost, diagonalCliqueCost);

    this->contourRelaxation->addFeature<contour::CompactnessFeature>(compactnessWeight, progressiveCompactnessCost);
    this->contourRelaxation->addFeature<contour::DisparityFeature>(disparityWeight);

#ifdef CARTSLAM_IMAGE_MAKE_GRAYSCALE
    this->contourRelaxation->addFeature<contour::GrayvalueFeature>(imageWeight);
#else
    this->contourRelaxation->addFeature<contour::ColorFeature>(imageWeight);
#endif
}

system_data_t SuperPixelModule::runInternal(System &system, SystemRunData &data) {
    cv::cuda::GpuMat image;
    cv::cuda::Stream stream;

#ifdef CARTSLAM_IMAGE_MAKE_GRAYSCALE
    // Generate a 3-channel version of the grayscale image, which we will need later on
    // to generate the boundary overlay. Save it in the "image" variable so we won't
    // have to care about the original type of the image anymore.
    cv::cuda::cvtColor(getReferenceImage(data.dataElement), image, cv::COLOR_GRAY2BGR, 0, stream);
#else
    // Convert image to YUV-like YCrCb for uncorrelated color channels.
    cv::cuda::cvtColor(getReferenceImage(data.dataElement), image, cv::COLOR_BGR2YCrCb, 0, stream);
#endif

    stream.waitForCompletion();

    cv::cuda::GpuMat disparityDerivative;

    if (this->requiresDisparityDerivative) {
        disparityDerivative = *data.getData<cv::cuda::GpuMat>(CARTSLAM_KEY_DISPARITY_DERIVATIVE);
    }

    const unsigned int numIterations = (data.id == 1 || data.id % this->resetIterations == 0) ? this->initialIterations : this->iterations;

    cv::cuda::GpuMat relaxedLabelImage;

    {
        // Lock the mutex to protect the contour relaxation object, which is not thread safe
        boost::lock_guard<boost::mutex> lock(this->mutex);

        // We do a reset every at resetIterations, to prevent the superpixels from becoming very messy
        // over time. This is especially important when the camera is moving, as the superpixels will
        // drift over time to weird shapes. This has to be within the critical section so we dont modify
        // the object while it is being used.
        if (data.id % this->resetIterations == 0) {
            // Reset the labels every resetIterations
            cv::cuda::GpuMat initialLabelImage;
            contour::createBlockInitialization(image.size(), this->blockSize, this->blockSize, initialLabelImage, this->maxLabelId, stream);

            stream.waitForCompletion();

            this->contourRelaxation->setLabelImage(initialLabelImage, this->maxLabelId);
        }

        this->contourRelaxation->relax(numIterations, image, disparityDerivative, relaxedLabelImage);
    }

    return MODULE_RETURN_ALL(
        MODULE_MAKE_PAIR(CARTSLAM_KEY_SUPERPIXELS, cv::cuda::GpuMat, boost::move(relaxedLabelImage)),
        MODULE_MAKE_PAIR(CARTSLAM_KEY_SUPERPIXELS_MAX_LABEL, contour::label_t, this->maxLabelId));
}

bool SuperPixelVisualizationModule::updateImage(System &system, SystemRunData &data, cv::Mat &image) {
    auto labels = data.getData<cv::cuda::GpuMat>(CARTSLAM_KEY_SUPERPIXELS);

    cv::cuda::GpuMat imageGpu = getReferenceImage(data.dataElement);
    cv::cuda::GpuMat boundaryOverlay;

    cart::contour::computeBoundaryOverlay(this->logger, imageGpu, *labels, boundaryOverlay);

    boundaryOverlay.download(image);
    return true;
}
}  // namespace cart