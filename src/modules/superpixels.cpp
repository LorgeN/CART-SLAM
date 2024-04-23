#include "modules/superpixels.hpp"

#include <opencv2/cudaimgproc.hpp>

#include "modules/superpixels/contourrelaxation/initialization.hpp"
#include "modules/superpixels/visualization.cuh"
#include "utils/modules.hpp"

namespace cart {
SuperPixelModule::SuperPixelModule(
    const unsigned int numIterations,
    const unsigned int blockWidth,
    const unsigned int blockHeight,
    const double directCliqueCost,
    const double compactnessWeight)
    : SuperPixelModule::SyncWrapperSystemModule("SuperPixelDetect"),
      numIterations(numIterations) {
    if (blockWidth < 1 || blockHeight < 1) {
        throw std::invalid_argument("blockWidth and blockHeight must be more than 1");
    }

    if (directCliqueCost < 0) {
        throw std::invalid_argument("directCliqueCost must be non-negative");
    }

    if (compactnessWeight < 0) {
        throw std::invalid_argument("compactnessWeight must be non-negative");
    }

#ifdef CARTSLAM_IMAGE_MAKE_GRAYSCALE
    std::vector<contour::FeatureType> enabledFeatures = {contour::Grayvalue, contour::Compactness};
#else
    std::vector<contour::FeatureType> enabledFeatures = {contour::Color, contour::Compactness};
#endif

    cv::Mat initialLabelImage = contour::createBlockInitialization(cv::Size(CARTSLAM_IMAGE_RES_X, CARTSLAM_IMAGE_RES_Y), blockWidth, blockHeight);
    this->contourRelaxation = boost::make_shared<contour::ContourRelaxation>(enabledFeatures, initialLabelImage, directCliqueCost, directCliqueCost / sqrt(2));
    this->contourRelaxation->setCompactnessData(compactnessWeight);
}

system_data_t SuperPixelModule::runInternal(System &system, SystemRunData &data) {
    // TODO: CUDA-ify this entire thing
    cv::cuda::GpuMat image;
    cv::cuda::Stream stream;

#ifdef CARTSLAM_IMAGE_MAKE_GRAYSCALE
    // Generate a 3-channel version of the grayscale image, which we will need later on
    // to generate the boundary overlay. Save it in the "image" variable so we won't
    // have to care about the original type of the image anymore.
    cv::Mat imageGray;
    cv::cuda::cvtColor(getReferenceImage(data.dataElement), image, cv::COLOR_GRAY2BGR, 0, stream);
    image.download(imageGray, stream);

    stream.waitForCompletion();
#else
    // Convert image to YUV-like YCrCb for uncorrelated color channels.
    cv::cuda::cvtColor(getReferenceImage(data.dataElement), image, cv::COLOR_BGR2YCrCb, 0, stream);
    std::vector<cv::Mat> imageYCrCbChannels;

    cv::Mat imageYCrCb;
    image.download(imageYCrCb, stream);

    stream.waitForCompletion();

    cv::split(imageYCrCb, imageYCrCbChannels);
#endif

    int numIterations = this->numIterations;

    cv::Mat relaxedLabelImage;

    {
        // Lock the mutex to protect the contour relaxation object, which is not thread safe
        boost::lock_guard<boost::mutex> lock(this->mutex);

#ifdef CARTSLAM_IMAGE_MAKE_GRAYSCALE
        this->contourRelaxation->setGrayvalueData(imageGray);
#else
        this->contourRelaxation->setColorData(imageYCrCbChannels[0], imageYCrCbChannels[1], imageYCrCbChannels[2]);
#endif

        this->contourRelaxation->relax(numIterations, relaxedLabelImage, cv::noArray());
    }

#ifndef CARTSLAM_IMAGE_MAKE_GRAYSCALE
    // Convert region-mean image back to BGR.
    // cv::cvtColor(regionMeanImage, regionMeanImage, cv::COLOR_YCrCb2BGR);
#endif

    return MODULE_RETURN_SHARED(CARTSLAM_KEY_SUPERPIXELS, cv::Mat, relaxedLabelImage);
}

boost::future<system_data_t> SuperPixelVisualizationModule::run(System &system, SystemRunData &data) {
    auto promise = boost::make_shared<boost::promise<system_data_t>>();

    boost::asio::post(system.getThreadPool(), [this, promise, &system, &data]() {
        auto labelImage = data.getData<cv::Mat>(CARTSLAM_KEY_SUPERPIXELS);

        cv::cuda::GpuMat labels;
        labels.upload(*labelImage);

        cv::cuda::GpuMat image = getReferenceImage(data.dataElement);

        cv::cuda::GpuMat boundaryOverlay;

        cart::contour::computeBoundaryOverlay(image, labels, boundaryOverlay);

        cv::Mat boundaryOverlayCpu;
        boundaryOverlay.download(boundaryOverlayCpu);

        this->imageThread->setImageIfLater(boundaryOverlayCpu, data.id);
        promise->set_value(MODULE_NO_RETURN_VALUE);
    });

    return promise->get_future();
}
}  // namespace cart