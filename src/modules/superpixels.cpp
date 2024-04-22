#include "modules/superpixels.hpp"

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
      numIterations(numIterations),
      blockWidth(blockWidth),
      blockHeight(blockHeight),
      directCliqueCost(directCliqueCost),
      diagonalCliqueCost(directCliqueCost / sqrt(2)) {
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

    this->contourRelaxation = boost::make_shared<contour::ContourRelaxation>(enabledFeatures);
    this->contourRelaxation->setCompactnessData(compactnessWeight);
}

system_data_t SuperPixelModule::runInternal(System &system, SystemRunData &data) {
    // TODO: CUDA-ify this entire thing
    cv::Mat image;
    getReferenceImage(data.dataElement).download(image);

#ifdef CARTSLAM_IMAGE_MAKE_GRAYSCALE
    // Generate a 3-channel version of the grayscale image, which we will need later on
    // to generate the boundary overlay. Save it in the "image" variable so we won't
    // have to care about the original type of the image anymore.
    cv::Mat imageGray = image.clone();
    cv::cvtColor(imageGray, image, cv::COLOR_GRAY2BGR);

    this->contourRelaxation->setGrayvalueData(imageGray);
#else
    // Convert image to YUV-like YCrCb for uncorrelated color channels.
    cv::Mat imageYCrCb;
    cv::cvtColor(image, imageYCrCb, cv::COLOR_BGR2YCrCb);
    std::vector<cv::Mat> imageYCrCbChannels;
    cv::split(imageYCrCb, imageYCrCbChannels);

    this->contourRelaxation->setColorData(imageYCrCbChannels[0], imageYCrCbChannels[1], imageYCrCbChannels[2]);
#endif

    cv::Mat labelImage;

    if (data.id == 1) {
        labelImage = contour::createBlockInitialization(image.size(), this->blockWidth, this->blockHeight);
    } else {
        // Retrieve the labels from the previous run, and use those as a starting point
        labelImage = data.getRelativeRun(-1)->getDataAsync<image_super_pixels_t>(CARTSLAM_KEY_SUPERPIXELS).get()->relaxedLabelImage;
    }

    cv::Mat relaxedLabelImage;
    cv::Mat regionMeanImage;

    this->contourRelaxation->relax(labelImage, this->directCliqueCost, this->diagonalCliqueCost,
                                   this->numIterations, relaxedLabelImage, regionMeanImage);

#ifndef CARTSLAM_IMAGE_MAKE_GRAYSCALE
    // Convert region-mean image back to BGR.
    //cv::cvtColor(regionMeanImage, regionMeanImage, cv::COLOR_YCrCb2BGR);
#endif

    return MODULE_RETURN_SHARED(CARTSLAM_KEY_SUPERPIXELS, image_super_pixels_t, relaxedLabelImage, regionMeanImage);
}

boost::future<system_data_t> SuperPixelVisualizationModule::run(System &system, SystemRunData &data) {
    auto promise = boost::make_shared<boost::promise<system_data_t>>();

    boost::asio::post(system.getThreadPool(), [this, promise, &system, &data]() {
        auto pixels = data.getData<image_super_pixels_t>(CARTSLAM_KEY_SUPERPIXELS);

        LOG4CXX_DEBUG(this->logger, "Visualizing superpixels for frame " << data.id << " with " << pixels->relaxedLabelImage.cols << "x" << pixels->relaxedLabelImage.rows << " pixels");

        cv::cuda::GpuMat labels;
        labels.upload(pixels->relaxedLabelImage);

        cv::cuda::GpuMat image = getReferenceImage(data.dataElement);

        cv::cuda::GpuMat boundaryOverlay;

        cart::contour::computeBoundaryOverlay(image, labels, boundaryOverlay);

        cv::Mat boundaryOverlayCpu;
        boundaryOverlay.download(boundaryOverlayCpu);

        LOG4CXX_DEBUG(this->logger, "Visualized superpixels for frame " << data.id << " with " << boundaryOverlayCpu.cols << "x" << boundaryOverlayCpu.rows << " pixels");

        this->imageThread->setImageIfLater(boundaryOverlayCpu, data.id);
        promise->set_value(MODULE_NO_RETURN_VALUE);
    });

    return promise->get_future();
}
}  // namespace cart