#pragma once

#include <opencv2/opencv.hpp>

#include "cartslam.hpp"
#include "datasource.hpp"
#include "modules/superpixels/contourrelaxation/contourrelaxation.hpp"
#include "utils/ui.hpp"

#define CARTSLAM_KEY_SUPERPIXELS "superpixels"

// Default value for compactness feature weight depends on the type of image.
// Since color images generate a total of 3 cost terms for the separate channels, we choose
// the compactness weight at 3 times the value for grayvalue images to preserve the relative weighting.
#ifdef CARTSLAM_IMAGE_MAKE_GRAYSCALE
#define CARTSLAM_DEFAULT_COMPACTNESS_WEIGHT 0.01
#else
#define CARTSLAM_DEFAULT_COMPACTNESS_WEIGHT 0.03
#endif

namespace cart {
struct image_super_pixels_t {
    image_super_pixels_t(const cv::Mat &relaxedLabelImage, const cv::Mat &regionMeanImage)
        : relaxedLabelImage(relaxedLabelImage), regionMeanImage(regionMeanImage){};

    cv::Mat relaxedLabelImage;
    cv::Mat regionMeanImage;
};

class SuperPixelModule : public SyncWrapperSystemModule {
   public:
    SuperPixelModule(
        const unsigned int numIterations = 3,
        const unsigned int blockWidth = 20,
        const unsigned int blockHeight = 20,
        const double directCliqueCost = 0.3,
        const double compactnessWeight = CARTSLAM_DEFAULT_COMPACTNESS_WEIGHT);

    system_data_t runInternal(System &system, SystemRunData &data) override;

   private:
    boost::shared_ptr<contour::ContourRelaxation> contourRelaxation;
    const unsigned int numIterations;
    boost::mutex mutex;  // Mutex to protect the contour relaxation object
};

class SuperPixelVisualizationModule : public SystemModule {
   public:
    SuperPixelVisualizationModule() : SystemModule("SuperPixelVisualization", {CARTSLAM_KEY_SUPERPIXELS}) {
        this->imageThread = ImageProvider::create("Super Pixels");
    };

    boost::future<system_data_t> run(System &system, SystemRunData &data) override;

   private:
    boost::shared_ptr<ImageProvider> imageThread;
};
}  // namespace cart
