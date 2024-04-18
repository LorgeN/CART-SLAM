#pragma once

#include <log4cxx/logger.h>

#include <boost/atomic.hpp>
#include <opencv2/ml.hpp>

#include "cartslam.hpp"
#include "modules/disparity.hpp"
#include "modules/optflow.hpp"
#include "utils/ui.hpp"

#define CARTSLAM_KEY_PLANES "planes"
#define CARTSLAM_KEY_PLANES_UNSMOOTHED "planes_unsmoothed"
#define CARTSLAM_KEY_PLANE_PARAMETERS "plane_parameters"
#define CARTSLAM_KEY_DISPARITY_DERIVATIVE_HIST "disp_derivative_histogram"
#define CARTSLAM_PLANE_COUNT 3

namespace cart {
struct PlaneParameters {
    PlaneParameters(const int horizontalCenter, const int verticalCenter, const std::pair<int, int> horizontalRange, const std::pair<int, int> verticalRange)
        : horizontalCenter(horizontalCenter), verticalCenter(verticalCenter), horizontalRange(horizontalRange), verticalRange(verticalRange){};

    const std::pair<int, int> horizontalRange;
    const std::pair<int, int> verticalRange;

    const int horizontalCenter;
    const int verticalCenter;
};

// CARTSLAM_PLANE_COUNT contains the number of planes that we are segmenting
enum Plane {
    HORIZONTAL = 0,
    VERTICAL = 1,
    UNKNOWN = 2
};

template <Plane>
struct PlaneColor {};

template <>
struct PlaneColor<Plane::HORIZONTAL> {
    static constexpr int r = 0;
    static constexpr int g = 0;
    static constexpr int b = 255;
};

template <>
struct PlaneColor<Plane::VERTICAL> {
    static constexpr int r = 0;
    static constexpr int g = 255;
    static constexpr int b = 0;
};

template <>
struct PlaneColor<Plane::UNKNOWN> {
    static constexpr int r = 255;
    static constexpr int g = 0;
    static constexpr int b = 0;
};

template <Plane P>
typename cv::Scalar planeColor() {
    // OpenCV uses BGR as the color order
    return cv::Scalar(PlaneColor<P>::b, PlaneColor<P>::g, PlaneColor<P>::r);
}

class DisparityPlaneSegmentationModule;
class DisparityPlaneSegmentationVisualizationModule;

class PlaneParameterProvider {
   public:
    PlaneParameters getPlaneParameters() const {
        return PlaneParameters(horizontalCenter, verticalCenter, horizontalRange, verticalRange);
    }

    friend class DisparityPlaneSegmentationModule;

   protected:
    PlaneParameterProvider(const int horizontalCenter = 0, const int verticalCenter = 0, const std::pair<int, int> horizontalRange = std::make_pair(0, 0), const std::pair<int, int> verticalRange = std::make_pair(0, 0))
        : horizontalCenter(horizontalCenter), verticalCenter(verticalCenter), horizontalRange(horizontalRange), verticalRange(verticalRange){};

    virtual void updatePlaneParameters(log4cxx::LoggerPtr logger, System& system, SystemRunData& data, cv::Mat& histogram) = 0;

    std::pair<int, int> horizontalRange;
    std::pair<int, int> verticalRange;

    int horizontalCenter;
    int verticalCenter;
};

class HistogramPeakPlaneParameterProvider : public PlaneParameterProvider {
   public:
    HistogramPeakPlaneParameterProvider(){};

   protected:
    void updatePlaneParameters(log4cxx::LoggerPtr logger, System& system, SystemRunData& data, cv::Mat& histogram) override;
};

class StaticPlaneParameterProvider : public PlaneParameterProvider {
   public:
    StaticPlaneParameterProvider(const int horizontalCenter, const int verticalCenter, const std::pair<int, int> horizontalRange, const std::pair<int, int> verticalRange)
        : PlaneParameterProvider(horizontalCenter, verticalCenter, horizontalRange, verticalRange){};

   protected:
    void updatePlaneParameters(log4cxx::LoggerPtr logger, System& system, SystemRunData& data, cv::Mat& histogram) override{};
};

class DisparityPlaneSegmentationModule : public SyncWrapperSystemModule {
   public:
    DisparityPlaneSegmentationModule(
        boost::shared_ptr<PlaneParameterProvider> planeParameterProvider,
        const int updateInterval = 30, const int resetInterval = 10, const bool useTemporalSmoothing = false)
        // Need optical flow for temporal smoothing
        : SyncWrapperSystemModule("PlaneSegmentation", useTemporalSmoothing ? std::vector<std::string>{CARTSLAM_KEY_DISPARITY, CARTSLAM_KEY_OPTFLOW} : std::vector<std::string>{CARTSLAM_KEY_DISPARITY}),
          planeParameterProvider(planeParameterProvider),
          updateInterval(updateInterval),
          resetInterval(resetInterval),
          useTemporalSmoothing(useTemporalSmoothing){};

    system_data_t runInternal(System& system, SystemRunData& data) override;

    friend class DisparityPlaneSegmentationVisualizationModule;

   private:
    void updatePlaneParameters(System& system, SystemRunData& data);

    const bool useTemporalSmoothing;
    const int updateInterval;
    const int resetInterval;

    boost::shared_ptr<PlaneParameterProvider> planeParameterProvider;

    boost::shared_mutex derivativeHistogramMutex;
    cv::cuda::GpuMat derivativeHistogram;
};

class DisparityPlaneSegmentationVisualizationModule : public SystemModule {
   public:
    DisparityPlaneSegmentationVisualizationModule(bool showHistogram = true, bool showStacked = true) 
    : SystemModule("PlaneSegmentationVisualization", {CARTSLAM_KEY_PLANES}), showHistogram(showHistogram), showStacked(showStacked){
        this->imageThread = ImageProvider::create("Plane Segmentation");
        this->histThread = ImageProvider::create("Plane Segmentation Histogram");
    };

    boost::future<system_data_t> run(System& system, SystemRunData& data) override;

   private:
    const bool showHistogram;
    const bool showStacked;

    boost::shared_ptr<ImageProvider> imageThread;
    boost::shared_ptr<ImageProvider> histThread;
};
}  // namespace cart
