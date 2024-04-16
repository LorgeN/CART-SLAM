#pragma once

#include <log4cxx/logger.h>

#include <boost/atomic.hpp>
#include <opencv2/ml.hpp>

#include "cartslam.hpp"
#include "modules/disparity.hpp"
#include "utils/ui.hpp"

#define CARTSLAM_KEY_PLANES "planes"
#define CARTSLAM_KEY_PLANE_PARAMETERS "plane_parameters"
#define CARTSLAM_KEY_DISPARITY_DERIVATIVE_HIST "disp_derivative_histogram"

namespace cart {
struct PlaneParameters {
    PlaneParameters() : horizontalCenter(0), horizontalVariance(0), verticalCenter(0), verticalVariance(0){};

    PlaneParameters(int horizontalCenter, int horizontalVariance, int verticalCenter, int verticalVariance)
        : horizontalCenter(horizontalCenter), horizontalVariance(horizontalVariance), verticalCenter(verticalCenter), verticalVariance(verticalVariance){};

    const int horizontalCenter;
    const int horizontalVariance;
    const int verticalCenter;
    const int verticalVariance;
};

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
        return PlaneParameters(horizontalCenter, horizontalVariance, verticalCenter, verticalVariance);
    }

    friend class DisparityPlaneSegmentationModule;

   protected:
    PlaneParameterProvider(int horizontalCenter, int horizontalVariance, int verticalCenter, int verticalVariance)
        : horizontalCenter(horizontalCenter), horizontalVariance(horizontalVariance), verticalCenter(verticalCenter), verticalVariance(verticalVariance){};
        
    virtual void updatePlaneParameters(log4cxx::LoggerPtr logger, System& system, SystemRunData& data, cv::Mat& histogram) = 0;

    int horizontalCenter;
    int horizontalVariance;
    int verticalCenter;
    int verticalVariance;
};

class HistogramPeakPlaneParameterProvider : public PlaneParameterProvider {
   public:
    HistogramPeakPlaneParameterProvider()
        : PlaneParameterProvider(0, 0, 0, 0){};

   protected:
    void updatePlaneParameters(log4cxx::LoggerPtr logger, System& system, SystemRunData& data, cv::Mat& histogram) override;
};

class DisparityPlaneSegmentationModule : public SyncWrapperSystemModule {
   public:
    DisparityPlaneSegmentationModule(
        boost::shared_ptr<PlaneParameterProvider> planeParameterProvider,
        const int updateInterval = 30, const int resetInterval = 10) : SyncWrapperSystemModule("PlaneSegmentation", {CARTSLAM_KEY_DISPARITY}),
                                                                       planeParameterProvider(planeParameterProvider),
                                                                       updateInterval(updateInterval),
                                                                       resetInterval(resetInterval){};

    system_data_t runInternal(System& system, SystemRunData& data) override;

    friend class DisparityPlaneSegmentationVisualizationModule;

   private:
    void updatePlaneParameters(System& system, SystemRunData& data);

    const int updateInterval;
    const int resetInterval;

    boost::shared_ptr<PlaneParameterProvider> planeParameterProvider;

    boost::shared_mutex derivativeHistogramMutex;
    cv::cuda::GpuMat derivativeHistogram;
};

class DisparityPlaneSegmentationVisualizationModule : public SystemModule {
   public:
    DisparityPlaneSegmentationVisualizationModule() : SystemModule("PlaneSegmentationVisualization", {CARTSLAM_KEY_PLANES}) {
        this->imageThread = ImageProvider::create("Plane Segmentation");
        this->histThread = ImageProvider::create("Plane Segmentation Histogram");
    };

    boost::future<system_data_t> run(System& system, SystemRunData& data) override;

   private:
    boost::shared_ptr<ImageProvider> imageThread;
    boost::shared_ptr<ImageProvider> histThread;
};
}  // namespace cart
