#pragma once

#include <log4cxx/logger.h>

#include <boost/atomic.hpp>
#include <opencv2/ml.hpp>

#include "cartslam.hpp"
#include "modules/disparity.hpp"
#include "utils/ui.hpp"

#define CARTSLAM_KEY_PLANES "planes"
#define CARTSLAM_KEY_PLANE_CENTER_HORIZONTAL "plane_center_horizontal"
#define CARTSLAM_KEY_PLANE_CENTER_VERTICAL "plane_center_vertical"
#define CARTSLAM_KEY_PLANE_VARIANCE_HORIZONTAL "plane_variance_horizontal"
#define CARTSLAM_KEY_PLANE_VARIANCE_VERTICAL "plane_variance_vertical"

namespace cart {
enum Plane {
    HORIZONTAL = 0,
    VERTICAL = 1,
    UNKNOWN = 2
};

class DisparityPlaneSegmentationModule : public SyncWrapperSystemModule {
   public:
    DisparityPlaneSegmentationModule(const int updateInterval = 1000) : SyncWrapperSystemModule("PlaneSegmentation", {CARTSLAM_KEY_DISPARITY}), updateInterval(updateInterval){};

    system_data_t runInternal(System& system, SystemRunData& data) override;

   private:
    void updatePlaneParameters(cv::cuda::GpuMat& derivates, SystemRunData& data);

    const int updateInterval;

    boost::atomic_bool planeParametersUpdated;
    boost::atomic_uint32_t lastUpdatedFrame;
    boost::atomic_int32_t horizontalCenter;
    boost::atomic_int32_t horizontalVariance;
    boost::atomic_int32_t verticalCenter;
    boost::atomic_int32_t verticalVariance;
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
