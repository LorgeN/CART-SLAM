#pragma once

#include "depth.hpp"
#include "disparity.hpp"
#include "module.hpp"
#include "superpixels.hpp"
#include "visualization.hpp"

#define CARTSLAM_KEY_PLANES_EQ "planes_eq"

namespace cart {

struct plane_fit_data_t {
    std::vector<cv::Vec4d> planes;
    std::vector<size_t> planeAssignments;
};

class SuperPixelPlaneFitModule : public SyncWrapperSystemModule {
   public:
    SuperPixelPlaneFitModule();

    system_data_t runInternal(System& system, SystemRunData& data) override;

   private:
    void attemptAssignment(
        const cv::cuda::GpuMat& superpixels,
        const cv::cuda::GpuMat& depth,
        const std::vector<cv::Vec4d>& planes,
        size_t& assignedPlanes,
        const cart::contour::label_t maxLabelId,
        const double threshold,
        std::vector<size_t>& planeAssignments);
};

class SuperPixelPlaneFitVisualizationModule : public VisualizationModule {
   public:
    SuperPixelPlaneFitVisualizationModule() : VisualizationModule("SuperPixelPlaneFitVisualization") {
        this->requiresData.push_back(module_dependency_t(CARTSLAM_KEY_PLANES_EQ));
        this->requiresData.push_back(module_dependency_t(CARTSLAM_KEY_SUPERPIXELS));
        this->requiresData.push_back(module_dependency_t(CARTSLAM_KEY_SUPERPIXELS_MAX_LABEL));
    };

    bool updateImage(System& system, SystemRunData& data, cv::Mat& image) override;
};
}  // namespace cart