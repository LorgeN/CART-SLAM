#pragma once

#include <cuda_runtime.h>
#include <math.h>

#include <opencv2/opencv.hpp>

#include "constants.hpp"
#include "features/ifeature.hpp"
#include "initialization.hpp"

namespace cart::contour {
/**
 * @brief Struct for storing a feature and its weight.
 */
struct feature_container_t {
    boost::shared_ptr<IFeature> feature;
    double weight;
};

/**
 * @class ContourRelaxation
 * @brief Main class for applying Contour Relaxation to a label image, using an arbitrary set of features.
 */
class ContourRelaxation {
   private:
    std::vector<feature_container_t> features;
    log4cxx::LoggerPtr logger;

    /**
     * @brief Create a binary map highlighting pixels on the boundary of their respective labels (1 for boundary pixels, 0 otherwise).
     * @param labelImage the current label image, contains one label identifier per pixel
     * @param out_boundaryMap the resulting boundary map, will be (re)allocated if necessary, binary by nature but stored as unsigned char
     */
    void computeBoundaryMap(const cv::cuda::GpuMat& labelImage, cv::cuda::GpuMat& out_boundaryMap) const;

    cv::cuda::GpuMat labelImage;
    const double directCliqueCost;
    const double diagonalCliqueCost;
    label_t maxLabelId;

   public:
    /**
     * @brief Constructor. Create a ContourRelaxation object with the specified features enabled.
     * @param initialLabelImage the initial label image, containing one label identifier per pixel
     * @param directCliqueCost the cost of a direct clique
     * @param diagonalCliqueCost the cost of a diagonal clique
     */
    ContourRelaxation(const cv::cuda::GpuMat initialLabelImage, const label_t maxLabelId, const double directCliqueCost = 0.3,
                      const double diagonalCliqueCost = 0.3 / sqrt(2));

    void setLabelImage(const cv::cuda::GpuMat& labelImage, const label_t maxLabelId);

    template <typename T, typename... Args>
    void addFeature(const double weight, Args... args) {
        if (weight <= 0) {
            return;
        }

        this->addFeature(boost::make_shared<T>(args...), weight);
    }

    void addFeature(boost::shared_ptr<IFeature> feature, const double weight);

    void relax(unsigned int const numIterations, const cv::cuda::GpuMat& image, const cv::cuda::GpuMat& disparity, cv::OutputArray out_labelImage);
};
}  // namespace cart::contour