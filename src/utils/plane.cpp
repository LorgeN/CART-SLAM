#include "utils/plane.hpp"

#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <unordered_set>
#include <vector>

#include "modules/planeseg.hpp"
#include "utils/random.hpp"

// Implementation based on https://github.com/isl-org/Open3D/blob/main/cpp/open3d/geometry/PointCloudSegmentation.cpp
namespace cart::util {

class RANSACResult {
   public:
    RANSACResult() : fitness(0), inlierRmse(0) {}
    ~RANSACResult() {}

   public:
    double fitness;
    double inlierRmse;
};

RANSACResult evaluateDistance(const std::vector<cv::Point3d> &points, const cv::Vec4d model, double threshold) {
    RANSACResult result;

    double error = 0;
    size_t inlierCount = 0;

#pragma omp parallel for reduction(+ : error, inlierCount)
    for (size_t idx = 0; idx < points.size(); ++idx) {
        cv::Vec4d point(points[idx].x, points[idx].y, points[idx].z, 1);
        double distance = std::abs(model.dot(point));

        if (distance < threshold) {
            error += distance * distance;
            inlierCount++;
        }
    }

    if (inlierCount > 0) {
        result.fitness = static_cast<double>(inlierCount) / static_cast<double>(points.size());
        result.inlierRmse = std::sqrt(error / static_cast<double>(inlierCount));
    }

    return result;
}

// Find the plane such that the summed squared distance from the
// plane to all points is minimized.
//
// Reference:
// https://www.ilikebigbits.com/2015_03_04_plane_from_points.html
cv::Vec4d getPlaneFromPoints(const std::vector<cv::Point3d> &points,
                             const std::vector<size_t> &inliers) {
    cv::Point3d centroid(0, 0, 0);
    for (size_t idx : inliers) {
        centroid += points[idx];
    }

    centroid /= static_cast<double>(inliers.size());

    double xx = 0, xy = 0, xz = 0, yy = 0, yz = 0, zz = 0;

    for (size_t idx : inliers) {
        cv::Point3d r = points[idx] - centroid;
        xx += r.x * r.x;
        xy += r.x * r.y;
        xz += r.x * r.z;
        yy += r.y * r.y;
        yz += r.y * r.z;
        zz += r.z * r.z;
    }

    double detX = yy * zz - yz * yz;
    double detY = xx * zz - xz * xz;
    double detZ = xx * yy - xy * xy;

    if (detX <= 0 && detY <= 0 && detZ <= 0) {
        return cv::Vec4d(0, 0, 0, 0);
    }

    cv::Vec3d abc;
    if (detX > detY && detX > detZ) {
        abc = cv::Vec3d(detX, xz * yz - xy * zz, xy * yz - xz * yy);
    } else if (detY > detZ) {
        abc = cv::Vec3d(xz * yz - xy * zz, detY, xy * xz - yz * xx);
    } else {
        abc = cv::Vec3d(xy * yz - xz * yy, xy * xz - yz * xx, detZ);
    }

    abc /= cv::norm(abc);
    double d = -abc.dot(centroid);
    return cv::Vec4d(abc[0], abc[1], abc[2], d);
}

cv::Vec4d segmentPlane(const std::vector<cv::Point3d> &points, const double distThreshold, const int ransacN, const int iters, const double probability) {
    if (probability <= 0 || probability > 1) {
        throw std::invalid_argument("Probability must be in (0, 1].");
    }

    RANSACResult result;

    // Initialize the best plane model.
    cv::Vec4d model = cv::Vec4d(0, 0, 0, 0);

    size_t numPoints = points.size();
    util::RandomSampler sampler(numPoints);

    // Return if ransac_n is less than the required plane model parameters.
    if (ransacN < 4) {
        throw std::invalid_argument(
            "The number of points to fit the plane must be at least 4.");
    }

    if (numPoints < size_t(ransacN)) {
        throw new std::invalid_argument(
            "There must be at least 'ransac_n' points.");
    }

    // Use size_t here to avoid large integer which acceed max of int.
    size_t breakIter = std::numeric_limits<size_t>::max();
    size_t iter = 0;

#pragma omp parallel for schedule(static)
    for (int itr = 0; itr < iters; itr++) {
        if (iter > breakIter) {
            continue;
        }

        // Access the pre-generated sampled indices
        std::vector<size_t> inliers = sampler(ransacN);

        // Fit model to num_model_parameters randomly selected points among the
        // inliers.
        cv::Vec4d currModel = getPlaneFromPoints(points, inliers);

        if (cv::norm(currModel) == 0) {
            continue;
        }

        RANSACResult currResult = evaluateDistance(points, currModel, distThreshold);

#pragma omp critical
        {
            if (currResult.fitness > result.fitness || (currResult.fitness == result.fitness && currResult.inlierRmse < result.inlierRmse)) {
                result = currResult;
                model = currModel;

                if (result.fitness < 1.0) {
                    breakIter = std::min(log(1 - probability) / log(1 - pow(result.fitness, ransacN)), (double)iters);
                } else {
                    breakIter = 0;
                }
            }

            iter++;
        }
    }

    // Find the final inliers using best_plane_model.
    std::vector<size_t> finalInliers;
    if (cv::norm(model) != 0) {
        for (size_t idx = 0; idx < points.size(); ++idx) {
            cv::Vec4d point(points[idx].x, points[idx].y, points[idx].z, 1);
            double distance = std::abs(model.dot(point));

            if (distance < distThreshold) {
                finalInliers.emplace_back(idx);
            }
        }
    } else {
        return cv::Vec4d(0, 0, 0, 0);
    }

    // Improve best_plane_model using the final inliers.
    return getPlaneFromPoints(points, finalInliers);
}

}  // namespace cart::util