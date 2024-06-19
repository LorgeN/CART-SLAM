#include "modules/planecluster.hpp"

#include "cartslam.hpp"
#include "utils/plane.hpp"

namespace cart {
struct vector_stats_t {
    cv::Vec4d plane;
    double d;
    double length;
    double yaw;
    double pitch;
    double yawSin;
    double yawCos;
    double pitchSin;
    double pitchCos;
};

system_data_t SuperPixelPlaneClusterModule::runInternal(System& system, SystemRunData& data) {
    const auto maxLabel = data.getData<contour::label_t>(CARTSLAM_KEY_SUPERPIXELS_MAX_LABEL);
    const auto superpixels = data.getData<cv::cuda::GpuMat>(CARTSLAM_KEY_SUPERPIXELS);
    const auto depth = data.getData<cv::cuda::GpuMat>(CARTSLAM_KEY_DEPTH);

    std::vector<std::vector<cv::Point3d>> superpixelPoints(*maxLabel + 1);

    cv::Mat superpixelsHost;
    cv::Mat depthHost;
    superpixels->download(superpixelsHost);
    depth->download(depthHost);

    for (int y = 0; y < superpixelsHost.rows; y++) {
        for (int x = 0; x < superpixelsHost.cols; x++) {
            const auto label = superpixelsHost.at<contour::label_t>(y, x);
            const auto point = depthHost.at<cv::Point3f>(y, x);
            if (point.z <= 0.0 || point.z > 40.0) {
                continue;
            }

            const cv::Point3d point3d(point.x, point.y, point.z);
            superpixelPoints[label].push_back(point3d);
        }
    }

    std::vector<vector_stats_t> planeStats(*maxLabel + 1);

#pragma omp parallel for
    for (contour::label_t label = 0; label <= *maxLabel; label++) {
        auto inliers = superpixelPoints[label];
        vector_stats_t& stats = planeStats[label];
        if (inliers.size() < 16) {
            stats.plane = cv::Vec4d(0, 0, 0, 0);
            continue;
        }

        stats.plane = util::segmentPlane(inliers);
        if (cv::norm(stats.plane) == 0) {
            continue;
        }

        stats.d = stats.plane[3];
        stats.length = cv::norm(cv::Vec3d(stats.plane[0], stats.plane[1], stats.plane[2]));
        stats.yaw = atan2(stats.plane[1], stats.plane[0]);
        stats.pitch = atan2(stats.plane[2], stats.length);
        stats.yawSin = sin(stats.yaw);
        stats.yawCos = cos(stats.yaw);
        stats.pitchSin = sin(stats.pitch);
        stats.pitchCos = cos(stats.pitch);
    }

    std::vector<std::set<contour::label_t>> labelNeighbours(*maxLabel + 1);

    // Find neighbours of labels
    for (int y = 0; y < superpixelsHost.rows; y++) {
        for (int x = 0; x < superpixelsHost.cols; x++) {
            const auto label = superpixelsHost.at<contour::label_t>(y, x);
            for (int dy = -1; dy <= 1; dy++) {
                for (int dx = -1; dx <= 1; dx++) {
                    const int ny = y + dy;
                    const int nx = x + dx;
                    if (ny < 0 || ny >= superpixelsHost.rows || nx < 0 || nx >= superpixelsHost.cols) {
                        continue;
                    }

                    const auto neighbourLabel = superpixelsHost.at<contour::label_t>(ny, nx);
                    if (neighbourLabel == label) {
                        continue;
                    }

                    labelNeighbours[label].insert(neighbourLabel);
                }
            }
        }
    }

    std::vector<vector_stats_t> planes;
    std::vector<size_t> planeAssignments(*maxLabel + 1, 0);

#pragma omp parallel for
    for (contour::label_t label = 0; label <= *maxLabel; label++) {
        const auto assignment = planeAssignments[label];
        if (assignment != 0) {
            continue;
        }

        const vector_stats_t& stats = planeStats[label];
        if (cv::norm(stats.plane) == 0) {
            continue;
        }

        // Compare similar planes
        std::vector<size_t> similarPlanes = {label};
        std::set<contour::label_t> seenLabels = {label};
        std::set<contour::label_t> neighbourLabels = labelNeighbours[label];

        while (!neighbourLabels.empty()) {
            cart::contour::label_t otherLabel = *neighbourLabels.begin();
            seenLabels.insert(otherLabel);
            neighbourLabels.erase(neighbourLabels.begin());
            const vector_stats_t& otherStats = planeStats[otherLabel];
            if (cv::norm(otherStats.plane) == 0) {
                continue;
            }

            const double yawTrigDiff = std::abs(stats.yawSin - otherStats.yawSin) + std::abs(stats.yawCos - otherStats.yawCos);
            const double pitchTrigDiff = std::abs(stats.pitchSin - otherStats.pitchSin) + std::abs(stats.pitchCos - otherStats.pitchCos);
            const double dDiff = std::abs(stats.d - otherStats.d);

            if (yawTrigDiff < 0.2 && pitchTrigDiff < 0.2 && dDiff < 3) {
                const auto currAssignment = planeAssignments[otherLabel];
                // Check if more similar to current assignment
                if (currAssignment != 0) {
                    const auto currStats = planes[currAssignment - 1];
                    const double currYawTrigDiff = std::abs(currStats.yawSin - otherStats.yawSin) + std::abs(currStats.yawCos - otherStats.yawCos);
                    const double currPitchTrigDiff = std::abs(currStats.pitchSin - otherStats.pitchSin) + std::abs(currStats.pitchCos - otherStats.pitchCos);

                    // Check if improved total
                    if (currYawTrigDiff + currPitchTrigDiff + dDiff < yawTrigDiff + pitchTrigDiff + dDiff) {
                        continue;
                    }
                }

                similarPlanes.push_back(otherLabel);
                for (const auto& neighbourLabel : labelNeighbours[otherLabel]) {
                    if (seenLabels.count(neighbourLabel) > 0) {
                        continue;
                    }

                    neighbourLabels.insert(neighbourLabel);
                }
            }
        }

        // If enough similar planes, merge
        if (similarPlanes.size() < 32) {
            continue;
        }

#pragma omp critical
        {
            planes.push_back(stats);

            // Assign all similar planes
            for (const auto& similarPlane : similarPlanes) {
                planeAssignments[similarPlane] = planes.size();
            }
        }
    }

    plane_fit_data_t planeFitData;
    planeFitData.planeAssignments = planeAssignments;

    for (const auto& plane : planes) {
        planeFitData.planes.push_back(plane.plane);
    }

    return MODULE_RETURN_SHARED(CARTSLAM_KEY_PLANES_EQ, plane_fit_data_t, boost::move(planeFitData));
}
}  // namespace cart