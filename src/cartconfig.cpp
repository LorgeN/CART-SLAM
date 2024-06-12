#include "cartconfig.hpp"

#ifdef CARTSLAM_JSON

#include <boost/shared_ptr.hpp>
#include <fstream>
#include <nlohmann/json.hpp>
#include <string>

#include "cartslam.hpp"
#include "datasource.hpp"
#include "modules/depth.hpp"
#include "modules/disparity.hpp"
#include "modules/features.hpp"
#include "modules/optflow.hpp"
#include "modules/planefit.hpp"
#include "modules/planeseg.hpp"
#include "modules/superpixels.hpp"
#include "sources/kitti.hpp"
#include "sources/zed.hpp"
#include "utils/path.hpp"

#define CART_CONFIG_KEY_DATA_SOURCE "data_source"
#define CART_CONFIG_KEY_MODULES "modules"

#endif

namespace cart::config {

#ifdef CARTSLAM_JSON
constexpr uint32_t hash(const std::string_view data) noexcept {
    uint32_t hash = 5385;
    for (const auto &e : data) hash = ((hash << 5) + hash) + e;
    return hash;
}

template <typename T>
inline const T get(const nlohmann::json &data, const std::string &key, const T &defaultValue) {
    if (data.find(key) == data.end()) {
        return defaultValue;
    }

    return data[key].get<T>();
}

template <typename T>
inline const T get(const nlohmann::json &data, const std::string &key) {
    if (data.find(key) == data.end()) {
        throw std::runtime_error("Key " + key + " not found.");
    }

    return data[key].get<T>();
}

inline boost::shared_ptr<PlaneParameterProvider> readParameterProvider(const nlohmann::json &data) {
    if (data.find("type") == data.end()) {
        throw std::runtime_error("Parameter provider type not found.");
    }

    const std::string providerType = data["type"].get<std::string>();

    switch (hash(providerType)) {
        case hash("static"): {
            auto horizontalRange = std::make_pair(get<int>(data, "horizontal_range_min"), get<int>(data, "horizontal_range_max"));
            auto verticalRange = std::make_pair(get<int>(data, "vertical_range_min"), get<int>(data, "vertical_range_max"));

            auto horizontalCenter = (horizontalRange.first + horizontalRange.second) / 2;
            auto verticalCenter = (verticalRange.first + verticalRange.second) / 2;

            return boost::make_shared<StaticPlaneParameterProvider>(horizontalCenter, verticalCenter, horizontalRange, verticalRange);
        }
        case hash("histogram_peak"):
            return boost::make_shared<HistogramPeakPlaneParameterProvider>();
        default:
            throw std::runtime_error("Unknown parameter provider type.");
    }

    return nullptr;
}

boost::shared_ptr<cart::System> readSystemConfig(const std::string path) {
    std::ifstream file(cart::util::resolvePath(path));

    if (!file.is_open()) {
        throw std::runtime_error("Could not open file " + path + ": " + strerror(errno));
    }

    nlohmann::json data = nlohmann::json::parse(file);

    if (data.find(CART_CONFIG_KEY_DATA_SOURCE) == data.end()) {
        throw std::runtime_error("Data source not found in configuration file.");
    }

    if (data.find(CART_CONFIG_KEY_MODULES) == data.end()) {
        throw std::runtime_error("Modules not found in configuration file.");
    }

    auto dataSourceConfig = data[CART_CONFIG_KEY_DATA_SOURCE];

    if (!dataSourceConfig.is_object()) {
        throw std::runtime_error("Data source configuration is not an object.");
    }

    boost::shared_ptr<cart::DataSource> dataSource;
    const std::string sourcePath = dataSourceConfig["path"].get<std::string>();
    const std::string dataSourceType = dataSourceConfig["type"].get<std::string>();
    switch (hash(dataSourceType)) {
        case hash("zed"): {
            const bool includeDisparity = get(dataSourceConfig, "include_disparity", false);
            dataSource = boost::make_shared<cart::sources::ZEDDataSource>(sourcePath, includeDisparity);
        } break;
        case hash("kitti"): {
            const int kittiSeq = get(dataSourceConfig, "sequence", 0);
            dataSource = boost::make_shared<cart::sources::KITTIDataSource>(sourcePath, kittiSeq);
        } break;
        default:
            throw std::runtime_error("Unknown data source type.");
    }

    auto system = boost::make_shared<cart::System>(dataSource);

    auto modulesConfig = data[CART_CONFIG_KEY_MODULES];

    if (!modulesConfig.is_array()) {
        throw std::runtime_error("Modules configuration is not an array.");
    }

    for (const auto &moduleConfig : modulesConfig) {
        if (!moduleConfig.is_object()) {
            throw std::runtime_error("Module configuration is not an object.");
        }

        const std::string moduleType = moduleConfig["type"].get<std::string>();

        switch (hash(moduleType)) {
            case hash("superpixels"):
                system->addModule<cart::SuperPixelModule>(
                    dataSource->getImageSize(),
                    get(moduleConfig, "initial_iterations", 18),
                    get(moduleConfig, "iterations", 6),
                    get(moduleConfig, "block_size", 12),
                    get(moduleConfig, "reset_iterations", 64),
                    get(moduleConfig, "direct_clique_cost", 0.25),
                    get(moduleConfig, "diagonal_clique_cost", 0.25 / sqrt(2)),
                    get(moduleConfig, "compactness_weight", 0.05),
                    get(moduleConfig, "progressive_compactness_cost", 1.0),
                    get(moduleConfig, "image_weight", 1.0),
                    get(moduleConfig, "disparity_weight", 1.25));
                break;
            case hash("superpixels_visualization"):
                system->addModule<cart::SuperPixelVisualizationModule>();
                break;
            case hash("depth"):
                system->addModule<cart::DepthModule>();
                break;
            case hash("depth_visualization"):
                system->addModule<cart::DepthVisualizationModule>();
                break;
            case hash("disparity"):
                system->addModule<cart::ImageDisparityModule>(
                    get(moduleConfig, "min_disparity", 0),
                    get(moduleConfig, "num_disparities", 256),
                    get(moduleConfig, "block_size", 3),
                    get(moduleConfig, "smoothing_radius", -1),
                    get(moduleConfig, "smoothing_iterations", 5));
                break;
            case hash("zed_disparity"):
                system->addModule<cart::ZEDImageDisparityModule>(
                    get(moduleConfig, "smoothing_radius", -1),
                    get(moduleConfig, "smoothing_iterations", 5));
                break;
            case hash("disparity_derivative"):
                system->addModule<cart::ImageDisparityDerivativeModule>();
                break;
            case hash("features"): {
                const std::string featureType = get<std::string>(moduleConfig, "feature_type", "orb");
                FeatureDetector detector;
                switch (hash(featureType)) {
                    case hash("orb"):
                        detector = cart::detectOrbFeatures;
                        break;
                    default:
                        throw std::runtime_error("Unknown feature type.");
                }

                system->addModule<cart::ImageFeatureDetectorModule>(detector);
            } break;
            case hash("features_visualization"):
                system->addModule<cart::ImageFeatureVisualizationModule>();
                break;
            case hash("optflow"):
                system->addModule<cart::ImageOpticalFlowModule>(dataSource->getImageSize());
                break;
            case hash("optflow_visualization"):
                system->addModule<cart::ImageOpticalFlowVisualizationModule>(dataSource->getImageSize(), get(moduleConfig, "points", 10));
                break;
            case hash("planefit"):
                system->addModule<cart::SuperPixelPlaneFitModule>();
                break;
            case hash("disparity_planeseg"): {
                const auto parameterProvider = readParameterProvider(moduleConfig["parameter_provider"]);
                system->addModule<cart::DisparityPlaneSegmentationModule>(
                    parameterProvider,
                    get(moduleConfig, "update_interval", 30),
                    get(moduleConfig, "reset_interval", 10),
                    get(moduleConfig, "use_temporal_smoothing", false),
                    get(moduleConfig, "temporal_smoothing_distance", CARTSLAM_PLANE_TEMPORAL_DISTANCE_DEFAULT));
            } break;
            case hash("disparity_planeseg_visualization"):
                system->addModule<cart::DisparityPlaneSegmentationVisualizationModule>(
                    get(moduleConfig, "show_histogram", true),
                    get(moduleConfig, "show_unsmoothed", true));
                break;
            case hash("superpixel_disparity_planeseg"): {
                const auto parameterProvider = readParameterProvider(moduleConfig["parameter_provider"]);
                system->addModule<cart::SuperPixelDisparityPlaneSegmentationModule>(
                    parameterProvider,
                    get(moduleConfig, "update_interval", 30),
                    get(moduleConfig, "reset_interval", 10),
                    get(moduleConfig, "use_temporal_smoothing", false),
                    get(moduleConfig, "temporal_smoothing_distance", CARTSLAM_PLANE_TEMPORAL_DISTANCE_DEFAULT));
            } break;
            case hash("bev_planeseg_visualization"):
                system->addModule<cart::PlaneSegmentationBEVVisualizationModule>();
                break;
        }
    }

    return system;
}
#else
boost::shared_ptr<cart::System> readSystemConfig(const std::string path) {
    throw std::runtime_error("JSON configuration not available.");
}
#endif
}  // namespace cart::config