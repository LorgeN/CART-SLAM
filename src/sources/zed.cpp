#include "sources/zed.hpp"

#ifdef CARTSLAM_ZED
#include <log4cxx/logger.h>

#include <opencv2/cudaimgproc.hpp>

#include "utils/path.hpp"

namespace cart::sources {
ZEDDataSource::ZEDDataSource(std::string path, bool extractDepthMeasure, cv::Size imageSize) : DataSource(imageSize), path(path), extractDisparityMeasure(extractDepthMeasure) {
    this->camera = boost::make_shared<sl::Camera>();

    sl::InitParameters params;

#ifdef CARTSLAM_ZED_REALTIME_MODE
    params.svo_real_time_mode = true;
#endif

    params.input.setFromSVOFile(cart::util::resolvePath(path).c_str());
    params.coordinate_units = sl::UNIT::METER;
    params.depth_mode = sl::DEPTH_MODE::QUALITY;
    params.depth_maximum_distance = 40.0f;

#ifdef CARTSLAM_DEBUG
    params.sdk_verbose = true;
#endif

    sl::ERROR_CODE err = this->camera->open(params);
    if (err != sl::ERROR_CODE::SUCCESS) {
        throw std::runtime_error("Failed to open ZED camera");
    }

    if (imageSize.width == 0 || imageSize.height == 0) {
        this->imageSize = cv::Size(1280, 720);
    }

    auto config = this->camera->getCameraInformation().camera_configuration;
    auto info = config.calibration_parameters;

    this->intrinsics.Q = cv::Mat::eye(4, 4, CV_32F);

    float scalingX = static_cast<float>(this->imageSize.width) / config.resolution.width;
    float scalingY = static_cast<float>(this->imageSize.height) / config.resolution.height;

    float left_cx = info.left_cam.cx * scalingX;
    float left_cy = info.left_cam.cy * scalingY;
    float left_fx = info.left_cam.fx * scalingX;
    float right_cx = info.right_cam.cx * scalingX;
    float baseline = -info.getCameraBaseline();

    this->intrinsics.Q.at<float>(0, 3) = -left_cx;
    this->intrinsics.Q.at<float>(1, 3) = -left_cy;
    this->intrinsics.Q.at<float>(2, 2) = 0;
    this->intrinsics.Q.at<float>(2, 3) = left_fx;
    this->intrinsics.Q.at<float>(3, 2) = -1.0 / baseline;
    this->intrinsics.Q.at<float>(3, 3) = ((left_cx - right_cx) / baseline);
}

ZEDDataSource::~ZEDDataSource() {
    this->camera->close();
}

bool ZEDDataSource::isNextReady() {
    if (!this->hasGrabbed || this->grabResult != sl::ERROR_CODE::SUCCESS) {
        this->grabResult = this->camera->grab();
        this->hasGrabbed = true;
    }

    return this->grabResult == sl::ERROR_CODE::SUCCESS;
}

bool ZEDDataSource::isFinished() {
    if (!this->hasGrabbed) {
        this->grabResult = this->camera->grab();
        this->hasGrabbed = true;
    }

    return this->grabResult == sl::ERROR_CODE::END_OF_SVOFILE_REACHED;
}

DataElementType ZEDDataSource::getProvidedType() {
    return DataElementType::STEREO;
}

boost::shared_ptr<DataElement> ZEDDataSource::getNextInternal(log4cxx::LoggerPtr logger, cv::cuda::Stream& stream) {
    if (!this->hasGrabbed) {
        this->grabResult = this->camera->grab();
        if (this->grabResult != sl::ERROR_CODE::SUCCESS) {
            LOG4CXX_ERROR(logger, "Failed to grab frame");
            return nullptr;
        }
    }

    this->hasGrabbed = false;

    const sl::Resolution res(this->imageSize.width, this->imageSize.height);

    sl::Mat left, right;
    sl::ERROR_CODE leftRes = this->camera->retrieveImage(left, sl::VIEW::LEFT, sl::MEM::GPU, res);
    if (leftRes != sl::ERROR_CODE::SUCCESS) {
        LOG4CXX_ERROR(logger, "Failed to retrieve left image");
        return nullptr;
    }

    sl::ERROR_CODE rightRes = this->camera->retrieveImage(right, sl::VIEW::RIGHT, sl::MEM::GPU, res);
    if (rightRes != sl::ERROR_CODE::SUCCESS) {
        LOG4CXX_ERROR(logger, "Failed to retrieve right image");
        return nullptr;
    }

    // Clone these because the ZED SDK has terrible memory management, and it was causing issues by
    // randomly freeing the memory.
    auto element = boost::make_shared<ZEDDataElement>(slMat2cvGpuMat(left).clone(), slMat2cvGpuMat(right).clone());

#ifdef CARTSLAM_IMAGE_MAKE_GRAYSCALE
    cv::cuda::cvtColor(element->left, element->left, cv::COLOR_BGRA2GRAY, 0, stream);
    cv::cuda::cvtColor(element->right, element->right, cv::COLOR_BGRA2GRAY, 0, stream);
#else
    cv::cuda::cvtColor(element->left, element->left, cv::COLOR_BGRA2BGR, 0, stream);
    cv::cuda::cvtColor(element->right, element->right, cv::COLOR_BGRA2BGR, 0, stream);
#endif

    if (this->extractDisparityMeasure) {
        sl::Mat disparityMeasure;
        sl::ERROR_CODE dispRes = this->camera->retrieveMeasure(disparityMeasure, sl::MEASURE::DISPARITY, sl::MEM::GPU, res);
        if (dispRes != sl::ERROR_CODE::SUCCESS) {
            LOG4CXX_ERROR(logger, "Failed to retrieve disparity measure");
            return nullptr;
        }

        element->disparityMeasure = slMat2cvGpuMat(disparityMeasure).clone();
    }

    return element;
}

int slMatType2cvType(sl::MAT_TYPE type) {
    switch (type) {
        case sl::MAT_TYPE::F32_C1:
            return CV_32FC1;
        case sl::MAT_TYPE::F32_C2:
            return CV_32FC2;
        case sl::MAT_TYPE::F32_C3:
            return CV_32FC3;
        case sl::MAT_TYPE::F32_C4:
            return CV_32FC4;
        case sl::MAT_TYPE::U8_C1:
            return CV_8UC1;
        case sl::MAT_TYPE::U8_C2:
            return CV_8UC2;
        case sl::MAT_TYPE::U8_C3:
            return CV_8UC3;
        case sl::MAT_TYPE::U8_C4:
            return CV_8UC4;
        case sl::MAT_TYPE::U16_C1:
            return CV_16UC1;
        default:
            return -1;
    }
}

cv::cuda::GpuMat slMat2cvGpuMat(sl::Mat& input) {
    int cv_type = slMatType2cvType(input.getDataType());
    return cv::cuda::GpuMat(input.getHeight(), input.getWidth(), cv_type, input.getPtr<sl::uchar1>(sl::MEM::GPU), input.getStepBytes(sl::MEM::GPU));
}

cv::Mat slMat2cvMat(sl::Mat& input) {
    int cv_type = slMatType2cvType(input.getDataType());
    return cv::Mat(input.getHeight(), input.getWidth(), cv_type, input.getPtr<sl::uchar1>(sl::MEM::CPU), input.getStepBytes(sl::MEM::CPU));
}
}  // namespace cart::sources
#endif