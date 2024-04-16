#include "sources/zed.hpp"

#ifdef CARTSLAM_ZED
#include <log4cxx/logger.h>

#include <opencv2/cudaimgproc.hpp>

namespace cart::sources {
const sl::Resolution RES = sl::Resolution(CARTSLAM_IMAGE_RES_X, CARTSLAM_IMAGE_RES_Y);

ZEDDataSource::ZEDDataSource(std::string path, bool extractDepthMeasure) : path(path), extractDisparityMeasure(extractDepthMeasure) {
    this->camera = boost::make_shared<sl::Camera>();

    sl::InitParameters params;

    // Enable for real time testing. This will skip frames if the processing is too slow, and will throw errors if
    // the processing is too fast.
    // params.svo_real_time_mode = true;

    params.input.setFromSVOFile(path.c_str());
    params.coordinate_units = sl::UNIT::METER;
    params.depth_mode = sl::DEPTH_MODE::QUALITY;

#ifdef CARTSLAM_DEBUG
    params.sdk_verbose = true;
#endif

    sl::ERROR_CODE err = this->camera->open(params);
    if (err != sl::ERROR_CODE::SUCCESS) {
        throw std::runtime_error("Failed to open ZED camera");
    }
}

ZEDDataSource::~ZEDDataSource() {
    this->camera->close();
}

bool ZEDDataSource::hasNext() {
    if (!this->hasGrabbed) {
        this->grabResult = this->camera->grab();
        this->hasGrabbed = true;
    }

    return this->grabResult == sl::ERROR_CODE::SUCCESS;
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

    sl::Mat left, right;
    sl::ERROR_CODE leftRes = this->camera->retrieveImage(left, sl::VIEW::LEFT, sl::MEM::GPU, RES);
    if (leftRes != sl::ERROR_CODE::SUCCESS) {
        LOG4CXX_ERROR(logger, "Failed to retrieve left image");
        return nullptr;
    }

    sl::ERROR_CODE rightRes = this->camera->retrieveImage(right, sl::VIEW::RIGHT, sl::MEM::GPU, RES);
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
        sl::ERROR_CODE dispRes = this->camera->retrieveMeasure(disparityMeasure, sl::MEASURE::DISPARITY, sl::MEM::GPU, RES);
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