#include "modules/optflow.hpp"

#include "cartslam.hpp"
#include "utils/colors.hpp"
#include "utils/modules.hpp"

// Drawing logic borrowed from https://github.com/opencv/opencv_contrib/blob/4.x/modules/cudaoptflow/samples/nvidia_optical_flow.cpp

inline bool isFlowCorrect(cv::Point2f u) {
    return !cvIsNaN(u.x) && !cvIsNaN(u.y) && fabs(u.x) < 1e9 && fabs(u.y) < 1e9;
}

// TODO: CUDA-ify this
void drawOpticalFlowInternal(const cv::Mat_<float> &flowx, const cv::Mat_<float> &flowy, cv::Mat &dst, float maxmotion = -1) {
    dst.create(flowx.size(), CV_8UC3);
    dst.setTo(cv::Scalar::all(0));

    // determine motion range:
    float maxrad = maxmotion;

    if (maxmotion <= 0) {
        maxrad = 1;
#pragma omp parallel for reduction(max : maxrad)
        for (int y = 0; y < flowx.rows; ++y) {
            for (int x = 0; x < flowx.cols; ++x) {
                cv::Point2f u(flowx(y, x), flowy(y, x));

                if (!isFlowCorrect(u))
                    continue;

                float value = u.x * u.x + u.y * u.y;
                maxrad = maxrad > value ? maxrad : value;
            }
        }
    }

    maxrad = sqrt(maxrad);

#pragma omp parallel for
    for (int y = 0; y < flowx.rows; ++y) {
        for (int x = 0; x < flowx.cols; ++x) {
            cv::Point2f u(flowx(y, x), flowy(y, x));

            if (isFlowCorrect(u)) {
                dst.at<cv::Vec3b>(y, x) = cart::util::computeColor(u.x / maxrad, u.y / maxrad);
            }
        }
    }
}

namespace cart {

ImageOpticalFlowModule::ImageOpticalFlowModule() : SyncWrapperSystemModule("ImageOpticalFlow") {
    this->providesData.push_back(CARTSLAM_KEY_OPTFLOW);

    this->opticalFlow = createOpticalFlow(this->stream);
}

cv::Ptr<cv::cuda::NvidiaOpticalFlow_2_0> createOpticalFlow(cv::cuda::Stream &stream) {
    return cv::cuda::NvidiaOpticalFlow_2_0::create(
        cv::Size(CARTSLAM_IMAGE_RES_X, CARTSLAM_IMAGE_RES_Y),
        cv::cuda::NvidiaOpticalFlow_2_0::NV_OF_PERF_LEVEL_MEDIUM,
        cv::cuda::NvidiaOpticalFlow_2_0::NV_OF_OUTPUT_VECTOR_GRID_SIZE_1,
        cv::cuda::NvidiaOpticalFlow_2_0::NV_OF_HINT_VECTOR_GRID_SIZE_1,
        true,
        false,
        false,
        0,
        stream,
        stream);
}

cv::Mat drawOpticalFlow(const cv::cuda::GpuMat &imageFlow, cv::Ptr<cv::cuda::NvidiaOpticalFlow_2_0> opticalFlow, cv::Mat &cpuFlow, cv::cuda::Stream &stream) {
    cv::cuda::GpuMat floatFlow;

    opticalFlow->convertToFloat(imageFlow, floatFlow);

    floatFlow.download(cpuFlow, stream);

    stream.waitForCompletion();

    cv::Mat flowX, flowY, flowImage;
    cv::Mat flowPlanes[2] = {flowX, flowY};

    cv::split(cpuFlow, flowPlanes);

    flowX = flowPlanes[0];
    flowY = flowPlanes[1];

    drawOpticalFlowInternal(flowX, flowY, flowImage, 10);

    return flowImage;
}

void ImageOpticalFlowModule::detectOpticalFlow(const image_t input,
                                               const image_t reference,
                                               cv::Ptr<cv::cuda::NvidiaOpticalFlow_2_0> opticalFlow,
                                               cv::cuda::Stream &stream,
                                               cv::cuda::GpuMat &flow) {
#ifndef CARTSLAM_IMAGE_MAKE_GRAYSCALE
    cv::cuda::GpuMat inputProc;
    cv::cuda::GpuMat referenceProc;

    cv::cuda::cvtColor(input, inputProc, cv::COLOR_BGR2GRAY, 0, stream);
    cv::cuda::cvtColor(reference, referenceProc, cv::COLOR_BGR2GRAY, 0, stream);
#endif

    {
        boost::unique_lock<boost::shared_mutex> lock(this->flowMutex);
#ifndef CARTSLAM_IMAGE_MAKE_GRAYSCALE
        opticalFlow->calc(inputProc, referenceProc, flow);
#else
        opticalFlow->calc(input, reference, flow);
#endif
        stream.waitForCompletion();
    }
}

system_data_t ImageOpticalFlowModule::runInternal(System &system, SystemRunData &data) {
    if (data.id <= 1) {  // First run, no previous data
        return MODULE_RETURN(CARTSLAM_KEY_OPTFLOW, boost::shared_ptr<void>());
    }

    boost::shared_ptr<SystemRunData> previousRun = data.getRelativeRun(-1);

    auto referenceCurrent = cart::getReferenceImage(data.dataElement);
    auto referencePrevious = cart::getReferenceImage(previousRun->dataElement);

    cv::cuda::GpuMat result;
    this->detectOpticalFlow(referenceCurrent, referencePrevious, this->opticalFlow, this->stream, result);

    return MODULE_RETURN_SHARED(CARTSLAM_KEY_OPTFLOW, cv::cuda::GpuMat, boost::move(result));
}

bool ImageOpticalFlowVisualizationModule::updateImage(System &system, SystemRunData &data, cv::Mat &image) {
    if (data.id <= 1) {  // First run, no previous data
        LOG4CXX_DEBUG(this->logger, "No previous data, skipping optical flow visualization");
        return false;
    }

    boost::shared_ptr<cv::cuda::GpuMat> flow = data.getData<cv::cuda::GpuMat>(CARTSLAM_KEY_OPTFLOW);

    cv::cuda::Stream stream;
    cv::Ptr<cv::cuda::NvidiaOpticalFlow_2_0> opticalFlow = createOpticalFlow(stream);

    cv::Mat cpuFlow;

    cv::Mat flowImageLeft = drawOpticalFlow(*flow, opticalFlow, cpuFlow, stream);

    cv::Mat flowX, flowY, flowImage;
    cv::Mat flowPlanes[2] = {flowX, flowY};

    cv::split(cpuFlow, flowPlanes);
    flowX = flowPlanes[0];
    flowY = flowPlanes[1];

    cv::Mat images[2];
    getReferenceImage(data.dataElement).download(images[0], stream);
    getReferenceImage(data.getRelativeRun(-1)->dataElement).download(images[1], stream);

    cv::vconcat(images[0], images[1], image);
    cv::vconcat(image, flowImageLeft, image);

    // Draw arrows on the image

    cv::Point2i prevImageOffset = cv::Point2i(0, images[0].rows);

    for (const auto &point : this->visualizationPoints) {
        cv::Point2f flowPoint(flowX.at<float>(point), flowY.at<float>(point));
        cv::arrowedLine(image, point + prevImageOffset, point - cv::Point2i(flowPoint.x, flowPoint.y), cv::Scalar(0, 255, 0), 1, cv::LINE_AA, 0, 0.05);
    }

    return true;
}

}  // namespace cart