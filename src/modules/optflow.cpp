#include "modules/optflow.hpp"

#include "utils/colors.hpp"

// Drawing logic borrowed from https://github.com/opencv/opencv_contrib/blob/4.x/modules/cudaoptflow/samples/nvidia_optical_flow.cpp

inline bool isFlowCorrect(cv::Point2f u) {
    return !cvIsNaN(u.x) && !cvIsNaN(u.y) && fabs(u.x) < 1e9 && fabs(u.y) < 1e9;
}

void drawOpticalFlowInternal(const cv::Mat_<float> &flowx, const cv::Mat_<float> &flowy, cv::Mat &dst, float maxmotion = -1) {
    dst.create(flowx.size(), CV_8UC3);
    dst.setTo(cv::Scalar::all(0));

    // determine motion range:
    float maxrad = maxmotion;

    if (maxmotion <= 0) {
        maxrad = 1;
        for (int y = 0; y < flowx.rows; ++y) {
            for (int x = 0; x < flowx.cols; ++x) {
                cv::Point2f u(flowx(y, x), flowy(y, x));

                if (!isFlowCorrect(u))
                    continue;

                maxrad = cv::max(maxrad, sqrt(u.x * u.x + u.y * u.y));
            }
        }
    }

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

cv::Ptr<cv::cuda::NvidiaOpticalFlow_2_0> createOpticalFlow(cv::cuda::Stream &stream) {
    return cv::cuda::NvidiaOpticalFlow_2_0::create(
        cv::Size(CARTSLAM_IMAGE_RES_X, CARTSLAM_IMAGE_RES_Y),
        cv::cuda::NvidiaOpticalFlow_2_0::NV_OF_PERF_LEVEL_MEDIUM,
        cv::cuda::NvidiaOpticalFlow_2_0::NV_OF_OUTPUT_VECTOR_GRID_SIZE_1,
        cv::cuda::NvidiaOpticalFlow_2_0::NV_OF_HINT_VECTOR_GRID_SIZE_1,
        false,
        false,
        true,
        0,
        stream,
        stream);
}

ImageOpticalFlow detectOpticalFlow(const CARTSLAM_IMAGE_TYPE input, const CARTSLAM_IMAGE_TYPE reference, const cv::Ptr<cv::cuda::NvidiaOpticalFlow_2_0> &opticalFlow) {
    cv::cuda::GpuMat flow;
    cv::cuda::GpuMat cost;

    opticalFlow->calc(input, reference, flow, cv::cuda::Stream::Null(), cv::noArray(), cost);

    return ImageOpticalFlow(flow, cost);
}

cv::Mat drawOpticalFlow(const ImageOpticalFlow &imageFlow, cv::Ptr<cv::cuda::NvidiaOpticalFlow_2_0> &opticalFlow, cv::cuda::Stream &stream) {
    cv::cuda::GpuMat floatFlow;

    opticalFlow->convertToFloat(imageFlow.flow, floatFlow);

    cv::Mat cpuFlow;
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

MODULE_RETURN_VALUE ImageOpticalFlowModule::runInternal(System &system, SystemRunData &data) {
    if (data.id == 0) {  // First run, no previous data
        return MODULE_RETURN_VALUE_PAIR(CARTSLAM_KEY_OPTFLOW, NULL);
    }

    cv::cuda::Stream stream;
    cv::Ptr<cv::cuda::NvidiaOpticalFlow_2_0> flow = createOpticalFlow(stream);

    ImageOpticalFlowVisitor visitor(data, flow, this->logger);

    auto result = visitor(data.dataElement);
    return MODULE_RETURN(CARTSLAM_KEY_OPTFLOW, boost::shared_ptr<void>(boost::move(result)));
}

void *ImageOpticalFlowVisitor::visitStereo(boost::shared_ptr<StereoDataElement> element) {
    boost::shared_ptr<SystemRunData> previousRun = this->data.getRelativeRun(-1);
    // The previous type of element should be the same as the current one
    boost::shared_ptr<StereoDataElement> previousElement = boost::static_pointer_cast<StereoDataElement>(previousRun->dataElement);

    ImageOpticalFlow flowLeft = detectOpticalFlow(element->left, previousElement->left, this->flow);
    ImageOpticalFlow flowRight = detectOpticalFlow(element->right, previousElement->right, this->flow);

    return new std::pair<ImageOpticalFlow, ImageOpticalFlow>(flowLeft, flowRight);
}

boost::future<MODULE_RETURN_VALUE> ImageOpticalFlowVisualizationModule::run(System &system, SystemRunData &data) {
    auto promise = boost::make_shared<boost::promise<MODULE_RETURN_VALUE>>();

    boost::asio::post(system.threadPool, [this, promise, &system, &data]() {
        if (data.id == 0) {  // First run, no previous data
            promise->set_value(MODULE_NO_RETURN_VALUE);
            LOG4CXX_DEBUG(this->logger, "No previous data, skipping optical flow visualization");
            return;
        }

        // TODO: Support for none-stereo images
        boost::shared_ptr<std::pair<ImageOpticalFlow, ImageOpticalFlow>> flows = data.getData<std::pair<ImageOpticalFlow, ImageOpticalFlow>>(CARTSLAM_KEY_OPTFLOW);

        cv::cuda::Stream stream;
        cv::Ptr<cv::cuda::NvidiaOpticalFlow_2_0> opticalFlow = createOpticalFlow(stream);

        cv::Mat flowImageLeft = drawOpticalFlow(flows->first, opticalFlow, stream);
        cv::Mat flowImageRight = drawOpticalFlow(flows->second, opticalFlow, stream);

        cv::Mat images[4];
        cv::Mat flowImages[2] = {flowImageLeft, flowImageRight};

        boost::shared_ptr<StereoDataElement> element1 = boost::static_pointer_cast<StereoDataElement>(data.dataElement);
        boost::shared_ptr<StereoDataElement> element2 = boost::static_pointer_cast<StereoDataElement>(data.getRelativeRun(-1)->dataElement);

        element1->left.download(images[0], stream);
        element1->right.download(images[1], stream);
        element2->left.download(images[2], stream);
        element2->right.download(images[3], stream);

        stream.waitForCompletion();

        for (int i = 0; i < 4; i++) {
            cv::cvtColor(images[i], images[i], cv::COLOR_GRAY2BGR);
        }

        cv::Mat prevElemConcat;
        cv::Mat currElemConcat;
        cv::Mat flowConcat;
        cv::Mat concatRes;

        cv::hconcat(images[0], images[1], prevElemConcat);
        cv::hconcat(images[2], images[3], currElemConcat);
        cv::hconcat(flowImages[0], flowImages[1], flowConcat);

        cv::vconcat(prevElemConcat, currElemConcat, concatRes);
        cv::vconcat(concatRes, flowConcat, concatRes);

        LOG4CXX_DEBUG(this->logger, "Displaying optical flow visualization");
        this->imageThread->setImageIfLater(concatRes, data.id);
        promise->set_value(MODULE_NO_RETURN_VALUE);
    });

    return promise->get_future();
}

}  // namespace cart