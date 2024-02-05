#include "optflow.hpp"

// Drawing logic borrowed from https://github.com/opencv/opencv_contrib/blob/4.x/modules/cudaoptflow/samples/nvidia_optical_flow.cpp

inline bool isFlowCorrect(cv::Point2f u) {
    return !cvIsNaN(u.x) && !cvIsNaN(u.y) && fabs(u.x) < 1e9 && fabs(u.y) < 1e9;
}

static cv::Vec3b computeColor(float fx, float fy) {
    static bool first = true;

    // relative lengths of color transitions:
    // these are chosen based on perceptual similarity
    // (e.g. one can distinguish more shades between red and yellow
    //  than between yellow and green)
    const int RY = 15;
    const int YG = 6;
    const int GC = 4;
    const int CB = 11;
    const int BM = 13;
    const int MR = 6;
    const int NCOLS = RY + YG + GC + CB + BM + MR;
    static cv::Vec3i colorWheel[NCOLS];

    if (first) {
        int k = 0;

        for (int i = 0; i < RY; ++i, ++k)
            colorWheel[k] = cv::Vec3i(255, 255 * i / RY, 0);

        for (int i = 0; i < YG; ++i, ++k)
            colorWheel[k] = cv::Vec3i(255 - 255 * i / YG, 255, 0);

        for (int i = 0; i < GC; ++i, ++k)
            colorWheel[k] = cv::Vec3i(0, 255, 255 * i / GC);

        for (int i = 0; i < CB; ++i, ++k)
            colorWheel[k] = cv::Vec3i(0, 255 - 255 * i / CB, 255);

        for (int i = 0; i < BM; ++i, ++k)
            colorWheel[k] = cv::Vec3i(255 * i / BM, 0, 255);

        for (int i = 0; i < MR; ++i, ++k)
            colorWheel[k] = cv::Vec3i(255, 0, 255 - 255 * i / MR);

        first = false;
    }

    const float rad = sqrt(fx * fx + fy * fy);
    const float a = atan2(-fy, -fx) / (float)CV_PI;

    const float fk = (a + 1.0f) / 2.0f * (NCOLS - 1);
    const int k0 = static_cast<int>(fk);
    const int k1 = (k0 + 1) % NCOLS;
    const float f = fk - k0;

    cv::Vec3b pix;

    for (int b = 0; b < 3; b++) {
        const float col0 = colorWheel[k0][b] / 255.0f;
        const float col1 = colorWheel[k1][b] / 255.0f;

        float col = (1 - f) * col0 + f * col1;

        if (rad <= 1)
            col = 1 - rad * (1 - col);  // increase saturation with radius
        else
            col *= .75;  // out of range

        pix[2 - b] = static_cast<uchar>(255.0 * col);
    }

    return pix;
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
                dst.at<cv::Vec3b>(y, x) = computeColor(u.x / maxrad, u.y / maxrad);
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

ImageOpticalFlow detectOpticalFlow(const CARTSLAM_IMAGE_TYPE input, const CARTSLAM_IMAGE_TYPE reference, cv::Ptr<cv::cuda::NvidiaOpticalFlow_2_0> &opticalFlow) {
    cv::cuda::GpuMat flow;
    cv::cuda::GpuMat cost;

    opticalFlow->calc(input, reference, flow, cv::cuda::Stream::Null(), cv::noArray(), cost);

    return ImageOpticalFlow(flow, cost);
};

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

}  // namespace cart