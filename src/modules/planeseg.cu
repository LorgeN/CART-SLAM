#include <opencv2/ximgproc/disparity_filter.hpp>

#include "modules/disparity.hpp"
#include "modules/planeseg.hpp"
#include "timing.hpp"
#include "utils/cuda.cuh"

#define LOW_PASS_FILTER_SIZE 5
#define LOW_PASS_FILTER_PADDING (LOW_PASS_FILTER_SIZE / 2)

#define THREADS_PER_BLOCK_X 16
#define THREADS_PER_BLOCK_Y 16
#define X_BATCH 4
#define Y_BATCH 4
#define SHARED_SIZE ((X_BATCH * THREADS_PER_BLOCK_X) * (Y_BATCH * (LOW_PASS_FILTER_PADDING * 2 + THREADS_PER_BLOCK_Y)))

#define LOCAL_INDEX(x, y) SHARED_INDEX(sharedPixelX + x, sharedPixelY + y, 0, LOW_PASS_FILTER_PADDING, sharedRowStep)

#define DISPARITY_SCALING (1.0 / 16.0)

#define ROUND_TO_INT(x) static_cast<int32_t>(round(x))

typedef int16_t derivative_t;

__global__ void calculateDerivatives(cv::cuda::PtrStepSz<cart::disparity_t> disparity, cv::cuda::PtrStepSz<derivative_t> output, int width, int height) {
    __shared__ cart::disparity_t sharedDisparity[SHARED_SIZE];

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int sharedPixelX = threadIdx.x * X_BATCH;
    int sharedPixelY = threadIdx.y * Y_BATCH;

    int pixelX = x * X_BATCH;
    int pixelY = y * Y_BATCH;

    size_t outputRowStep = output.step / sizeof(derivative_t);
    size_t sharedRowStep = X_BATCH * blockDim.x;

    copyToShared<cart::disparity_t>(sharedDisparity, disparity, X_BATCH, Y_BATCH, LOW_PASS_FILTER_PADDING, 0, width, height);

    __syncthreads();

    // Perform vertical low pass filter
    for (int j = 0; j < X_BATCH; j++) {
        // Sliding window sum
        derivative_t sum = 0;

        cart::disparity_t previous[LOW_PASS_FILTER_PADDING] = {0};
        size_t previousIndex = 0;

        for (int i = -LOW_PASS_FILTER_PADDING; i < LOW_PASS_FILTER_PADDING; i++) {
            cart::disparity_t value = sharedDisparity[LOCAL_INDEX(j, i)];
            sum += value;

            if (i < 0) {
                previous[previousIndex] = value;
                previousIndex++;
            }
        }

        previousIndex = 0;

        for (int i = 0; i < Y_BATCH; i++) {
            sum += sharedDisparity[LOCAL_INDEX(j, i + LOW_PASS_FILTER_PADDING)];

            cart::disparity_t current = sharedDisparity[LOCAL_INDEX(j, i)];

            sharedDisparity[LOCAL_INDEX(j, i)] = sum / LOW_PASS_FILTER_SIZE;

            sum -= previous[previousIndex];
            previous[previousIndex] = current;
            previousIndex = (previousIndex + 1) % LOW_PASS_FILTER_PADDING;
        }
    }

    __syncthreads();

    // Calculate vertical derivatives
    for (int i = 0; i < Y_BATCH; i++) {
        for (int j = 0; j < X_BATCH; j++) {
            if (pixelX + j >= width || pixelY + i >= height) {
                continue;
            }

            derivative_t derivative =
                sharedDisparity[LOCAL_INDEX(j, i + 1)] -
                sharedDisparity[LOCAL_INDEX(j, i - 1)];

            output[INDEX(pixelX + j, pixelY + i, outputRowStep)] = derivative;
        }
    }
}

cv::Mat makeHistogram(cv::cuda::GpuMat& derivatives, int histSize = 256) {
    cv::Mat hostDerivatives;
    derivatives.download(hostDerivatives);

    hostDerivatives.convertTo(hostDerivatives, CV_8UC1, 1.0, 127);

    float range[] = {0, 256};  // the upper boundary is exclusive
    const float* histRange[] = {range};

    cv::Mat hist;

    cv::calcHist(&hostDerivatives, 1, 0, cv::Mat(), hist, 1, &histSize, histRange, true, false);

    return hist;
}

namespace cart {
module_result_t DisparityPlaneSegmentationModule::runInternal(System& system, SystemRunData& data) {
    LOG4CXX_DEBUG(this->logger, "Running disparity plane segmentation");
    auto disparity = data.getData<cv::cuda::GpuMat>(CARTSLAM_KEY_DISPARITY);

    if (disparity->empty()) {
        LOG4CXX_WARN(system.logger, "Disparity is empty");
        return MODULE_NO_RETURN_VALUE;
    }

    if (disparity->type() != CV_16SC1) {
        LOG4CXX_WARN(system.logger, "Disparity must be of type CV_16SC1, was " << disparity->type() << " (Depth: " << disparity->depth() << ", channels: " << disparity->channels() << ")");
        throw std::runtime_error("Disparity must be of type CV_16SC1");
    }

    cv::cuda::GpuMat derivatives;
    derivatives.create(disparity->size(), CV_16SC1);

    dim3 threadsPerBlock(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y);
    dim3 numBlocks((disparity->cols + (threadsPerBlock.x * X_BATCH - 1)) / (threadsPerBlock.x * X_BATCH),
                   (disparity->rows + (threadsPerBlock.y * Y_BATCH - 1)) / (threadsPerBlock.y * Y_BATCH));

    LOG4CXX_DEBUG(this->logger, "Launching kernel with " << numBlocks.x << "x" << numBlocks.y << " blocks and "
                                                         << threadsPerBlock.x << "x" << threadsPerBlock.y << " threads");
    LOG4CXX_DEBUG(this->logger, "Shared memory size: " << SHARED_SIZE * 2);

    calculateDerivatives<<<numBlocks, threadsPerBlock>>>(*disparity, derivatives, disparity->cols, disparity->rows);

    CUDA_SAFE_CALL(this->logger, cudaPeekAtLastError());
    CUDA_SAFE_CALL(this->logger, cudaDeviceSynchronize());
    LOG4CXX_DEBUG(this->logger, "Derivatives calculated");

    this->updatePlaneParameters(derivatives, data);

    return MODULE_RETURN(CARTSLAM_KEY_PLANES, boost::make_shared<cv::cuda::GpuMat>(boost::move(derivatives)));
}

void DisparityPlaneSegmentationModule::updatePlaneParameters(cv::cuda::GpuMat& derivatives, SystemRunData& data) {
    if (this->lastUpdatedFrame > 0 && data.id - this->lastUpdatedFrame < this->updateInterval) {
        return;
    }

    this->lastUpdatedFrame = data.id;

    LOG4CXX_DEBUG(this->logger, "Updating plane parameters");

    cv::Mat hostDerivatives;
    derivatives.download(hostDerivatives);
    hostDerivatives = hostDerivatives.reshape(1, hostDerivatives.rows * hostDerivatives.cols);

    cv::Ptr<cv::ml::EM> em = cv::ml::EM::create();
    em->setClustersNumber(2);

    cv::TermCriteria criteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 100, 0.1);
    em->setTermCriteria(criteria);

    CARTSLAM_START_TIMING(em_train);
    em->trainEM(hostDerivatives);
    CARTSLAM_END_TIMING(em_train);

    std::vector<cv::Mat> covs;
    em->getCovs(covs);

    LOG4CXX_DEBUG(this->logger, "Raw means: " << em->getMeans());
    LOG4CXX_DEBUG(this->logger, "Raw covs: " << covs[0].at<double>(0, 0) << ", " << covs[1].at<double>(0, 0));

    this->verticalCenter = ROUND_TO_INT(em->getMeans().at<double>(0, 0));
    this->horizontalCenter = ROUND_TO_INT(em->getMeans().at<double>(1, 0));

    LOG4CXX_DEBUG(this->logger, "Horizontal center: " << this->horizontalCenter << ", vertical center: " << this->verticalCenter);
    LOG4CXX_DEBUG(this->logger, "Limiting variance to " << abs(this->horizontalCenter - this->verticalCenter) / 2);

    // Ensure no overlap between horizontal and vertical variance
    this->verticalVariance = min(abs(this->horizontalCenter - this->verticalCenter) / 2, ROUND_TO_INT(covs[0].at<double>(0, 0)));
    this->horizontalVariance = min(abs(this->horizontalCenter - this->verticalCenter) / 2, ROUND_TO_INT(covs[1].at<double>(0, 0)));

    LOG4CXX_DEBUG(this->logger, "Horizontal variance: " << this->horizontalVariance << ", vertical variance: " << this->verticalVariance);

    this->planeParametersUpdated = true;
}

boost::future<module_result_t> DisparityPlaneSegmentationVisualizationModule::run(System& system, SystemRunData& data) {
    auto promise = boost::make_shared<boost::promise<module_result_t>>();

    boost::asio::post(system.threadPool, [this, promise, &system, &data]() {
        auto planes = data.getData<cv::cuda::GpuMat>(CARTSLAM_KEY_PLANES);

        if (!planes->empty()) {
            // Show image
            cv::Mat image;
            planes->download(image);

            image.convertTo(image, CV_8UC1, 1.0, 127);

            int histSize = 256;

            cv::Mat hist = makeHistogram(*planes, histSize);

            int hist_w = 1024, hist_h = 800;
            int bin_w = cvRound((double)hist_w / histSize);

            cv::Mat histImage(hist_h, hist_w, CV_8UC3, cv::Scalar(0, 0, 0));

            cv::normalize(hist, hist, 0, histImage.rows, cv::NORM_MINMAX, -1);

            for (int i = 1; i < histSize; i++) {
                cv::line(histImage, cv::Point(bin_w * (i - 1), hist_h - cvRound(hist.at<float>(i - 1))),
                         cv::Point(bin_w * (i), hist_h - cvRound(hist.at<float>(i))),
                         cv::Scalar(255, 0, 0), 2, 8, 0);
            }

            this->histThread->setImageIfLater(histImage, data.id);
            this->imageThread->setImageIfLater(image, data.id);
        }

        promise->set_value(MODULE_NO_RETURN_VALUE);
    });

    return promise->get_future();
}
}  // namespace cart