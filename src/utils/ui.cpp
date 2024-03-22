#include "utils/ui.hpp"

namespace cart {
void ImageThread::setImage(const cv::Mat &image) {
    this->setImageIfLater(image, 0);
}

void ImageThread::setImageIfLater(const cv::Mat &image, const uint32_t frameIndex) {
    boost::lock_guard<boost::mutex> lock(this->dataMutex);
    // If the frame index is less than the current frame index, ignore
    // Do not strictly require higher frame index, so we can reuse the same as in #setImage
    if (frameIndex < this->frameIndex) {
        return;
    }

    this->frameIndex = frameIndex;
    this->image = image;

    if (this->thread.joinable()) {
        return;
    }

    this->thread = boost::thread(boost::bind(&ImageThread::run, this));
}

void ImageThread::run() {
    cv::Mat localImage;
    while (true) {
        try {
            boost::this_thread::interruption_point();
        } catch (boost::thread_interrupted &) {
            LOG4CXX_DEBUG(this->logger, "Thread " << this->name << " has been interrupted");
            return;
        }

        {
            boost::lock_guard<boost::mutex> lock(this->dataMutex);
            if (this->image.empty()) {
                continue;
            }

            localImage = this->image.clone();
        }

        cv::imshow(this->name, localImage);
        cv::waitKey(25);  // 40 FPS
    }
}
}  // namespace cart