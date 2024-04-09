#include "utils/ui.hpp"

namespace cart {
void ImageThread::appendImageProvider(const boost::weak_ptr<ImageProvider> provider) {
    boost::lock_guard<boost::mutex> lock(this->dataMutex);
    auto sharedProvider = provider.lock();
    if (!sharedProvider) {
        return;
    }

    this->providers.push_back(provider);
    LOG4CXX_DEBUG(this->logger, "Registered provider " << sharedProvider->name);
}

ImageThread::ImageThread() {
    this->logger = getLogger("UI Thread");

    LOG4CXX_DEBUG(this->logger, "Starting thread");
    this->thread = boost::thread(boost::bind(&ImageThread::run, this));
}

void ImageProvider::setImage(const cv::Mat &image) {
    this->setImageIfLater(image, 0);
}

void ImageProvider::setImageIfLater(const cv::Mat &image, const uint32_t frameIndex) {
    boost::lock_guard<boost::mutex> lock(this->dataMutex);
    // If the frame index is less than the current frame index, ignore
    // Do not strictly require higher frame index, so we can reuse the same as in #setImage
    if (frameIndex < this->frameIndex) {
        return;
    }

    this->frameIndex = frameIndex;
    this->image = image;
}

void ImageThread::run() {
    cv::Mat localImage;

    while (true) {
        try {
            boost::this_thread::interruption_point();
        } catch (boost::thread_interrupted &) {
            LOG4CXX_DEBUG(this->logger, "Thread has been interrupted");
            return;
        }

        std::vector<boost::shared_ptr<ImageProvider>> providers;

        {
            boost::lock_guard<boost::mutex> lock(this->dataMutex);

            if (!this->providers.empty()) {
                auto it = this->providers.begin();

                while (it != this->providers.end()) {
                    auto sharedProvider = it->lock();
                    if (!sharedProvider) {
                        it = this->providers.erase(it);
                        continue;
                    }

                    providers.push_back(sharedProvider);
                    it++;
                }
            }
        }

        if (providers.size() == 0) {
            boost::this_thread::sleep_for(boost::chrono::milliseconds(100));
            continue;
        }

        bool allEmpty = true;
        for (auto provider : providers) {
            {
                boost::lock_guard<boost::mutex> lock(provider->dataMutex);
                if (provider->image.empty()) {
                    continue;
                }

                allEmpty = false;
                localImage = provider->image.clone();
            }

            cv::imshow(provider->name, localImage);
        }

        if (allEmpty) {
            boost::this_thread::sleep_for(boost::chrono::milliseconds(100));
            continue;
        }

        cv::waitKey(25);  // 40 FPS
    }
}
}  // namespace cart