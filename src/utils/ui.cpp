#include "utils/ui.hpp"

#include <filesystem>
#include <random>

namespace cart {
std::vector<cv::Point2i> getRandomPoints(const int count, const cv::Size &size) {
    std::vector<cv::Point2i> points;

    std::random_device rd;
    std::mt19937 gen(rd());

    std::uniform_int_distribution<int> xDist(0, size.width - 1);
    std::uniform_int_distribution<int> yDist(0, size.height - 1);

    for (int i = 0; i < count; i++) {
        points.push_back(cv::Point2i(xDist(gen), yDist(gen)));
    }

    return points;
}

void ImageThread::stop() {
    this->thread.interrupt();
    this->thread.join();

    cv::destroyAllWindows();

#ifdef CARTSLAM_RECORD_SAMPLES
    for (auto provider : this->providers) {
        auto sharedProvider = provider.lock();
        if (!sharedProvider) {
            continue;
        }

        if (sharedProvider->videoWriter.isOpened()) {
            sharedProvider->videoWriter.release();
            std::cout << "Saved video for " << sharedProvider->name << std::endl;
        }
    }
#endif
}

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

#if defined(CARTSLAM_SAVE_SAMPLES) || defined(CARTSLAM_RECORD_SAMPLES)
    // Check if samples folder exists
    if (!std::filesystem::exists("samples")) {
        std::filesystem::create_directory("samples");
    }
#endif
}

void ImageProvider::setImage(const cv::Mat &image) {
    this->setImageIfLater(image, 0);
}

void ImageProvider::setImageIfLater(const cv::Mat &image, const uint32_t frameIndex) {
#ifdef CARTSLAM_SAVE_SAMPLES
    if (frameIndex % 30 == 0) {
        std::stringstream ss;
        ss << "samples/sample_" << this->name << "_" << frameIndex << ".png";
        cv::imwrite(ss.str(), image);
    }
#endif

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

#ifdef CARTSLAM_RECORD_SAMPLES
            if (!provider->videoWriter.isOpened()) {
                std::stringstream ss;
                ss << "samples/recording_" << provider->name << ".avi";
                if (!provider->videoWriter.open(ss.str(), cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 10, localImage.size())) {
                    LOG4CXX_ERROR(this->logger, "Could not open video writer for " << provider->name);
                }
            }
#endif

            cv::imshow(provider->name, localImage);

#ifdef CARTSLAM_RECORD_SAMPLES
            provider->videoWriter.write(localImage);
#endif
        }

        if (allEmpty) {
            boost::this_thread::sleep_for(boost::chrono::milliseconds(100));
            continue;
        }

        cv::waitKey(25);  // 40 FPS
    }
}
}  // namespace cart