#pragma once

#include <log4cxx/logger.h>

#include <boost/thread.hpp>
#include <opencv2/opencv.hpp>

#include "../logging.hpp"

namespace cart {
std::vector<cv::Point2i> getRandomPoints(const int count, const cv::Size &size);

class ImageProvider;

class ImageThread {
   public:
    static ImageThread &getInstance() {
        static ImageThread instance;
        return instance;
    }

    void appendImageProvider(const boost::weak_ptr<ImageProvider> provider);

    void stop();

   private:
    ImageThread();

    void run();

    std::vector<boost::weak_ptr<ImageProvider>> providers;

    log4cxx::LoggerPtr logger;
    boost::mutex dataMutex;
    boost::thread thread;
};

class ImageProvider : public boost::enable_shared_from_this<ImageProvider> {
   public:
    static boost::shared_ptr<ImageProvider> create(const std::string name) {
        auto ptr = boost::shared_ptr<ImageProvider>(new ImageProvider(name));
        ImageThread::getInstance().appendImageProvider(ptr->weak_from_this());
        return ptr;
    }

    void setImage(const cv::Mat &image);

    void setImageIfLater(const cv::Mat &image, const uint32_t frameIndex);

    const std::string name;

    friend class ImageThread;

   private:
    ImageProvider(const std::string name) : name(name){};

    uint32_t frameIndex = 0;
    cv::Mat image;
    boost::mutex dataMutex;

#ifdef CARTSLAM_RECORD_SAMPLES
    cv::VideoWriter videoWriter;
#endif
};
}  // namespace cart
