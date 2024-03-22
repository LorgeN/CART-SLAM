#ifndef CARTSLAM_UI_HPP
#define CARTSLAM_UI_HPP

#include <log4cxx/logger.h>

#include <boost/thread.hpp>
#include <opencv2/opencv.hpp>

#include "../logging.hpp"

namespace cart {
class ImageThread {
   public:
    ImageThread(const std::string name) : name(name) {
        this->logger = getLogger(name);
    };

    ~ImageThread() {
        // Stop the thread only if it is still running
        if (!this->thread.joinable()) {
            return;
        }

        this->thread.interrupt();
        this->thread.join();
    }

    void setImage(const cv::Mat &image);

    void setImageIfLater(const cv::Mat &image, const uint32_t frameIndex);

    const std::string name;

   private:
    void run();

    log4cxx::LoggerPtr logger;
    cv::Mat image;
    uint32_t frameIndex = 0;
    boost::mutex dataMutex;
    boost::thread thread;
};
}  // namespace cart

#endif  // CARTSLAM_UI_HPP