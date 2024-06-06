#pragma once

#include <boost/make_shared.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/thread/lock_guard.hpp>
#include <boost/thread/mutex.hpp>
#include <chrono>
#include <ctime>
#include <fstream>
#include <iostream>
#include <string>

#include "logging.hpp"
#include "utils/csv.hpp"

namespace cart::timing {
struct timing_handle_t {
    const std::string name;
    const std::chrono::time_point<std::chrono::high_resolution_clock> init;
    std::chrono::time_point<std::chrono::high_resolution_clock> start;
    std::chrono::time_point<std::chrono::high_resolution_clock> end;

    timing_handle_t(const std::string name, const std::chrono::time_point<std::chrono::high_resolution_clock> init)
        : name(name), init(init) {}
};

inline const std::string timeToString(const std::chrono::time_point<std::chrono::system_clock> &time) {
    std::time_t t = std::chrono::system_clock::to_time_t(time);
    char buf[21];
    strftime(buf, 20, "%d.%m.%Y %H:%M:%S", localtime(&t));
    return std::string(buf);
}

inline const long long getMilliseconds(const std::chrono::time_point<std::chrono::high_resolution_clock> &time) {
    return std::chrono::duration_cast<std::chrono::milliseconds>(time.time_since_epoch()).count();
}

// Use current time to generate a unique filename
inline const std::string generateFileName() {
    return "timing-" + timeToString(std::chrono::system_clock::now()) + ".csv";
}

inline boost::shared_ptr<timing_handle_t> initTiming(const std::string &name) {
    return boost::make_shared<timing_handle_t>(name, std::chrono::high_resolution_clock::now());
}

inline void startTiming(boost::shared_ptr<timing_handle_t> handle) {
    handle->start = std::chrono::high_resolution_clock::now();
}

inline void endTiming(boost::shared_ptr<timing_handle_t> handle) {
    static utils::csvfile timingFile(generateFileName(), ";", std::vector<std::string>{"name", "time_init", "time_start", "time_end", "duration_ms"});
    static boost::mutex mutex;

    handle->end = std::chrono::high_resolution_clock::now();

    // Write to file
    {
        boost::lock_guard<boost::mutex> lock(mutex);
        timingFile << handle->name << getMilliseconds(handle->init) << getMilliseconds(handle->start) << getMilliseconds(handle->end)
                   << std::chrono::duration_cast<std::chrono::milliseconds>(handle->end - handle->start).count() << utils::endrow;
    }
}
}  // namespace cart::timing
