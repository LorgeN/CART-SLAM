#ifndef CARTSLAM_TIMING_HPP
#define CARTSLAM_TIMING_HPP

#define CARTSLAM_END_TIMING(name) CARTSLAM_END_TIMING_TO(name, std::cout)

#ifdef CARTSLAM_TIMING

#include <chrono>
#include <iostream>

#define CARTSLAM_START_TIMING(name) auto name##_start = std::chrono::high_resolution_clock::now()
#define CARTSLAM_END_TIMING_TO(name, write_to)                                                                   \
    auto name##_end = std::chrono::high_resolution_clock::now();                                                 \
    write_to << "TIMING: " << #name << " - "                                                                     \
             << std::chrono::duration_cast<std::chrono::milliseconds>(name##_end - name##_start).count() << "ms" \
             << std::endl

#define CARTSLAM_START_AVERAGE_TIMING(name) \
    int name##_average_count = 0;           \
    long long name##_average_total = 0
#define CARTSLAM_INCREMENT_AVERAGE_TIMING(name)                                                                       \
    name##_average_total += std::chrono::duration_cast<std::chrono::milliseconds>(name##_end - name##_start).count(); \
    name##_average_count++
#define CARTSLAM_END_AVERAGE_TIMING(name)                                                                      \
    std::cout << "TIMING AVERAGE: " << #name << " - " << name##_average_total / name##_average_count << "ms (" \
              << name##_average_count << " runs, " << 1000.0 / (name##_average_total / name##_average_count) << " FPS)" << std::endl

#else

#define CARTSLAM_START_TIMING(name)
#define CARTSLAM_END_TIMING_TO(name, write_to)

#define CARTSLAM_START_AVERAGE_TIMING(name)
#define CARTSLAM_INCREMENT_AVERAGE_TIMING(name)
#define CARTSLAM_END_AVERAGE_TIMING(name)

#endif

#endif  // CARTSLAM_TIMING_HPP