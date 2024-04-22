#include "utils/cuda.cuh"

namespace cart {
void reportMemoryUsage(log4cxx::LoggerPtr logger) {
    size_t free_byte, total_byte;

    CUDA_SAFE_CALL(logger, cudaMemGetInfo(&free_byte, &total_byte));

    double free_db = (double)free_byte;
    double total_db = (double)total_byte;
    double used_db = total_db - free_db;

    LOG4CXX_INFO(logger, "GPU memory usage: used = " << used_db / 1024 / 1024 << " MB, free = " << free_db / 1024 / 1024 << " MB, total = " << total_db / 1024 / 1024 << " MB");
}
}  // namespace cart