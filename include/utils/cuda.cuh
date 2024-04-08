#ifndef CARTSLAM_CUDA_HPP
#define CARTSLAM_CUDA_HPP

#include <log4cxx/logger.h>

#define CUDA_SAFE_CALL(logger, ans) \
    { cart::gpuAssert((logger), (ans), __FILE__, __LINE__); }

namespace cart {
inline void gpuAssert(log4cxx::LoggerPtr logger, cudaError_t code, const char *file, int line, bool abort = true) {
    if (code == cudaSuccess) {
        return;
    }

    LOG4CXX_ERROR(logger, "An error occurred while performing CUDA operation: " << cudaGetErrorString(code) << " " << file << " " << line);
    if (abort) {
        exit(code);
    }
}
}  // namespace cart

#endif  // CARTSLAM_CUDA_HPP