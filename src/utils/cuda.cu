#include "utils/cuda.cuh"

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 600
__device__ double atomicAdd(double *address, double val) {
    unsigned long long int *address_as_ull =
        (unsigned long long int *)address;
    unsigned long long int old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val +
                                             __longlong_as_double(assumed)));

        // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);

    return __longlong_as_double(old);
}
#endif

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