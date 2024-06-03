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

struct DeviceColor {
    uint8_t color[3];
};

__constant__ DeviceColor COLOR_WHEEL_DEVICE[NCOLS];

void copyColorWheelToDevice(cudaStream_t &stream) {
    const util::ColorWheel colorWheel;
    cudaMemcpyToSymbolAsync(COLOR_WHEEL_DEVICE, colorWheel.colorWheel, NCOLS * sizeof(DeviceColor), 0, cudaMemcpyHostToDevice, stream);
}

__device__ void assignColor(float fx, float fy, uint8_t *pix) {
    const float rad = sqrt(fx * fx + fy * fy);
    const float a = atan2(-fy, -fx) / (float)M_PI;

    const float fk = (a + 1.0f) / 2.0f * (NCOLS - 1);
    const int k0 = static_cast<int>(fk);
    const int k1 = (k0 + 1) % NCOLS;
    const float f = fk - k0;

    for (int b = 0; b < 3; b++) {
        const float col0 = COLOR_WHEEL_DEVICE[k0].color[b] / 255.0f;
        const float col1 = COLOR_WHEEL_DEVICE[k1].color[b] / 255.0f;

        float col = (1 - f) * col0 + f * col1;

        if (rad <= 1)
            col = 1 - rad * (1 - col);  // increase saturation with radius
        else
            col *= .75;  // out of range

        pix[2 - b] = static_cast<uint8_t>(255.0 * col);
    }
}
}  // namespace cart